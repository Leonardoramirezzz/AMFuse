"""
test_m1_m2.py — Prueba de integración: dataset → M1 → M2

Verifica que el flujo completo de los primeros dos módulos funciona
con datos reales de CMU-MOSEI.

Ejecutar desde la raíz del proyecto:
    python test_m1_m2.py

Estructura de carpetas esperada:
    amfuse/
    ├── amfuse/
    │   ├── dataset.py
    │   └── modules/
    │       ├── m1_status.py
    │       └── m2_encoders.py
    ├── config.py
    └── data/
        └── mosei_aligned.pkl
"""

import torch
import sys
import os

# Asegurar que Python encuentra los módulos del proyecto
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from dataset import get_dataloaders
from amfuse.modules.m1_status import ModalityStatusEncoder
from amfuse.modules.m2_encoders import UnimodalEncoders

# ── Dispositivo ──────────────────────────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Dispositivo: {device}")
if device.type == 'cuda':
    print(f"GPU        : {torch.cuda.get_device_name(0)}\n")
else:
    print()

# ── Cargar dataset ───────────────────────────────────────────────────────────
print("=" * 55)
print("PASO 1 — Cargar DataLoader")
print("=" * 55)

loaders = get_dataloaders(
    pkl_path     = 'data/mosei_aligned.pkl',
    batch_size   = 8,           # batch pequeño para la prueba
    missing_rate = 0.3,         # 30% modality dropout en train
)

# Tomar un batch real de entrenamiento
batch = next(iter(loaders['train']))

print("\nBatch de train (con modality dropout 30%):")
for key, val in batch.items():
    if isinstance(val, torch.Tensor):
        print(f"  {key:6s}: {tuple(val.shape)}  dtype={val.dtype}")
    elif val is None:
        print(f"  {key:6s}: None  ← eliminada por modality dropout")
    else:
        print(f"  {key:6s}: {type(val).__name__}")

# ── Instanciar módulos ───────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("PASO 2 — Instanciar M1 y M2")
print("=" * 55)

m1 = ModalityStatusEncoder()
m2 = UnimodalEncoders(
    text_dim   = config.TEXT_DIM,    # 768
    audio_dim  = config.AUDIO_DIM,   # 74
    video_dim  = config.VIDEO_DIM,   # 35  ← tu versión de FACET
    latent_dim = config.LATENT_DIM,  # 256
).to(device)

print(f"\n{m1}\n")
print(f"{m2}\n")

# ── Mover el batch al dispositivo ────────────────────────────────────────────
# Solo los tensores no-None se mueven a GPU
def to_device(x):
    return x.to(device) if x is not None else None

text  = to_device(batch['text'])     # [B, 768]    o None
audio = to_device(batch['audio'])    # [B, 50, 74] o None
video = to_device(batch['video'])    # [B, 50, 35] o None

# El dataloader produce texto como [B, 768] (ya promediado en el pkl)
# M2 espera [B, T, dim] para hacer mean(dim=1)
# → Añadimos dimensión temporal artificial si el texto viene plano

if text is not None and text.dim() == 2:
    text = text.unsqueeze(1)    # [B, 768] → [B, 1, 768]

if audio is not None and audio.dim() == 2:
    audio = audio.unsqueeze(1)

if video is not None and video.dim() == 2:
    video = video.unsqueeze(1)

# ── M1: codificador de estado ─────────────────────────────────────────────────
print("=" * 55)
print("PASO 3 — Módulo 1: codificador de estado")
print("=" * 55)

m_vec, tau = m1(text, audio, video)

print(f"\n  m   = {m_vec.tolist()}   (1=presente, 0=ausente)")
print(f"  tau = {tau}")
print(f"  modalidades disponibles : {m1.available_modalities(m_vec)}")
print(f"  modalidades faltantes   : {m1.missing_modalities(m_vec)}")

modal_names = ['texto', 'audio', 'video']
for idx, (present, tipo) in enumerate(zip(m_vec.tolist(), tau)):
    estado = f"✓ presente ({tipo})" if present else "✗ ausente"
    print(f"    [{idx}] {modal_names[idx]:6s}: {estado}")

# ── M2: codificadores unimodales ──────────────────────────────────────────────
print("\n" + "=" * 55)
print("PASO 4 — Módulo 2: codificadores unimodales")
print("=" * 55)

embeddings = m2(text, audio, video, m_vec)

print("\nEmbeddings proyectados a ℝ²⁵⁶:")
for modal, emb in embeddings.items():
    if emb is not None:
        print(f"  {modal:6s}: {tuple(emb.shape)}  "
              f"mean={emb.mean().item():.4f}  "
              f"std={emb.std().item():.4f}")
    else:
        print(f"  {modal:6s}: None  ← pendiente de generación en M3")

# ── Verificar gradientes ──────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("PASO 5 — Verificación de gradientes")
print("=" * 55)

available_embeddings = [e for e in embeddings.values() if e is not None]

if len(available_embeddings) == 0:
    print("\n  ⚠ Todos los embeddings son None en este batch.")
    print("  Tomando un nuevo batch con modalidades forzadas...")

    # Tomar batch del loader de validación (sin dropout, siempre completo)
    val_batch = next(iter(loaders['valid']))
    text_v  = to_device(val_batch['text']).unsqueeze(1)   if val_batch['text']  is not None else None
    audio_v = to_device(val_batch['audio'])                if val_batch['audio'] is not None else None
    video_v = to_device(val_batch['video'])                if val_batch['video'] is not None else None

    m_vec_v, _ = m1(text_v, audio_v, video_v)
    embeddings  = m2(text_v, audio_v, video_v, m_vec_v)
    available_embeddings = [e for e in embeddings.values() if e is not None]
    print(f"  Nuevo batch: {sum(1 for e in embeddings.values() if e is not None)}/3 modalidades")

# Pérdida ficticia para verificar backprop
loss = sum(e.mean() for e in available_embeddings)
loss.backward()

print("\nGradientes en capas de M2:")
for name, param in m2.named_parameters():
    has_grad  = param.grad is not None
    grad_norm = param.grad.norm().item() if has_grad else 0.0
    status    = f"✓  norm={grad_norm:.4f}" if has_grad else "✗  SIN gradiente"
    print(f"  {name:35s}: {status}")

# ── Resumen de un batch de val (sin dropout) ──────────────────────────────────
print("\n" + "=" * 55)
print("PASO 6 — Batch de validación (sin modality dropout)")
print("=" * 55)

val_batch = next(iter(loaders['valid']))
print("\nBatch de valid (missing_rate=0.0, todas presentes):")
for key, val in val_batch.items():
    if isinstance(val, torch.Tensor):
        print(f"  {key:6s}: {tuple(val.shape)}")
    elif val is None:
        print(f"  {key:6s}: None  ← inesperado en validación")
    else:
        print(f"  {key:6s}: {type(val).__name__}")

# ── Resultado final ───────────────────────────────────────────────────────────
print("\n" + "=" * 55)
n_present = sum(1 for e in embeddings.values() if e is not None)
n_missing = 3 - n_present
print(f"✓ M1 y M2 funcionando correctamente")
print(f"  Modalidades presentes en este batch : {n_present}/3")
print(f"  Modalidades a generar en M3         : {n_missing}/3")
print(f"  Output de M2: dict con embeddings ∈ ℝ²⁵⁶ para modalidades presentes")
print("=" * 55)
print("\nPróximo paso: implementar M3 (generación de faltantes + confiabilidad)")