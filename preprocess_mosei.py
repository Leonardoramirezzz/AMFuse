"""
preprocess_mosei.py — Unifica BERT, COVAREP y FACET en un único .pkl limpio

Correcciones aplicadas:
    - Detecta automáticamente las dimensiones reales de audio y video
      (tu FACET tiene 35 dim, no 47 como en la versión estándar)
    - Alinea BERT con COVAREP/FACET por posición usando solo las
      primeras N muestras de COVAREP/FACET donde N = len(BERT)
      (BERT tiene 22860, COVAREP/FACET tienen 23248 — diferencia de 388
      utterances filtradas en el preprocesamiento de BERT)

Ejecutar desde la raíz del proyecto:
    python preprocess_mosei.py
"""

import pickle
import numpy as np
from tqdm import tqdm

# ── Configuración ────────────────────────────────────────────────────────────
MAX_SEQ_LEN = 50
DATA_DIR    = 'data'

BERT_PATH    = f'{DATA_DIR}/BERT_MOSEI.pkl'
COVAREP_PATH = f'{DATA_DIR}/COAVAREP_aligned_MOSEI.pkl'
FACET_PATH   = f'{DATA_DIR}/FACET_aligned_MOSEI.pkl'
OUTPUT_PATH  = f'{DATA_DIR}/mosei_aligned.pkl'

# ── Utilidades ───────────────────────────────────────────────────────────────

def pad_or_truncate(arr: np.ndarray, max_len: int) -> np.ndarray:
    """Ajusta [T, D] → [max_len, D] con padding de ceros o truncado."""
    T, D = arr.shape
    if T >= max_len:
        return arr[:max_len, :]
    padded = np.zeros((max_len, D), dtype=arr.dtype)
    padded[:T, :] = arr
    return padded


def extract_cs_data(cs_obj) -> dict:
    """Extrae dict { 'videoID[seg]': ndarray[T, D] } de un computational_sequence."""
    return {key: val['features'] for key, val in cs_obj.data.items()}


# ── Carga ────────────────────────────────────────────────────────────────────

print("Cargando BERT_MOSEI.pkl ...")
with open(BERT_PATH, 'rb') as f:
    bert_raw = pickle.load(f)

bert_tensor = bert_raw['Data'].numpy()    # [N_bert, 768]
labels      = bert_raw['level'].numpy()   # [N_bert]
N_bert      = bert_tensor.shape[0]
print(f"  BERT : {bert_tensor.shape}  etiquetas: {labels.shape}")

print("\nCargando COAVAREP_aligned_MOSEI.pkl ...")
with open(COVAREP_PATH, 'rb') as f:
    covarep_cs = pickle.load(f)
covarep_dict = extract_cs_data(covarep_cs)
print(f"  COVAREP: {len(covarep_dict)} utterances")

print("\nCargando FACET_aligned_MOSEI.pkl ...")
with open(FACET_PATH, 'rb') as f:
    facet_cs = pickle.load(f)
facet_dict = extract_cs_data(facet_cs)
print(f"  FACET  : {len(facet_dict)} utterances")

# ── Detectar dimensiones reales ──────────────────────────────────────────────

first_covarep_key = sorted(covarep_dict.keys())[0]
first_facet_key   = sorted(facet_dict.keys())[0]

AUDIO_DIM = covarep_dict[first_covarep_key].shape[1]
VIDEO_DIM = facet_dict[first_facet_key].shape[1]

print(f"\nDimensiones detectadas automáticamente:")
print(f"  TEXT_DIM  = {bert_tensor.shape[1]}  (BERT)")
print(f"  AUDIO_DIM = {AUDIO_DIM}  (COVAREP)")
print(f"  VIDEO_DIM = {VIDEO_DIM}  (FACET)")

# ── Alineación ───────────────────────────────────────────────────────────────
# BERT tiene N_bert=22860 muestras indexadas por posición (sin IDs).
# COVAREP/FACET tienen 23248 utterances con IDs.
#
# Estrategia: tomar las primeras N_bert keys (orden alfabético determinístico)
# de COVAREP/FACET para alinearlas con las N_bert muestras de BERT.
# Esto asume que BERT fue generado sobre el mismo subconjunto en el mismo orden.

all_keys    = sorted(covarep_dict.keys())   # orden determinístico
facet_keys  = set(facet_dict.keys())
common_keys = [k for k in all_keys if k in facet_keys]

print(f"\nAlineación:")
print(f"  Keys comunes (COVAREP ∩ FACET): {len(common_keys)}")
print(f"  Muestras BERT                 : {N_bert}")

if len(common_keys) >= N_bert:
    # Tomar solo las primeras N_bert keys para alinear con BERT
    aligned_keys = common_keys[:N_bert]
    N_use = N_bert
    print(f"  Usando primeras {N_use} keys de COVAREP/FACET para alinear con BERT")
else:
    N_use = len(common_keys)
    aligned_keys = common_keys
    bert_tensor = bert_tensor[:N_use]
    labels      = labels[:N_use]
    print(f"  ⚠ COVAREP/FACET tienen menos keys que BERT — usando {N_use} muestras")

# ── Construcción de arrays ───────────────────────────────────────────────────

print(f"\nProcesando {N_use} utterances (max_seq_len={MAX_SEQ_LEN}) ...")

text_arr  = bert_tensor.astype(np.float32)                         # [N, 768]
audio_arr = np.zeros((N_use, MAX_SEQ_LEN, AUDIO_DIM), np.float32) # [N, 50, 74]
video_arr = np.zeros((N_use, MAX_SEQ_LEN, VIDEO_DIM), np.float32) # [N, 50, D]
label_arr = labels.astype(np.float32)                              # [N]

for i, key in enumerate(tqdm(aligned_keys)):
    audio_raw = covarep_dict[key].astype(np.float32)   # [T, AUDIO_DIM]
    video_raw = facet_dict[key].astype(np.float32)     # [T, VIDEO_DIM]

    audio_arr[i] = pad_or_truncate(audio_raw, MAX_SEQ_LEN)
    video_arr[i] = pad_or_truncate(video_raw, MAX_SEQ_LEN)

# ── División train / val / test (70 / 10 / 20) ──────────────────────────────

n_train = int(N_use * 0.70)
n_val   = int(N_use * 0.10)

splits = {
    'train': {
        'text'  : text_arr[:n_train],
        'audio' : audio_arr[:n_train],
        'video' : video_arr[:n_train],
        'labels': label_arr[:n_train],
        'keys'  : aligned_keys[:n_train],
        'audio_dim': AUDIO_DIM,
        'video_dim': VIDEO_DIM,
    },
    'valid': {
        'text'  : text_arr[n_train:n_train+n_val],
        'audio' : audio_arr[n_train:n_train+n_val],
        'video' : video_arr[n_train:n_train+n_val],
        'labels': label_arr[n_train:n_train+n_val],
        'keys'  : aligned_keys[n_train:n_train+n_val],
        'audio_dim': AUDIO_DIM,
        'video_dim': VIDEO_DIM,
    },
    'test': {
        'text'  : text_arr[n_train+n_val:],
        'audio' : audio_arr[n_train+n_val:],
        'video' : video_arr[n_train+n_val:],
        'labels': label_arr[n_train+n_val:],
        'keys'  : aligned_keys[n_train+n_val:],
        'audio_dim': AUDIO_DIM,
        'video_dim': VIDEO_DIM,
    },
}

# ── Guardar ──────────────────────────────────────────────────────────────────

print(f"\nGuardando en {OUTPUT_PATH} ...")
with open(OUTPUT_PATH, 'wb') as f:
    pickle.dump(splits, f, protocol=4)

# ── Resumen ──────────────────────────────────────────────────────────────────

print("\n" + "="*55)
print("RESUMEN")
print("="*55)
for split_name, split_data in splits.items():
    n = len(split_data['labels'])
    print(f"\n  {split_name.upper()} ({n} muestras):")
    print(f"    text  : {split_data['text'].shape}")
    print(f"    audio : {split_data['audio'].shape}")
    print(f"    video : {split_data['video'].shape}")
    print(f"    labels: {split_data['labels'].shape}")
    print(f"    label min={split_data['labels'].min():.2f}  "
          f"max={split_data['labels'].max():.2f}  "
          f"mean={split_data['labels'].mean():.2f}")

print(f"\n✓ Guardado en {OUTPUT_PATH}")
print(f"\nActualiza config.py con las dimensiones detectadas:")
print(f"  AUDIO_DIM = {AUDIO_DIM}")
print(f"  VIDEO_DIM = {VIDEO_DIM}")
print(f"\nPróximo paso: python amfuse/dataset.py")