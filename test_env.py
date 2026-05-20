"""
Verificación del entorno AMFuse
Ejecutar antes de comenzar la implementación:
    python test_env.py
"""

import sys

# ── Python ──────────────────────────────────────────────────────────────────
print("=" * 55)
print("VERIFICACIÓN DE ENTORNO AMFuse")
print("=" * 55)

print(f"\n[1] Python: {sys.version}")
major, minor = sys.version_info[:2]
if (major, minor) == (3, 10):
    print("    ✓ Python 3.10 OK")
else:
    print(f"    ⚠ Se recomienda Python 3.10 (tienes {major}.{minor})")

# ── PyTorch + CUDA ───────────────────────────────────────────────────────────
print("\n[2] PyTorch y CUDA:")
try:
    import torch
    print(f"    versión PyTorch : {torch.__version__}")
    print(f"    versión CUDA    : {torch.version.cuda}")

    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        mem  = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"    ✓ GPU detectada : {name}")
        print(f"    ✓ VRAM total    : {mem:.1f} GB")

        # prueba real: tensor en GPU
        x = torch.randn(512, 512, device='cuda')
        y = x @ x.T
        print(f"    ✓ operación en GPU OK  (shape resultado: {y.shape})")
        del x, y
        torch.cuda.empty_cache()
    else:
        print("    ✗ GPU NO detectada — PyTorch usará CPU")
        print("      Solución: instala PyTorch nightly con CUDA 12.8")
        print("      pip install --pre torch torchvision torchaudio \\")
        print("          --index-url https://download.pytorch.org/whl/nightly/cu128")

except ImportError:
    print("    ✗ PyTorch no instalado")

# ── HuggingFace Transformers ─────────────────────────────────────────────────
print("\n[3] HuggingFace Transformers:")
try:
    import transformers
    print(f"    ✓ versión: {transformers.__version__}")

    # prueba: cargar tokenizador BERT (sin descargar el modelo completo)
    from transformers import BertTokenizer
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    tokens = tokenizer("Verificación de entorno AMFuse", return_tensors="pt")
    print(f"    ✓ BertTokenizer OK  (tokens shape: {tokens['input_ids'].shape})")

except ImportError:
    print("    ✗ transformers no instalado")
except Exception as e:
    print(f"    ⚠ transformers instalado pero error al cargar BERT: {e}")
    print("      Puede ser que no hayas descargado el modelo aún — es normal")

# ── Librerías científicas ─────────────────────────────────────────────────────
print("\n[4] Librerías científicas:")
libs = {
    'numpy':        'numpy',
    'pandas':       'pandas',
    'sklearn':      'scikit-learn',
    'matplotlib':   'matplotlib',
    'tqdm':         'tqdm',
}
for import_name, pip_name in libs.items():
    try:
        mod = __import__(import_name)
        ver = getattr(mod, '__version__', 'OK')
        print(f"    ✓ {pip_name}: {ver}")
    except ImportError:
        print(f"    ✗ {pip_name} no instalado  →  pip install {pip_name}")

# ── mmsdk ────────────────────────────────────────────────────────────────────
print("\n[5] CMU-MultimodalSDK (mmsdk):")
try:
    import mmsdk
    print("    ✓ mmsdk importado correctamente")
except ImportError:
    print("    ✗ mmsdk no encontrado")
    print("      Solución: xcopy CMU-MultimodalSDK\\mmsdk mmsdk /E /I")

# ── Resumen ──────────────────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("Si todos los ítems muestran ✓ estás listo para")
print("comenzar la implementación de AMFuse.")
print("=" * 55)