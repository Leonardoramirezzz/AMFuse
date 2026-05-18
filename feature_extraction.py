import os
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from PIL import Image
from tqdm import tqdm
from datetime import datetime

# === CONFIGURACIÓN ===


#---------------------------------------------------------------
#---------------------   MVSA - MULTIPLE   ---------------------
#--------------------------------------------------------------- 

# entrenamiento imagenes del training_set
#IMAGE_DIR = r"E:\Leo_Semestre_X\PFC1\datasets\itin\MVSA-Multiple\training_set"
#OUTPUT_DIR = r"E:\Leo_Semestre_X\PFC1\ITIN3\features\multiple\training\image"
#SUMMARY_PATH = os.path.join(OUTPUT_DIR, "summary_multiple_training.txt")

# entrenamiento imagenes del validation_set
#IMAGE_DIR = r"E:\Leo_Semestre_X\PFC1\datasets\itin\MVSA-Multiple\validation_set"
#OUTPUT_DIR = r"E:\Leo_Semestre_X\PFC1\ITIN3\features\multiple\validation\image"
#SUMMARY_PATH = os.path.join(OUTPUT_DIR, "summary_multiple_validation.txt")


# entenamiento imagenes del test_set
IMAGE_DIR = r"E:\Leo_Semestre_X\PFC1\datasets\itin\MVSA-Multiple\test_set"
OUTPUT_DIR = r"E:\Leo_Semestre_X\PFC1\ITIN3\features\multiple\test\image"
SUMMARY_PATH = os.path.join(OUTPUT_DIR, "summary_multiple_test.txt")


#---------------------------------------------------------------
#---------------------    MVSA - SINGLE    ---------------------
#--------------------------------------------------------------- 

'''
# entrenamiento imagenes del training_set

#IMAGE_DIR = r"E:\Leo_Semestre_X\PFC1\datasets\itin\MVSA-Single\training_set"
#OUTPUT_DIR = r"E:\Leo_Semestre_X\PFC1\ITIN3\features\single\training\image"
#SUMMARY_PATH = os.path.join(OUTPUT_DIR, "summary_single_training.txt")


# entrenamiento imagenes del validation_set

#IMAGE_DIR = r"E:\Leo_Semestre_X\PFC1\datasets\itin\MVSA-Single\validation_set"
#OUTPUT_DIR = r"E:\Leo_Semestre_X\PFC1\ITIN3\features\single\validation\image"
#SUMMARY_PATH = os.path.join(OUTPUT_DIR, "summary_single_validation.txt").



# entrenamiento imagenes del test_set

#IMAGE_DIR = r"E:\Leo_Semestre_X\PFC1\datasets\itin\MVSA-Single\test_set"
#OUTPUT_DIR = r"E:\Leo_Semestre_X\PFC1\ITIN3\features\single\test\image"
#SUMMARY_PATH = os.path.join(OUTPUT_DIR, "summary_single_test.txt").


'''

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Parámetros

M = 2           # máximo de regiones a proponer
D = 512         # dimensión de la proyección lineal
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# cargar modelo
print("Cargando modelo Faster R-CNN")
model = fasterrcnn_resnet50_fpn(pretrained=True).to(DEVICE)
model.eval()

# capa de proyección lineal
projection = nn.Linear(2048, D).to(DEVICE)
torch.nn.init.xavier_uniform_(projection.weight)
torch.nn.init.zeros_(projection.bias)


transform = transforms.Compose([
    transforms.ToTensor(),
])

# log
with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
    f.write(f"=== ITIN Image Feature Extraction Summary ===\n")
    f.write(f"Fecha de inicio: {datetime.now()}\n")
    f.write(f"Dataset: MVSA-Single\n")
    f.write(f"Backbone: Faster R-CNN (resnet50 preentrenado)\n")
    f.write(f"m = {M}, d = {D}\n")
    f.write("="*60 + "\n\n")

all_images = [f for f in os.listdir(IMAGE_DIR) if f.lower().endswith(".jpg")]
all_images.sort(key=lambda x: int(os.path.splitext(x)[0]))

procesadas = []
falladas = []

print("Iniciando extracción de características...\n")

# loop de extracción
for img_file in tqdm(all_images, desc="Procesando imágenes"):
    img_id = os.path.splitext(img_file)[0]
    out_path = os.path.join(OUTPUT_DIR, f"{img_id}.pt")

    if os.path.exists(out_path):
        procesadas.append(img_id)
        continue

    try:
        img_path = os.path.join(IMAGE_DIR, img_file)
        image = Image.open(img_path).convert("RGB")
        img_tensor = transform(image).to(DEVICE)

        # detección de regiones
        with torch.no_grad():
            outputs = model([img_tensor])[0]

        boxes = outputs["boxes"]
        scores = outputs["scores"]

        if len(scores) == 0:
            print(f"!!!!!No se detectaron regiones en {img_id}")
            falladas.append(img_id)
            continue

        top_indices = torch.argsort(scores, descending=True)[:M]
        top_boxes = boxes[top_indices]

        # === EXTRAER FEATURES DE ROI HEADS ===
        with torch.no_grad():
            features = model.backbone(img_tensor.unsqueeze(0))
            box_features = model.roi_heads.box_roi_pool(features, [top_boxes], [img_tensor.shape[1:]])
            box_features = model.roi_heads.box_head(box_features)  # [M, 1024]

        # padding si es necesario
        if box_features.shape[1] != 2048:
            box_features = torch.cat([box_features, box_features], dim=1)

        # proyección final -> D dimensiones
        with torch.no_grad():
            r_i = projection(box_features)  # [M, D]

        # === GUARDAR ===
        torch.save(r_i.cpu(), out_path)
        procesadas.append(img_id)

    except Exception as e:
        print(f"!!!!!Error procesando {img_id}: {e}")
        falladas.append(f"{img_id} ({e})")

# === GUARDAR LOG ===
with open(SUMMARY_PATH, "a", encoding="utf-8") as f:
    f.write(f"\n=== FINALIZADO ===\n")
    f.write(f"Fecha de fin: {datetime.now()}\n\n")
    f.write(f"Total de imágenes: {len(all_images)}\n")
    f.write(f"Procesadas correctamente: {len(procesadas)}\n")
    f.write(f"Falladas: {len(falladas)}\n\n")

    f.write("=== IDs procesadas ===\n")
    f.write(", ".join(procesadas) + "\n\n")

    if falladas:
        f.write("=== IDs falladas ===\n")
        for fail in falladas:
            f.write(str(fail) + "\n")

print("\n-----Extracción completada.")
print(f"Características guardadas en: {OUTPUT_DIR}")
print(f"Resumen detallado en: {SUMMARY_PATH}")
