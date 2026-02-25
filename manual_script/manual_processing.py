"""
Script de procesamiento manual local.
Ejecuta el pipeline de granulometría (pasos 2 al 7) sobre imágenes en una carpeta local.
"""
import logging
import sys
from pathlib import Path
import cv2
import matplotlib.pyplot as plt

# Aseguramos que el directorio raíz esté en el path para los imports
sys.path.append(str(Path(__file__).parent.parent))

from config.settings import PATH_MODEL
from src.utils import load_model, load_camera_params
from src.domain.granulometry.models import RosinRammler
from src.nodes.ia import OnnxInferenceNode, OnnxPostProcessingNode
from src.nodes.granulometry import (FixedDistanceTransformer,
                                    GranulometryStatsNode, GranulometryModelNode)
from src.nodes.viz import ResultVisualizerNode
from src.nodes.base import PipelineNode 

# Configuración de carpetas
INPUT_DIR = Path(__file__).parent.parent / "manual_script" / "images"
OUTPUT_DIR = Path(__file__).parent.parent / "manual_script" / "results"

def main():
    # Configuración de logs
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    logging.info(">>> Iniciando script de procesamiento manual...")

    # 1. Verificar directorios
    if not INPUT_DIR.exists():
        logging.error(f"No se encontró la carpeta de entrada: {INPUT_DIR.absolute()}")
        logging.info("Por favor crea la carpeta 'images' y coloca las fotografías ahí.")
        return

    OUTPUT_DIR.mkdir(exist_ok=True)

    # 2. Cargar Modelo y Parámetros
    try:
        logging.info(f"Cargando modelo ONNX desde: {PATH_MODEL}")
        model = load_model(PATH_MODEL)
        
        logging.info("Cargando parámetros de cámara...")
        camera_params = load_camera_params()
    except Exception as e:
        logging.critical(f"Error fatal cargando recursos: {e}")
        return

    # 3. Construir los nodos del Pipeline (Pasos 2 al 7)
    # Se replican los nodos usados en src/pipeline.py excluyendo descarga y subida.
    nodes = [
        # Paso 2: Inferencia
        OnnxInferenceNode(onnx_session=model, name="IA_Inference"),
        
        # Paso 3: Post-procesamiento (Mask -> Polygons)
        OnnxPostProcessingNode(name="PostProcessing"),
        
        # Paso 4: Transformación (Pixels -> mm/inch)
        FixedDistanceTransformer(camera_params=camera_params, name="PolygonsToRocks"),
        
        # Paso 5: Estadísticas (Histograma, Curva Real)
        GranulometryStatsNode(nbins=30, name="Statistics_Raw"),
        
        # Paso 6: Modelado Matemático (Rosin-Rammler)
        GranulometryModelNode(model_strategy=RosinRammler(), name="Statistics_Modeled"),
        
        # Paso 7: Visualización (Generar Figura)
        ResultVisualizerNode(name="Visualization")
    ]

    # 4. Listar imágenes
    supported_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tif']
    image_files = []
    for ext in supported_extensions:
        image_files.extend(INPUT_DIR.glob(ext))
        image_files.extend(INPUT_DIR.glob(ext.upper()))
    
    image_files = sorted(list(set(image_files))) # Eliminar duplicados y ordenar

    if not image_files:
        logging.warning(f"No se encontraron imágenes en {INPUT_DIR}")
        return

    logging.info(f"Se encontraron {len(image_files)} imágenes para procesar.")

    # 5. Loop de procesamiento
    for img_path in image_files:
        logging.info(f"--- Procesando: {img_path.name} ---")
        
        image = cv2.imread(str(img_path))
        if image is None:
            logging.error(f"Error al leer la imagen: {img_path}")
            continue

        context = {
            "image": image,
            "image_url": str(img_path),
            "job_id": img_path.stem,
            "lidar_distance": 20.0, # Valor por defecto para pruebas locales
            "original_shape": (image.shape[1], image.shape[0])
        }

        try:
            for node in nodes:
                context = node.run(context)

            fig = context.get('output_figure')
            if fig:
                output_path = OUTPUT_DIR / f"{img_path.stem}_result.png"
                fig.savefig(output_path, dpi=300, bbox_inches="tight")
                plt.close(fig)
                logging.info(f"✅ Guardado: {output_path}")
            else:
                logging.warning(f"⚠️ No se generó figura para {img_path.name}")

        except Exception as e:
            logging.error(f"❌ Fallo al procesar {img_path.name}: {e}", exc_info=True)

    logging.info(">>> Procesamiento finalizado.")

if __name__ == "__main__":
    main()
