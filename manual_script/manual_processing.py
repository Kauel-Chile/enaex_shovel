"""
Script de procesamiento manual local con exportación a TXT.
"""
import logging
import sys
from pathlib import Path
import cv2
import matplotlib.pyplot as plt

# Aseguramos que el directorio raíz esté en el path
sys.path.append(str(Path(__file__).parent.parent))

from config.settings import PATH_MODEL
from src.utils import load_model, load_camera_params
from src.domain.granulometry.models import RosinRammler
from src.nodes.ia import OnnxInferenceNode, OnnxPostProcessingNode
from src.nodes.granulometry import (FixedDistanceTransformer,
                                    GranulometryStatsNode, GranulometryModelNode)
from src.nodes.viz import ResultVisualizerNode

# Configuración de carpetas
INPUT_DIR = Path(__file__).parent.parent / "manual_script" / "images"
OUTPUT_DIR = Path(__file__).parent.parent / "manual_script" / "results"

def save_granulometry_data(path, granulometry_results, filename):
    """
    Extrae datos del esquema GranulometryResults y los guarda en TXT.
    """
    try:
        with open(path, 'w', encoding='utf-8') as f:
            f.write(f"REPORTE GRANULOMÉTRICO: {filename}\n")
            f.write("=" * 50 + "\n\n")

            # 1. Información General y Parámetros del Modelo
            f.write(f"Modelo aplicado: {granulometry_results.model_name}\n")
            f.write(f"Total de rocas detectadas: {granulometry_results.total_rocks}\n")
            f.write(f"Parámetro xc (Característico): {granulometry_results.xc:.4f}\n")
            f.write(f"Parámetro n (Uniformidad): {granulometry_results.n:.4f}\n")
            f.write(f"X50 (Mediana): {granulometry_results.x_50:.4f}\n\n")

            # 2. Curva Modelada (P-valores y puntos)
            f.write("--- CURVA MODELADA ---\n")
            mc = granulometry_results.modeled_curve
            
            if mc:
                # Extraer P-Valores (asumiendo que mc tiene un dict p_values o similar)
                # Si GranulometryCurveData tiene un atributo p_values:
                if hasattr(mc, 'p_values') and mc.p_values:
                    f.write("P-Valores:\n")
                    for p_name, p_val in mc.p_values.items():
                        f.write(f"  {p_name}: {p_val:.2f} \n")
                
                # Topsize (usualmente el valor máximo de x o un P95/P100)
                topsize = max(mc.x_values) if mc.x_values else 0
                f.write(f"Topsize (Máximo detectado): {topsize:.2f}  \n\n")

                # 3. Puntos de la curva (X = Tamaño, Y = Pasante Acumulado)
                f.write("Tabla de la Curva Modelada (Tamaño vs % Pasante):\n")
                f.write(f"{'Tamaño':<15} | {'Pasante (%)':<15}\n")
                f.write("-" * 35 + "\n")
                for x, y in zip(mc.x_values, mc.y_values):
                    # Multiplicamos y por 100 si viene en formato 0-1
                    val_y = y * 100 if max(mc.y_values) <= 1.0 else y
                    f.write(f"{x:<15.4f} | {val_y:<15.2f}%\n")
            else:
                f.write("AVISO: No se generó curva modelada en este proceso.\n")

    except Exception as e:
        logging.error(f"Error al escribir el archivo TXT: {e}")

def main():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    logging.info(">>> Iniciando script con exportación de datos...")

    if not INPUT_DIR.exists():
        logging.error(f"No se encontró: {INPUT_DIR.absolute()}")
        return

    OUTPUT_DIR.mkdir(exist_ok=True)

    # 2. Cargar recursos
    try:
        model = load_model(PATH_MODEL)
        camera_params = load_camera_params()
    except Exception as e:
        logging.critical(f"Error cargando recursos: {e}")
        return

    # 3. Nodos
    nodes = [
        OnnxInferenceNode(onnx_session=model, name="IA_Inference"),
        OnnxPostProcessingNode(name="PostProcessing"),
        FixedDistanceTransformer(camera_params=camera_params, name="PolygonsToRocks"),
        GranulometryStatsNode(nbins=30, name="Statistics_Raw"),
        GranulometryModelNode(model_strategy=RosinRammler(), name="Statistics_Modeled"),
        ResultVisualizerNode(name="Visualization")
    ]

    image_files = sorted([f for f in INPUT_DIR.glob('*') if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']])

    # 5. Loop de procesamiento
    for img_path in image_files:
        logging.info(f"--- Procesando: {img_path.name} ---")
        
        image = cv2.imread(str(img_path))
        if image is None: continue

        context = {
            "image": image,
            "image_url": str(img_path),
            "job_id": img_path.stem,
            "lidar_distance": 20.0, 
            "original_shape": (image.shape[1], image.shape[0])
        }

        try:
            for node in nodes:
                context = node.run(context)

            # --- GUARDAR IMAGEN ---
            fig = context.get('output_figure')
            if fig:
                fig.savefig(OUTPUT_DIR / f"{img_path.stem}_result.png", dpi=300)
                plt.close(fig)

            # --- GUARDAR DATOS TXT ---
            # Asumimos que GranulometryModelNode guarda los resultados en 'model_results'
            # o directamente en las llaves del contexto (ajusta según tu implementación)
            stats_data = context.get('granulometry_result', context) 
            print(stats_data)
            txt_path = OUTPUT_DIR / f"{img_path.stem}_data.txt"
            
            save_granulometry_data(txt_path, stats_data, img_path.name)
            
            logging.info(f"✅ Resultados exportados para {img_path.name}")

        except Exception as e:
            logging.error(f"❌ Fallo en {img_path.name}: {e}", exc_info=True)

    logging.info(">>> Procesamiento finalizado.")

if __name__ == "__main__":
    main()