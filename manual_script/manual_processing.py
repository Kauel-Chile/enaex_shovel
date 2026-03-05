import logging
import sys
import time
import os
import gc
import psutil
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
BASE_DIR = Path(__file__).parent.parent
INPUT_DIR = BASE_DIR / "manual_script" / "images"
OUTPUT_DIR = BASE_DIR / "manual_script" / "results"

def get_resource_usage(process):
    """
    Captura el uso de RAM más preciso disponible (USS).
    USS (Unique Set Size) es la memoria que se liberaría si el proceso terminara ahora.
    """
    # Forzamos al recolector de basura para limpiar objetos huérfanos antes de medir
    gc.collect()
    
    try:
        # 'uss' es mucho más preciso que 'rss' para medir fugas o consumo real
        mem_info = process.memory_full_info()
        ram_mb = mem_info.uss / (1024 * 1024)
    except (psutil.AccessDenied, AttributeError):
        # Fallback a rss si uss no está disponible en el OS
        ram_mb = process.memory_info().rss / (1024 * 1024)
        
    cpu_usage = process.cpu_percent(interval=None)
    return {
        "ram_mb": ram_mb,
        "cpu_pct": cpu_usage
    }

def save_granulometry_data(path, granulometry_results, filename):
    """Extrae datos y guarda en TXT."""
    try:
        with open(path, 'w', encoding='utf-8') as f:
            f.write(f"REPORTE GRANULOMÉTRICO: {filename}\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Modelo aplicado: {granulometry_results.model_name}\n")
            f.write(f"Total de rocas detectadas: {granulometry_results.total_rocks}\n")
            f.write(f"X50 (Mediana): {granulometry_results.x_50:.4f}\n\n")

            mc = granulometry_results.modeled_curve
            if mc:
                f.write("--- CURVA MODELADA ---\n")
                x_vals = getattr(mc, 'x_axis', getattr(mc, 'x_values', []))
                y_vals = getattr(mc, 'y_axis', getattr(mc, 'y_values', []))
                
                topsize = max(x_vals) if x_vals else 0
                f.write(f"Topsize: {topsize:.2f}\n")
                f.write(f"{'Tamaño':<15} | {'Pasante (%)':<15}\n")
                f.write("-" * 35 + "\n")
                for x, y in zip(x_vals, y_vals):
                    val_y = y * 100 if max(y_vals) <= 1.0 else y
                    f.write(f"{x:<15.4f} | {val_y:<15.2f}%\n")
    except Exception as e:
        logging.error(f"Error al escribir el archivo TXT: {e}")

def main():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    current_process = psutil.Process(os.getpid())
    logging.info(">>> Iniciando script con monitoreo de precisión (USS RAM)...")

    if not INPUT_DIR.exists():
        logging.error(f"No se encontró la carpeta de entrada: {INPUT_DIR.absolute()}")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Cargar recursos
    t_init = time.perf_counter()
    try:
        model = load_model(PATH_MODEL)
        camera_params = load_camera_params()
        res_load = get_resource_usage(current_process)
        logging.info(f"📦 Recursos cargados en {time.perf_counter() - t_init:.2f}s")
        logging.info(f"💻 Consumo base (Modelo + Librerías): {res_load['ram_mb']:.2f} MB")
    except Exception as e:
        logging.critical(f"Error cargando recursos: {e}")
        return

    # 2. Nodos
    nodes = [
        OnnxInferenceNode(onnx_session=model, name="IA_Inference"),
        OnnxPostProcessingNode(name="PostProcessing"),
        FixedDistanceTransformer(camera_params=camera_params, name="PolygonsToRocks"),
        GranulometryStatsNode(nbins=30, name="Statistics_Raw"),
        GranulometryModelNode(model_strategy=RosinRammler(), name="Statistics_Modeled"),
        ResultVisualizerNode(name="Visualization")
    ]

    image_files = sorted([f for f in INPUT_DIR.glob('*') if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']])

    for img_path in image_files:
        logging.info(f"\n--- 🚀 Procesando: {img_path.name} ---")
        
        # Limpieza preventiva antes de empezar una nueva imagen
        gc.collect()
        t_start_img = time.perf_counter()
        initial_res = get_resource_usage(current_process)
        
        image = cv2.imread(str(img_path))
        if image is None: 
            logging.warning(f"No se pudo leer la imagen: {img_path.name}")
            continue

        context = {
            "image": image,
            "image_url": str(img_path),
            "job_id": img_path.stem,
            "lidar_distance": 20.0, 
            "original_shape": (image.shape[1], image.shape[0]),
            "execution_times": {} 
        }

        try:
            for node in nodes:
                t_node_start = time.perf_counter()
                context = node.run(context)
                duration = time.perf_counter() - t_node_start
                context["execution_times"][node.name] = duration
                
                # Medición por nodo
                m_node = get_resource_usage(current_process)['ram_mb']
                logging.info(f"  [Node] {node.name:<20} | {duration:.4f}s | RAM: {m_node:.2f}MB")

            # --- EXPORTACIÓN ---
            t_save_start = time.perf_counter()

            # Guardar Gráfico con limpieza profunda de memoria
            fig = context.get('output_figure')
            if fig:
                fig.savefig(OUTPUT_DIR / f"{img_path.stem}_plot.png", dpi=150) # DPI 150 es suficiente para reportes
                fig.clf()            # Limpia la figura actual
                plt.close(fig)       # Cierra la figura
                plt.close('all')     # Asegura que no queden procesos de Qt/Matplotlib colgados
                del fig              # Elimina referencia

            # Guardar Datos TXT
            stats_data = context.get('granulometry_result') 
            if stats_data:
                txt_save_path = OUTPUT_DIR / f"{img_path.stem}_data.txt"
                save_granulometry_data(txt_save_path, stats_data, img_path.name)
            
            t_save_end = time.perf_counter()
            logging.info(f"  [Save] Export Results       | {t_save_end - t_save_start:.4f}s")
            
            # REPORTE FINAL DE LA IMAGEN
            total_img_time = time.perf_counter() - t_start_img
            
            # Limpiamos el contexto antes de la medición final de la imagen
            del context
            del image
            gc.collect()
            
            final_res = get_resource_usage(current_process)
            
            logging.info(f"✅ FINALIZADO: {img_path.name}")
            logging.info(f"⏱️ Tiempo Total: {total_img_time:.3f}s")
            logging.info(f"🧠 RAM USS: {initial_res['ram_mb']:.2f}MB -> {final_res['ram_mb']:.2f}MB")
            logging.info(f"📈 Diferencia neta: {final_res['ram_mb'] - initial_res['ram_mb']:.2f} MB")
            logging.info("-" * 45)

        except Exception as e:
            logging.error(f"❌ Fallo en {img_path.name}: {e}", exc_info=True)
            # Intentar limpiar incluso si falla
            plt.close('all')
            gc.collect()

    logging.info(f"\n>>> Script completado. Resultados en: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()