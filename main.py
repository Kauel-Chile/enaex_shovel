
import logging
import sys

from azure.storage.blob import BlobServiceClient

from config.settings import STORAGE_CONN_STR, QUEUE_CONN_STR, QUEUE_INPUT_NAME, PATH_MODEL
from src.utils import load_model, load_camera_params
from src.worker import Worker

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logging.getLogger("azure").setLevel(logging.WARNING)
logging.getLogger("uamqp").setLevel(logging.WARNING)


if __name__ == "__main__":
    logging.info(">>> Iniciando servicio de Granulometría...")

    # Verificación crítica de configuración antes de arrancar.
    if not STORAGE_CONN_STR:
        logging.error("La variable de entorno 'storage_account_conn_str' (STORAGE_CONN_STR) no está definida. El servicio no puede iniciar.")
        sys.exit(1)

    if not QUEUE_CONN_STR:
        logging.error("La variable de entorno 'service_bus_conn_str' (QUEUE_CONN_STR) no está definida. El servicio no puede iniciar.")
        sys.exit(1)

    worker = None
    try:
        model = load_model(PATH_MODEL)
        camera_params = load_camera_params()
        blob_service_client = BlobServiceClient.from_connection_string(STORAGE_CONN_STR)
        
        worker = Worker(
            sb_conn_str=QUEUE_CONN_STR,
            queue_name=QUEUE_INPUT_NAME,
            blob_service_client=blob_service_client,
            model_onnx=model, 
            camera_params=camera_params
        )

        worker.start()

    except KeyboardInterrupt:
        logging.warning("\nInterrupción manual detectada. Finalizando servicio...")
        if worker:
            worker.stop()
        logging.info("Servicio detenido limpiamente. ¡Adiós!")
        
    except Exception as e:
        logging.critical(f"Error fatal en el ciclo principal: {e}", exc_info=True)
        # Si ocurre un error inesperado, se intenta detener el worker
        # de forma ordenada antes de terminar el proceso.
        if worker:
            worker.stop()
        sys.exit(1)