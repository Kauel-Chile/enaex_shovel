import gc
import json
import statistics
import ast
import logging
from azure.servicebus import ServiceBusClient, ServiceBusReceiver
from src.pipeline import GranulometryPipeline

class Worker:
    def __init__(self, 
                 sb_conn_str: str, 
                 queue_name: str, 
                 blob_service_client, 
                 model_onnx,
                 camera_params):
        
        self.logger = logging.getLogger(__name__)
        
        if not self.logger.hasHandlers() and not logging.getLogger().hasHandlers():
            logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

        self.pipeline = GranulometryPipeline(
            blob_service_client=blob_service_client,
            model_onnx=model_onnx, 
            camera_params=camera_params
        )
        
        self.queue_name = queue_name
        self.sb_conn_str = sb_conn_str
        self.is_running = False

    def start(self):
        """
        Inicia el proceso de escucha.
        """
        self.is_running = True
        self.logger.info(f"🚀 Worker iniciado. Conectando a cola: '{self.queue_name}'")

        servicebus_client = None
        receiver = None

        try:
            # Creamos el cliente
            servicebus_client = ServiceBusClient.from_connection_string(conn_str=self.sb_conn_str)
            
            # Obtenemos el receiver
            receiver = servicebus_client.get_queue_receiver(
                queue_name=self.queue_name,
                max_lock_renewal_duration=3600  # 1 hora para procesar antes de perder el lock
            )
            
            # Iniciamos la conexión explícitamente con el 'with'
            with receiver:
                self.logger.info("👂 Conexión establecida. Esperando mensajes...")
                
                while self.is_running:
                    msgs = receiver.receive_messages(max_message_count=1, max_wait_time=5)

                    for msg in msgs:
                        self.logger.info(f"📩 Mensaje recibido (Seq: {msg.sequence_number})")
                        self._process_message(receiver, msg)
                        gc.collect() 

        except Exception as e:
            self.logger.critical(f"🔥 Error crítico en el Worker: {e}", exc_info=True)
            raise e
            
        finally:
            self.logger.info("🛑 Cerrando recursos del Worker...")
            if servicebus_client:
                servicebus_client.close()
            self.logger.info("Worker finalizado.")

    def _process_message(self, receiver: ServiceBusReceiver, msg):
        """
        Lógica de procesamiento individual.
        """
        context = {}
        try:
            body_str = str(msg)
            job_data = json.loads(body_str)
            
            # Validación de datos mínimos
            image_url = job_data.get("image_url") or job_data.get("url")
            job_id = str(job_data.get("id", "unknown"))

            if not image_url:
                raise ValueError(f"JSON sin 'image_url'. Keys: {list(job_data.keys())}")

            lidar_distance = self._get_distance_lidar(job_data)
            
            self.logger.info(f"data raw: {job_data}")
            self.logger.info(f"📐 Distancia LIDAR calculada: {lidar_distance:.2f} unidades.")

            context = {
                "job_id": job_id,
                "image_url": image_url,
                "lidar_distance": lidar_distance
            }

            self.logger.info(f"⚙️  Procesando Job ID: {job_id}...")
            result = self.pipeline.run(context)

            #self._log_execution_times(job_id, result.get('execution_times', {}))

            receiver.complete_message(msg)
            self.logger.info(f"✅ Job {job_id} completado exitosamente.")

        except json.JSONDecodeError:
            self.logger.error(f"❌ JSON inválido. Enviando a DeadLetter.")
            receiver.dead_letter_message(msg, reason="InvalidJSON", error_description="Parsing Failed")

        except Exception as e:
            job_id = context.get('job_id', 'Unknown')
            self.logger.error(f"❌ Error procesando Job {job_id}: {e}", exc_info=True)
            
            # Abandonamos el mensaje para que Azure lo reintente
            receiver.abandon_message(msg)

    def _log_execution_times(self, job_id: str, times: dict):
        if not times: return
        total = times.get('total_pipeline', 0)
        
        log_msg = [f"📊 TIEMPOS - JOB {job_id} (Total: {total:.2f}s)"]
        nodes = {k: v for k, v in times.items() if k != 'total_pipeline'}
        for node, dur in nodes.items():
            pct = (dur / total * 100) if total > 0 else 0
            log_msg.append(f"   • {node:<15}: {dur:.4f}s ({pct:.1f}%)")
        
        # Imprimimos todo en un solo bloque de log
        self.logger.info("\n".join(log_msg))

    def _get_distance_lidar(self, job_data: dict) -> float:
        # 1. Usar minúsculas para coincidir con el JSON 'l1' y 'l2'
        l1_raw = job_data.get("l1")
        l2_raw = job_data.get("l2")
        
        def parse_to_dict(val):
            if isinstance(val, dict):
                return val
            if isinstance(val, str):
                try:
                    # Convierte el string "{'a': 1}" en un dict real
                    return ast.literal_eval(val)
                except (ValueError, SyntaxError):
                    return {}
            return {}

        l1 = parse_to_dict(l1_raw)
        l2 = parse_to_dict(l2_raw)

        all_values = list(l1.values()) + list(l2.values())
        if not all_values:
            return 0.0

        return statistics.median(all_values)

    def stop(self):
        self.logger.info("🛑 Solicitud de parada recibida.")
        self.is_running = False