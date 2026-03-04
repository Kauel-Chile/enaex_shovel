import gc
import json
import statistics
import ast
import logging
import psutil
import os
import time
from azure.servicebus import ServiceBusClient, ServiceBusReceiver
from src.pipeline import GranulometryPipeline

class Worker:
    def __init__(self, sb_conn_str, queue_name, blob_service_client, model_onnx, camera_params):
        self.logger = logging.getLogger(__name__)
        
        # Inicializar Pipeline
        self.pipeline = GranulometryPipeline(
            blob_service_client=blob_service_client,
            model_onnx=model_onnx, 
            camera_params=camera_params
        )
        
        self.queue_name = queue_name
        self.sb_conn_str = sb_conn_str
        self.is_running = False
        
        # Referencia al proceso actual para métricas de hardware
        self.process = psutil.Process(os.getpid())

    def _get_resource_usage(self):
        """Retorna el uso actual de RAM (MB) y CPU (%)."""
        mem_info = self.process.memory_info()
        # interval=None para no bloquear el hilo; mide desde la última llamada
        cpu_usage = self.process.cpu_percent(interval=None) 
        return {
            "ram_mb": mem_info.rss / (1024 * 1024),
            "cpu_pct": cpu_usage
        }

    def start(self):
        self.is_running = True
        self.logger.info(f"Worker activo en cola: '{self.queue_name}'")

        try:
            servicebus_client = ServiceBusClient.from_connection_string(conn_str=self.sb_conn_str)
            receiver = servicebus_client.get_queue_receiver(
                queue_name=self.queue_name,
                max_lock_renewal_duration=3600 
            )
            
            with receiver:
                while self.is_running:
                    # Espera mensajes (polling de 5 seg)
                    msgs = receiver.receive_messages(max_message_count=1, max_wait_time=5)

                    for msg in msgs:
                        self._process_message(receiver, msg)
                        gc.collect() # Limpieza de memoria tras cada inferencia

        except Exception as e:
            self.logger.critical(f"Error en loop de Service Bus: {e}", exc_info=True)
            raise e

    def _process_message(self, receiver: ServiceBusReceiver, msg):
        context = {}
        # --- MEDICIÓN DE INICIO ---
        start_wall_time = time.perf_counter()
        initial_res = self._get_resource_usage()

        try:
            job_data = json.loads(str(msg))
            job_id = str(job_data.get("id", "unknown"))
            image_url = job_data.get("image_url") or job_data.get("url")

            if not image_url:
                raise ValueError("Mensaje recibido sin URL de imagen.")

            lidar_dist = self._get_distance_lidar(job_data)
            context = {"job_id": job_id, "image_url": image_url, "lidar_distance": lidar_dist}

            self.logger.info(f"Procesando Job: {job_id}")
            
            # EJECUCIÓN DEL PIPELINE
            result = self.pipeline.run(context)

            # --- MEDICIÓN DE FIN ---
            end_wall_time = time.perf_counter()
            final_res = self._get_resource_usage()
            total_duration = end_wall_time - start_wall_time

            # LOG DE MÉTRICAS CONSOLIDADAS
            self._log_performance_metrics(
                job_id, 
                total_duration, 
                result.get('execution_times', {}), 
                initial_res, 
                final_res
            )

            receiver.complete_message(msg)

        except json.JSONDecodeError:
            self.logger.error("JSON Corrupto. Enviando a DeadLetter.")
            receiver.dead_letter_message(msg, reason="InvalidJSON")
        except Exception as e:
            self.logger.error(f"Error en Job {context.get('job_id')}: {e}")
            receiver.abandon_message(msg)

    def _log_performance_metrics(self, job_id, total_dur, stages, init_res, fin_res):
        """Imprime un reporte detallado del uso de recursos y tiempos por etapa."""
        
        log_lines = [
            f"\n{'='*40}",
            f"PERFORMANCE REPORT - JOB: {job_id}",
            f"{'-'*40}",
            f"Tiempo Total Proceso: {total_dur:.3f}s",
            f"RAM: {init_res['ram_mb']:.1f}MB - {fin_res['ram_mb']:.1f}MB (Δ: {fin_res['ram_mb']-init_res['ram_mb']:.1f}MB)",
            f"CPU Avg: {fin_res['cpu_pct']}%",
            f"{'-'*40}",
            "Desglose por Etapas:"
        ]

        # Si el pipeline devuelve tiempos internos, los iteramos
        if stages:
            for stage, dur in stages.items():
                if stage == 'total_pipeline': continue
                pct = (dur / total_dur * 100) if total_dur > 0 else 0
                log_lines.append(f"   • {stage:<20}: {dur:.4f}s ({pct:.1f}%)")
        
        log_lines.append(f"{'='*40}\n")
        self.logger.info("\n".join(log_lines))

    def _get_distance_lidar(self, job_data: dict) -> float:
        l1 = parse_to_dict(job_data.get("l1"))
        l2 = parse_to_dict(job_data.get("l2"))
        all_vals = list(l1.values()) + list(l2.values())
        return statistics.median(all_vals) if all_vals else 0.0

    def stop(self):
        self.is_running = False

def parse_to_dict(val):
    if isinstance(val, dict): return val
    if isinstance(val, str):
        try: return ast.literal_eval(val)
        except: return {}
    return {}