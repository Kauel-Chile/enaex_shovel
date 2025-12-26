import json
import logging
import dataclasses
import numpy as np 

from datetime import datetime
from typing import Any, Dict

from azure.servicebus import ServiceBusClient, ServiceBusMessage
from src.nodes.base import PipelineNode

class ServiceBusSenderNode(PipelineNode):
    """
    Nodo que toma el resultado de granulometría (Dataclass), lo convierte a Dict,
    resuelve tipos numéricos incompatibles (NumPy) y lo envía a Azure Service Bus.
    """

    def __init__(self, 
                 connection_string: str, 
                 queue_name: str, 
                 input_key: str = "granulometry_result",
                 name: str = "Sender_Queue"): 
        super().__init__(name)
        self.connection_string = connection_string
        self.queue_name = queue_name
        self.input_key = input_key
        self.logger = logging.getLogger(name)

    def _json_serializer_helper(self, obj):
        """
        Ayudante para que json.dumps pueda manejar tipos que no son estándar,
        como números de NumPy o Dataclasses anidados.
        """
        # 1. Tipos de NumPy (int)
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        
        # 2. Tipos de NumPy (float)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            # A veces numpy genera NaNs o Infinitos que rompen el JSON estándar
            if np.isnan(obj) or np.isinf(obj):
                return None  # O 0.0, según tu preferencia
            return float(obj)
        
        # 3. Arrays de NumPy
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        
        # 4. Dataclasses (por si alguno escapó al asdict inicial)
        elif dataclasses.is_dataclass(obj):
            return dataclasses.asdict(obj)

        # 5. Fallback a string para fechas u otros objetos
        return str(obj)

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        self.logger.info(f"🚀 Ejecutando nodo de envío: {self.name}")

        # --- 1. Obtener Datos ---
        raw_result = context.get(self.input_key)
        
        if raw_result is None:
            self.logger.warning(f"⚠️ No hay datos en '{self.input_key}'. Se omite envío.")
            return context

        # --- 2. Preparar Estructura (Dataclass -> Dict) ---
        # Usamos asdict para convertir recursivamente GranulometryResult y sus hijos
        # (HistogramBin, GranulometryCurveData) en diccionarios puros de Python.
        if dataclasses.is_dataclass(raw_result):
            result_dict = dataclasses.asdict(raw_result)
        else:
            # Si por alguna razón ya era un dict o es None
            result_dict = raw_result
        # Estructura final del mensaje
        payload = {
            "id": context.get("job_id"),
            "url_raw": context.get("image_url"),
            "url_result": context.get("uploaded_url"),   
            "timestamp": str(str(datetime.now())), # Ejemplo de metadata extra
            "granulometry": result_dict
        }

        try:
            # --- 3. Serializar a JSON ---
            # El parámetro 'default' se encarga de limpiar los tipos NumPy dentro del dict
            message_body = json.dumps(payload, default=self._json_serializer_helper)
            
            print(message_body)

            # --- 4. Enviar a Azure ---
            client = ServiceBusClient.from_connection_string(self.connection_string)
            with client:
                sender = client.get_queue_sender(self.queue_name)
                with sender:
                    message = ServiceBusMessage(message_body)
                    message.content_type = "application/json"
                    sender.send_messages(message)
                    
                    self.logger.info(f"✅ Resultados enviados a cola '{self.queue_name}'. Job ID: {payload['id']}")

        except TypeError as e:
            self.logger.error(f"❌ Error de tipos al serializar JSON: {e}")
            raise e
        except Exception as e:
            self.logger.error(f"❌ Error de comunicación con Service Bus: {e}", exc_info=True)
            # raise e # Descomentar si quieres detener el pipeline ante fallos de red

        return context