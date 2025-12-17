"""
Define la interfaz base para todos los nodos del pipeline de procesamiento.
"""
from abc import ABC, abstractmethod
from typing import Any, Dict


class PipelineNode(ABC):
    """
    Clase base abstracta para un nodo de procesamiento en el pipeline.

    Cada 'nodo' representa un paso atómico y cohesivo en el proceso de análisis
    (ej: descargar imagen, ejecutar inferencia, calcular estadísticas).
    Esta clase define el contrato que todos los nodos deben seguir, garantizando
    que puedan ser encadenados y ejecutados de manera uniforme por el orquestador
    del pipeline.

    Attributes:
        name (str): El nombre del nodo, utilizado para logging y seguimiento.
    """
    def __init__(self, name: str):
        """
        Inicializa el nodo del pipeline.

        Args:
            name (str): Un nombre descriptivo para el nodo.
        """
        self.name = name

    @abstractmethod
    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Ejecuta la lógica de procesamiento del nodo.

        Este método es el corazón del nodo. Recibe un diccionario de 'contexto'
        que contiene los datos de entrada generados por nodos anteriores. El
        nodo debe leer la información que necesita de este contexto, realizar
        su procesamiento y escribir sus resultados de nuevo en el mismo
        diccionario. De esta forma, los datos fluyen a través del pipeline.

        Args:
            context (Dict[str, Any]): Un diccionario que actúa como estado
                compartido, transportando datos entre los nodos del pipeline.

        Returns:
            Dict[str, Any]: El diccionario de contexto, modificado con los
                resultados del procesamiento de este nodo.
        """
        pass