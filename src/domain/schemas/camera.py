"""
Define el esquema de datos para los parámetros intrínsecos de una cámara.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class CameraParameters:
    """
    Almacena los parámetros intrínsecos de la cámara y su sensor.

    Esta estructura de datos es inmutable y contiene la información esencial
    utilizada en cálculos fotogramétricos para relacionar las medidas en
    píxeles de una imagen con las medidas en unidades del mundo real (e.g., mm).
    Es un componente clave para el nodo `FixedDistanceTransformer`.

    Attributes:
        focal_length (float): La distancia focal de la lente en milímetros (mm).
        sensor_width (float): El ancho del sensor de la cámara en milímetros (mm).
        sensor_height (float): La altura del sensor de la cámara en milímetros (mm).
    """
    focal_length: float
    sensor_width: float
    sensor_height: float