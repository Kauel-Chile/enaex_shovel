"""
Define el esquema de datos para una roca individual detectada.
"""
from dataclasses import dataclass
from shapely.geometry import Polygon


@dataclass(frozen=True)
class Rock:
    """
    Representa una única roca o partícula detectada y sus propiedades.

    Esta estructura de datos está altamente optimizada para la memoria mediante
    el uso de `__slots__`. Esto es crucial porque un solo análisis puede
    generar miles o millones de instancias de esta clase. Al no crear un

    diccionario `__dict__` para cada objeto, se reduce significativamente el
    consumo de memoria.

    La clase se define como inmutable (`frozen=True`) para garantizar que los
    datos no se modifiquen después de su creación.

    Attributes:
        area (float): El área de la sección transversal de la roca en unidades
            del mundo real al cuadrado (e.g., mm²).
        diameter (float): El diámetro equivalente de la roca, calculado a
            partir del área. Se expresa en unidades del mundo real (e.g., mm).
        contorno (Polygon): El polígono de Shapely que representa el contorno
            de la roca en el espacio de píxeles original de la imagen.
    """
    __slots__ = ['area', 'diameter', 'contorno']
    area: float
    diameter: float
    contorno: Polygon
