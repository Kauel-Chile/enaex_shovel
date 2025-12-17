"""
Define los esquemas de datos para los resultados de los modelos matemáticos
de ajuste de curvas de granulometría.

Estas clases actúan como una interfaz común que las diferentes estrategias de
modelado deben devolver para ser compatibles con `GranulometryModelNode`.
"""
from dataclasses import dataclass
from typing import Dict
import numpy as np


@dataclass(frozen=True)
class RosinRammlerResult:
    """
    Contenedor de resultados específico para el modelo Rosin-Rammler.

    Attributes:
        curve (np.ndarray): La curva de % pasante ajustada por el modelo.
        p_values (Dict[str, float]): Diccionario de percentiles clave (P-values)
            calculados a partir de la curva modelada.
        granulometry_xc (float): El tamaño característico (d' o d63.2), que
            representa el tamaño de tamiz a través del cual pasa el 63.2% del material.
        granulometry_n (float): El coeficiente de uniformidad, que describe la
            dispersión de la distribución de tamaños. Valores más altos indican
            una distribución más uniforme.
        granulometry_x50 (float): El P50 o tamaño mediano de la distribución
            según el modelo.
    """
    curve: np.ndarray
    p_values: Dict[str, float]
    granulometry_xc: float
    granulometry_n: float
    granulometry_x50: float
    # El resto de parámetros son opcionales o para uso interno
    granulometry_b: float
    granulometry_xmax: float


@dataclass(frozen=True)
class SwebrecResult:
    """
    Contenedor de resultados específico para el modelo Swebrec.

    Attributes:
        curve (np.ndarray): La curva de % pasante ajustada por el modelo.
        p_values (Dict[str, float]): Diccionario de percentiles clave (P-values).
        granulometry_xmax (float): El tamaño máximo de partícula.
        granulometry_x50 (float): La mediana de la distribución (P50).
        granulometry_b (float): El exponente de la curva.
    """
    curve: np.ndarray
    p_values: Dict[str, float]
    granulometry_xmax: float
    granulometry_x50: float
    granulometry_b: float
    # Los siguientes se pueden añadir si el modelo los devuelve de forma nativa
    granulometry_xc: float
    granulometry_n: float