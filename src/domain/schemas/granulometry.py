"""
Define los esquemas de datos para encapsular los resultados de un análisis
de granulometría.
"""
from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass(frozen=True)
class GranulometryCurveData:
    """
    Almacena los puntos de datos para una curva de granulometría.

    Esta estructura contiene los ejes `x` (tamaño de partícula) e `y` (% pasante)
    necesarios para graficar, así como los valores de percentiles clave.

    Attributes:
        x_axis (List[float]): Los centros de los bins de tamaño (diámetro).
        y_axis (List[float]): El porcentaje pasante acumulado, en un rango de 0.0 a 1.0.
        p_values (Dict[str, float]): Un diccionario con los valores de tamaño
            para percentiles clave (e.g., 'p80', 'p50', 'p20').
    """
    x_axis: List[float]
    y_axis: List[float]
    p_values: Dict[str, float]


@dataclass(frozen=True)
class HistogramBin:
    """
    Representa una única 'gaveta' o 'bin' en un histograma de distribución de tamaños.

    Cada bin corresponde a un rango de diámetros de partículas.

    Attributes:
        range_start (float): Límite inferior del rango de diámetros del bin.
        range_end (float): Límite superior del rango de diámetros del bin.
        count (int): Número de rocas cuyo diámetro cae dentro de este bin.
        area_sum (float): Suma total del área de las rocas en este bin.
        retained_pct (float): El porcentaje de material retenido en este bin
            respecto al total.
    """
    range_start: float
    range_end: float
    count: int
    area_sum: float
    retained_pct: float


@dataclass
class GranulometryResult:
    """
    Contenedor final para todos los resultados de un análisis de granulometría.

    Esta clase es un objeto de transferencia de datos (DTO) diseñado para ser
    agnóstico a cualquier framework de cálculo o visualización. Agrega toda la
    información relevante generada por el pipeline.

    Attributes:
        total_rocks (int): El número total de rocas detectadas en la imagen.
        histogram (List[HistogramBin]): Lista de bins que componen el histograma
            de distribución de tamaños.
        real_curve (GranulometryCurveData): Los datos de la curva de granulometría
            calculada directamente a partir de los datos (la curva real).
        modeled_curve (Optional[GranulometryCurveData]): Los datos de la curva
            ajustada por un modelo matemático (e.g., Rosin-Rammler). Es opcional
            en caso de que el modelado no se realice o falle.
        model_name (str): Nombre del modelo matemático utilizado (e.g., "RosinRammler").
        xc (float): Parámetro de tamaño característico del modelo (e.g., d63.2 para RR).
        n (float): Parámetro de uniformidad o dispersión del modelo (e.g., exponente de RR).
        x_max (float): Tamaño máximo de partícula estimado por el modelo.
        x_50 (float): Mediana de la distribución (P50) según el modelo.
    """
    total_rocks: int
    histogram: List[HistogramBin]
    real_curve: GranulometryCurveData
    modeled_curve: Optional[GranulometryCurveData] = None
    model_name: str = "None"
    xc: float = 0.0
    n: float = 0.0
    x_50: float = 0.0
    
    # x_max se puede añadir si es relevante para el modelo
    # x_max: float = 0.0