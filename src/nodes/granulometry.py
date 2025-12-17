"""
Nodos del pipeline para el análisis de granulometría.

Este módulo contiene los nodos que realizan la transformación de píxeles a
unidades físicas, el cálculo estadístico de la distribución de tamaños y el
ajuste de modelos matemáticos.
"""
import logging
from typing import Any, Dict, List

import numpy as np

from src.domain import (CameraParameters, GranulometryCurveData,
                        GranulometryResult, HistogramBin, Rock)

from .base import PipelineNode


class FixedDistanceTransformer(PipelineNode):
    """
    Convierte polígonos en el espacio de píxeles a objetos `Rock` con
    propiedades físicas (área, diámetro) en unidades del mundo real.
    """
    def __init__(
        self,
        camera_params: CameraParameters,
        name: str = "distance_transformer"
    ):
        """
        Inicializa el nodo de transformación.

        Args:
            camera_params (CameraParameters): Objeto con los parámetros
                intrínsecos del sensor y la lente.
            distance (float): Distancia fija en metros desde la cámara al
                plano del material.
            name (str): Nombre del nodo.
        """
        super().__init__(name)
        self.camera_params = camera_params

        # Pre-cálculo de constantes de conversión
        self._mm_to_inch = 0.0393701
        self._m_to_mm = 1000.0
        self._diam_const = 2.0 / np.sqrt(np.pi)  # Para cálculo de diámetro equivalente

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calcula un factor de escala basado en la distancia y los parámetros de
        la cámara, y lo aplica a los polígonos de entrada para crear rocas.

        Context Inputs:
            - `polygons` (List[Polygon]): Lista de polígonos en píxeles.
            - `original_shape` (tuple): Tupla (ancho, alto) de la imagen original.

        Context Outputs:
            - `rocks` (List[Rock]): Lista de objetos `Rock` con propiedades
              físicas, ordenadas por diámetro.
        """
        polygons = context.get('polygons')
        original_shape = context.get('original_shape')

        if not polygons:
            logging.warning("[%s] No se encontraron polígonos en el contexto. El resultado será una lista de rocas vacía.", self.name)
            context['rocks'] = []
            return context

        if original_shape is None:
            raise ValueError(f"[{self.name}] 'original_shape' no encontrada en el contexto.")

        img_w, img_h = original_shape
        distance_mm = context.get("lidar_distance", 20.0) * self._m_to_mm

        # Calcular mm por píxel usando la fórmula de la cámara pinhole.
        x_mm_px = (distance_mm * self.camera_params.sensor_width) / (img_w * self.camera_params.focal_length)
        y_mm_px = (distance_mm * self.camera_params.sensor_height) / (img_h * self.camera_params.focal_length)

        # Convertir a pulgadas por píxel (estándar en minería)
        x_factor_inch = x_mm_px * self._mm_to_inch
        y_factor_inch = y_mm_px * self._mm_to_inch
        area_scale_factor = abs(x_factor_inch * y_factor_inch)

        logging.info("[%s] Factor de escala de área calculado: %f in²/px²", self.name, area_scale_factor)

        rocks_list = []
        for poly in polygons:
            phys_area = poly.area * area_scale_factor
            phys_diameter = self._diam_const * np.sqrt(phys_area)
            rocks_list.append(Rock(area=phys_area, diameter=phys_diameter, contorno=poly))

        rocks_list.sort(key=lambda r: r.diameter)

        context['rocks'] = rocks_list
        logging.info("[%s] Transformación completada. %d rocas generadas.", self.name, len(rocks_list))
        return context


class GranulometryStatsNode(PipelineNode):
    """
    Calcula estadísticas de distribución de tamaños (histograma, curva acumulada,
    P-values) a partir de una lista de rocas.
    """
    def __init__(self, nbins: int = 30, name: str = "stats_calculator"):
        """
        Args:
            nbins (int): Número de bins a usar para el histograma de tamaños.
            name (str): Nombre del nodo.
        """
        super().__init__(name)
        self.nbins = nbins

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Procesa la lista de rocas para generar el objeto de resultado principal.

        Context Inputs:
            - `rocks` (List[Rock]): Lista de rocas con sus propiedades físicas.

        Context Outputs:
            - `granulometry_result` (GranulometryResult): Objeto que encapsula
              todos los resultados estadísticos.
            - `bin_centers_raw` (np.array): Datos crudos de los centros de bin,
              para uso del nodo de modelado.
            - `curve_values_raw` (np.array): Datos crudos de la curva acumulada,
              para uso del nodo de modelado.
        """
        rocks: List[Rock] = context.get('rocks', [])
        if not rocks:
            raise ValueError(f"[{self.name}] No hay rocas en el contexto para procesar.")

        logging.info("[%s] Calculando estadísticas para %d rocas...", self.name, len(rocks))
        diameters = np.array([r.diameter for r in rocks])
        areas = np.array([r.area for r in rocks])

        max_diameter = np.max(diameters)
        if max_diameter == 0: max_diameter = 1.0

        bin_edges = np.linspace(0, max_diameter, self.nbins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        # Usar np.digitize y np.bincount para un cálculo eficiente.
        bin_indices = np.digitize(diameters, bin_edges) - 1
        bin_indices[bin_indices == self.nbins] = self.nbins - 1
        counts_per_bin = np.bincount(bin_indices, minlength=self.nbins)
        areas_per_bin = np.bincount(bin_indices, weights=areas, minlength=self.nbins)

        total_rock_area = np.sum(areas)
        retained_pct = areas_per_bin / total_rock_area if total_rock_area > 0 else np.zeros(self.nbins)
        
        # El % pasante es 1 - % retenido acumulado. Pero es más fácil
        # calcular el retenido acumulado y luego la curva pasante.
        passing_pct_curve = 1.0 - np.cumsum(retained_pct)
        # Aseguramos que la curva esté en el rango correcto y tenga la forma adecuada.
        passing_pct_curve = np.insert(passing_pct_curve[:-1], 0, 1.0)

        # Cálculo de P-Values (e.g., P80 es el tamaño por donde pasa el 80%)
        p_vals_dict = {
            f"P{p}": float(np.interp(p / 100.0, passing_pct_curve, bin_centers))
            for p in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        }
        
        histogram_list = [
            HistogramBin(
                range_start=float(bin_edges[i]), range_end=float(bin_edges[i+1]),
                count=int(counts_per_bin[i]), area_sum=float(areas_per_bin[i]),
                retained_pct=float(retained_pct[i])
            ) for i in range(self.nbins)
        ]

        result = GranulometryResult(
            total_rocks=len(rocks),
            histogram=histogram_list,
            real_curve=GranulometryCurveData(
                x_axis=bin_centers.tolist(), y_axis=passing_pct_curve.tolist(),
                p_values=p_vals_dict
            )
        )

        context['granulometry_result'] = result
        context['bin_centers_raw'] = bin_centers
        context['curve_values_raw'] = passing_pct_curve # Pasamos la curva pasante
        logging.info("[%s] Estadísticas calculadas. P50 real: %.4f", self.name, p_vals_dict.get('P50', 0))
        return context


class GranulometryModelNode(PipelineNode):
    """
    Ajusta un modelo matemático a la curva de granulometría obtenida.
    Utiliza un patrón de diseño 'Strategy' para el modelo de ajuste.
    """
    def __init__(self, model_strategy: Any, eps: float = 1e-6, name: str = "granulometry_model"):
        """
        Args:
            model_strategy (Any): Una instancia de una clase de estrategia de
                modelo (e.g., RosinRammler). Debe tener un método `.fit(x, y)`
                y un atributo `.name`.
            eps (float): Un valor épsilon para evitar divisiones por cero.
            name (str): Nombre del nodo.
        """
        super().__init__(name)
        self.model_strategy = model_strategy
        self.eps = eps

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Toma los datos de la curva real, ejecuta el ajuste del modelo y
        actualiza el objeto de resultado con los datos del modelo.

        Context Inputs:
            - `granulometry_result` (GranulometryResult): El objeto de resultados
              creado en el nodo anterior.
            - `bin_centers_raw` (np.array): Eje X para el ajuste.
            - `curve_values_raw` (np.array): Eje Y para el ajuste.

        Context Outputs:
            - Modifica `granulometry_result` in-place, añadiendo los
              resultados del modelo (`modeled_curve`, `model_name`, etc.).
        """
        result = context.get('granulometry_result')
        x_data = context.get('bin_centers_raw')
        y_data = context.get('curve_values_raw')

        if result is None or x_data is None or y_data is None:
            logging.warning("[%s] Faltan datos para el modelado. Saltando este paso.", self.name)
            return context

        logging.info("[%s] Ajustando modelo: %s...", self.name, self.model_strategy.name)

        fitted = self.model_strategy.fit(bin_centers=x_data, curve=y_data, eps=self.eps)
        modeled_y = np.nan_to_num(fitted.curve, nan=0.0)

        modeled_curve_data = GranulometryCurveData(
            x_axis=x_data.tolist(),
            y_axis=modeled_y.tolist(),
            p_values=fitted.p_values
        )

        # Actualiza el objeto de resultado que ya está en el contexto.
        result.modeled_curve = modeled_curve_data
        result.model_name = self.model_strategy.name
        
        # Mapea los atributos del resultado del modelo al objeto GranulometryResult.
        param_mapping = {
            'granulometry_xc': 'xc', 'granulometry_n': 'n',
            'granulometry_x50': 'x_50',
        }
        for source_attr, dest_attr in param_mapping.items():
            if hasattr(fitted, source_attr):
                setattr(result, dest_attr, float(getattr(fitted, source_attr)))

        logging.info("[%s] Modelo ajustado. P50 modelado: %.4f", self.name, result.x_50)
        return context