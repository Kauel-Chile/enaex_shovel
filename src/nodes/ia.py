"""
Nodos del pipeline responsables de la ejecución de la inferencia del modelo de
Machine Learning y el post-procesamiento de sus resultados.
"""
import logging

import cv2
import numpy as np
import onnxruntime as ort
from shapely.geometry import Polygon
from skimage.segmentation import watershed
from typing import Any, Dict

from .base import PipelineNode


class OnnxInferenceNode(PipelineNode):
    """
    Ejecuta la inferencia de un modelo ONNX sobre una imagen de entrada.
    """
    def __init__(
        self,
        onnx_session: ort.InferenceSession,
        name: str = "inference_node"
    ):
        """
        Inicializa el nodo de inferencia.

        Args:
            onnx_session (ort.InferenceSession): Sesión de ONNX Runtime ya
                cargada e inicializada.
            name (str): Nombre del nodo.
        """
        super().__init__(name)
        self.session = onnx_session
        self.input_name = self.session.get_inputs()[0].name

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Realiza el pre-procesamiento, ejecuta el modelo y guarda las salidas en el contexto.

        Context Inputs:
            - `image` (np.ndarray): La imagen de entrada a procesar.

        Context Outputs:
            - `raw_outputs` (tuple): Una tupla con los arrays de salida crudos del modelo.
            - `original_shape` (tuple): La forma (ancho, alto) de la imagen original.
        """
        image = context.get('image')
        if image is None:
            raise ValueError(f"[{self.name}] 'image' no encontrada en el contexto.")

        original_h, original_w = image.shape[:2]
        context['original_shape'] = (original_w, original_h)

        logging.info("[%s] Pre-procesando imagen...", self.name)
        img_input = self.__preprocessing(image)

        logging.info("[%s] Ejecutando inferencia ONNX...", self.name)
        outputs_raw = self.__inference(img_input)
        
        context['raw_outputs'] = outputs_raw
        logging.info("[%s] Inferencia completada.", self.name)
        return context

    def __preprocessing(self, image: np.ndarray) -> np.ndarray:
        """Prepara la imagen para el formato que espera el modelo ONNX."""
        # Redimensiona a 1024x1024, normaliza y añade una dimensión de batch.
        img = cv2.resize(image, (1024, 1024)).astype(np.float32)
        img /= 255.0
        return img[None, ...]

    def __inference(self, image: np.ndarray) -> tuple:
        """Ejecuta la sesión de inferencia de ONNX."""
        return tuple(self.session.run(None, {self.input_name: image}))


class OnnxPostProcessingNode(PipelineNode):
    """
    Post-procesa las salidas crudas de un modelo de segmentación para extraer
    polígonos de objetos.
    """
    def __init__(self, name: str = "postproc_node"):
        super().__init__(name)

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Toma las salidas del modelo y las convierte en polígonos geométricos.

        Context Inputs:
            - `raw_outputs` (tuple): Salidas crudas del `OnnxInferenceNode`.
            - `original_shape` (tuple): Forma (ancho, alto) de la imagen original.

        Context Outputs:
            - `polygons` (List[Polygon]): Lista de polígonos de Shapely, cada
              uno representando una partícula detectada.
            - `mask_fine` (np.ndarray): Máscara binaria del material fino.
            - `mask_bg` (np.ndarray): Máscara binaria del fondo.
        """
        outputs = context.get('raw_outputs')
        original_shape = context.get('original_shape')

        if outputs is None or original_shape is None:
            raise ValueError(f"[{self.name}] Faltan 'raw_outputs' u 'original_shape' en el contexto.")

        logging.info("[%s] Iniciando post-procesamiento de salidas del modelo...", self.name)
        polygons, mask_fine, mask_bg = self.__process_logic(outputs, target_size=original_shape)

        context['polygons'] = polygons
        context['mask_fine'] = mask_fine
        context['mask_bg'] = mask_bg
        logging.info("[%s] Post-procesamiento finalizado. Se extrajeron %d polígonos.", self.name, len(polygons))
        
        return context

    def __process_logic(self, outputs: tuple, target_size: tuple) -> tuple:
        """Contiene la lógica de visión computacional para segmentar las partículas."""
        # 1. Desempaquetar y preparar canales de probabilidad del modelo.
        o_class, o_ncs = outputs[0][0], outputs[1][0]
        prob_fine, prob_stones = o_class[..., 0], o_class[..., 1]
        prob_centers, prob_edges = o_ncs[..., 1], self.__remap(o_ncs[..., 2])
        prob_stones = self.__smooth_channel(prob_stones)
        prob_centers = self.__smooth_channel(prob_centers)

        # 2. Crear máscaras binarias a partir de umbrales.
        mask_stones_area = prob_stones >= 0.5
        centers_bin = np.logical_and(prob_centers >= 0.1, prob_edges < 0.25)
        edges_bin = prob_edges >= 0.25

        # 3. Aplicar Watershed para separar instancias de rocas que se tocan.
        labels = self.__apply_watershed(centers_bin, edges_bin, mask=mask_stones_area)

        # 4. Extraer contornos de las etiquetas y convertirlos a polígonos.
        polygons = self.__extract_polygons(labels, target_size)

        # 5. Crear máscaras finales y redimensionarlas al tamaño original.
        mask_fine_raw = prob_fine * np.logical_not(edges_bin)
        mask_fine = (mask_fine_raw > 0.78).astype(np.uint8) * 255
        mask_bg = (prob_fine > 0.78).astype(np.uint8) * 255

        if target_size != (1024, 1024):
            mask_fine = cv2.resize(mask_fine, target_size, interpolation=cv2.INTER_AREA)
            mask_bg = cv2.resize(mask_bg, target_size, interpolation=cv2.INTER_AREA)

        return polygons, mask_fine, mask_bg

    def __remap(self, x: np.ndarray) -> np.ndarray:
        """Normaliza un array a un rango de [0, 1]."""
        min_val, max_val = x.min(), x.max()
        diff = max_val - min_val
        return np.zeros_like(x) if diff < 1e-6 else (x - min_val) / diff

    def __smooth_channel(self, channel: np.ndarray) -> np.ndarray:
        """Aplica un remapeo y un suavizado de caja."""
        x = self.__remap(channel)
        return cv2.blur(x, (3, 3))

    def __apply_watershed(self, centers_bin, edges_bin, mask) -> np.ndarray:
        """Ejecuta el algoritmo Watershed para la segmentación de instancias."""
        centers_u8 = (centers_bin.astype(np.uint8) * 255)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        centers_u8 = cv2.morphologyEx(centers_u8, cv2.MORPH_CLOSE, kernel)
        _, markers = cv2.connectedComponents(centers_u8)
        
        edges_inv_u8 = ((~edges_bin).astype(np.uint8) * 255)
        dist_transform = cv2.distanceTransform(edges_inv_u8, cv2.DIST_LABEL_PIXEL, 0)

        return watershed(-dist_transform, markers, mask=mask, watershed_line=True)

    def __extract_polygons(self, labels: np.ndarray, target_size: tuple) -> list:
        """Extrae contornos de una máscara de etiquetas y los convierte a Polígonos de Shapely."""
        polygons = []
        w_target, h_target = target_size
        scale_factor = np.array([w_target / 1024.0, h_target / 1024.0], dtype=np.float32)

        unique_labels = np.unique(labels)
        for i in unique_labels:
            if i <= 0: continue # Omitir fondo

            lbl_mask = (labels == i).astype(np.uint8)
            contours, _ = cv2.findContours(lbl_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for cnt in contours:
                if len(cnt) < 3: continue
                
                # Simplificar contorno y escalar a tamaño original
                epsilon = 0.002 * cv2.arcLength(cnt, True)
                cnt_approx = cv2.approxPolyDP(cnt, epsilon, True)
                if len(cnt_approx) < 3: continue
                
                points_scaled = cnt_approx.squeeze() * scale_factor
                if len(points_scaled) < 3: continue
                
                # Crear y validar polígono de Shapely
                poly = Polygon(points_scaled)
                if not poly.is_valid: poly = poly.buffer(0)
                
                if not poly.is_empty:
                    if poly.geom_type == 'MultiPolygon':
                        for geom in poly.geoms:
                            if not geom.is_empty: polygons.append(geom)
                    else:
                        polygons.append(poly)
        return polygons