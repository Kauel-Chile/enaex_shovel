import logging
import cv2
import numpy as np
import onnxruntime as ort
from shapely.geometry import Polygon
from skimage.segmentation import watershed
from typing import Any, Dict, List

from .base import PipelineNode

def get_valid_inference_size(shape: tuple, depth: int = 7) -> tuple:
    """
    Calcula las dimensiones (W, H) más cercanas que son divisibles por 2^depth.
    Esto es crítico para modelos con arquitectura U-Net o Encoder-Decoder.
    """
    divisor = 2 ** depth
    h, w = shape[:2]
    new_w = (w // divisor) * divisor
    new_h = (h // divisor) * divisor
    
    # Aseguramos un tamaño mínimo de un bloque
    return max(new_w, divisor), max(new_h, divisor)


class OnnxInferenceNode(PipelineNode):
    """
    Ejecuta la inferencia de un modelo ONNX adaptando la imagen a dimensiones
    compatibles con la profundidad del modelo (múltiplos de 128).
    """
    def __init__(self, onnx_session: ort.InferenceSession, name: str = "inference_node"):
        super().__init__(name)
        self.session = onnx_session
        self.input_name = self.session.get_inputs()[0].name

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        image = context.get('image')
        if image is None:
            raise ValueError(f"[{self.name}] 'image' no encontrada en el contexto.")

        # 1. Guardar dimensiones originales (H, W)
        orig_h, orig_w = image.shape[:2]
        context['original_shape'] = (orig_w, orig_h)

        # 2. Calcular tamaño óptimo (múltiplo de 128)
        target_w, target_h = get_valid_inference_size(image.shape, depth=7)
        context['inference_shape'] = (target_w, target_h)

        logging.info("[%s] Redimensionando de %dx%d a %dx%d para inferencia...", 
                     self.name, orig_w, orig_h, target_w, target_h)
        
        # 3. Pre-procesamiento
        img_input = self.__preprocessing(image, (target_w, target_h))

        # 4. Inferencia
        logging.info("[%s] Ejecutando inferencia ONNX...", self.name)
        outputs_raw = self.__inference(img_input)
        
        context['raw_outputs'] = outputs_raw
        return context

    def __preprocessing(self, image: np.ndarray, target_size: tuple) -> np.ndarray:
        """Redimensiona, normaliza y ajusta dimensiones para el modelo."""
        # target_size es (W, H) para OpenCV
        img = cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR).astype(np.float32)
        img /= 255.0
        # El modelo espera [Batch, H, W, C]
        return img[None, ...]

    def __inference(self, image: np.ndarray) -> tuple:
        return tuple(self.session.run(None, {self.input_name: image}))


class OnnxPostProcessingNode(PipelineNode):
    """
    Post-procesa las salidas del modelo escalando los resultados de vuelta
    a la resolución original del usuario.
    """
    def __init__(self, name: str = "postproc_node"):
        super().__init__(name)

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        outputs = context.get('raw_outputs')
        orig_size = context.get('original_shape')      # (W, H)
        inf_size = context.get('inference_shape')     # (W, H)

        if any(v is None for v in [outputs, orig_size, inf_size]):
            raise ValueError(f"[{self.name}] Faltan datos críticos en el contexto.")

        logging.info("[%s] Iniciando post-procesamiento...", self.name)
        
        polygons, mask_fine, mask_bg = self.__process_logic(
            outputs, 
            target_size=orig_size, 
            inference_size=inf_size
        )

        context['polygons'] = polygons
        context['mask_fine'] = mask_fine
        context['mask_bg'] = mask_bg
        
        logging.info("[%s] Finalizado. %d polígonos extraídos.", self.name, len(polygons))
        return context

    def __process_logic(self, outputs: tuple, target_size: tuple, inference_size: tuple) -> tuple:
        # 1. Desempaquetar (Asumiendo salida del modelo con la forma de la inferencia)
        o_class, o_ncs = outputs[0][0], outputs[1][0]
        
        prob_fine = o_class[..., 0]
        prob_stones = self.__smooth_channel(o_class[..., 1])
        prob_centers = self.__smooth_channel(o_ncs[..., 1])
        prob_edges = self.__remap(o_ncs[..., 2])

        # 2. Umbrales binarios
        mask_stones_area = prob_stones >= 0.75
        centers_bin = np.logical_and(prob_centers >= 0.1, prob_edges < 0.25)
        edges_bin = prob_edges >= 0.25

        # 3. Watershed (Segmentación de instancias)
        labels = self.__apply_watershed(centers_bin, edges_bin, mask=mask_stones_area)

        # 4. Extraer polígonos escalando de inference_size -> target_size
        polygons = self.__extract_polygons(labels, target_size, inference_size)

        # 5. Máscaras de material fino y fondo
        mask_fine_raw = prob_fine * np.logical_not(edges_bin)
        mask_fine = (mask_fine_raw > 0.78).astype(np.uint8) * 255
        mask_bg = (prob_fine > 0.78).astype(np.uint8) * 255

        # Redimensionar máscaras a tamaño original
        mask_fine = cv2.resize(mask_fine, target_size, interpolation=cv2.INTER_AREA)
        mask_bg = cv2.resize(mask_bg, target_size, interpolation=cv2.INTER_AREA)

        return polygons, mask_fine, mask_bg

    def __remap(self, x: np.ndarray) -> np.ndarray:
        diff = x.max() - x.min()
        return np.zeros_like(x) if diff < 1e-6 else (x - x.min()) / diff

    def __smooth_channel(self, channel: np.ndarray) -> np.ndarray:
        return cv2.blur(self.__remap(channel), (3, 3))

    def __apply_watershed(self, centers_bin, edges_bin, mask) -> np.ndarray:
        centers_u8 = (centers_bin.astype(np.uint8) * 255)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        centers_u8 = cv2.morphologyEx(centers_u8, cv2.MORPH_CLOSE, kernel)
        
        _, markers = cv2.connectedComponents(centers_u8)
        edges_inv_u8 = ((~edges_bin).astype(np.uint8) * 255)
        dist_transform = cv2.distanceTransform(edges_inv_u8, cv2.DIST_LABEL_PIXEL, 0)

        return watershed(-dist_transform, markers, mask=mask, watershed_line=True)

    def __extract_polygons(self, labels: np.ndarray, target_size: tuple, inference_size: tuple) -> list:
        polygons = []
        w_t, h_t = target_size
        w_i, h_i = inference_size
        
        # Factor de escala: Resolución Original / Resolución de Inferencia
        scale_factor = np.array([w_t / w_i, h_t / h_i], dtype=np.float32)

        for i in np.unique(labels):
            if i <= 0: continue

            lbl_mask = (labels == i).astype(np.uint8)
            contours, _ = cv2.findContours(lbl_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for cnt in contours:
                if len(cnt) < 3: continue
                
                # Simplificación
                epsilon = 0.002 * cv2.arcLength(cnt, True)
                cnt_approx = cv2.approxPolyDP(cnt, epsilon, True)
                
                if len(cnt_approx) < 3: continue
                
                # Escalar puntos a la resolución original
                points_scaled = cnt_approx.squeeze() * scale_factor
                
                try:
                    poly = Polygon(points_scaled)
                    if not poly.is_valid: poly = poly.buffer(0)
                    if not poly.is_empty:
                        if poly.geom_type == 'MultiPolygon':
                            polygons.extend([g for g in poly.geoms if not g.is_empty])
                        else:
                            polygons.append(poly)
                except Exception:
                    continue
        return polygons