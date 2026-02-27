import cv2
import logging
from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D

from .base import PipelineNode


class ResultVisualizerNode(PipelineNode):
    """
    Visualiza los resultados del análisis superponiendo polígonos y máscaras
    sobre la imagen original completa, adaptándose a tamaños dinámicos.
    """
    def __init__(
        self,
        fill_alpha: float = 0.5,
        contour_width: float = 0.8,
        name: str = "visualizer_node"
    ):
        super().__init__(name)
        self.fill_alpha = fill_alpha
        self.contour_width = contour_width
        
        self.COLORS = {
            "P00-P20": (12/255, 143/255, 250/255),  # Azul
            "P20-P40": (12/255, 250/255, 96/255),   # Verde
            "P40-P60": (0/255, 250/255, 230/255),   # Cian
            "P60-P80": (250/255, 162/255, 25/255),  # Naranja
            "P80-P100": (250/255, 33/255, 25/255),  # Rojo
            "FINE": (0.792, 0, 1)                    # Violeta
        }

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        logging.info("[%s] Iniciando generación de la visualización...", self.name)
        
        # Obtenemos la imagen original (la de resolución completa)
        img = context.get('original_image', context.get('image'))
        rocks = context.get('rocks', [])
        mask_bg = context.get('mask_bg') # Esta máscara ya fue reescalada en el nodo anterior
        granulometry_result = context.get('granulometry_result')

        if img is None:
            raise ValueError(f"[{self.name}] No se encontró 'image' en el contexto.")
        
        h_orig, w_orig = img.shape[:2]
        
        # Determinamos los umbrales de tamaño (P-values)
        p_vals = {}
        if granulometry_result:
            if granulometry_result.modeled_curve:
                p_vals = granulometry_result.modeled_curve.p_values
            elif granulometry_result.real_curve:
                p_vals = granulometry_result.real_curve.p_values
        
        p20 = p_vals.get("P20", 0)
        p40 = p_vals.get("P40", 0)
        p60 = p_vals.get("P60", 0)
        p80 = p_vals.get("P80", 0)

        # Configuración de la figura
        # Para imágenes muy grandes, Matplotlib puede sufrir; ajustamos el DPI.
        fig, ax = plt.subplots(figsize=(12, 9), dpi=100)
        ax.imshow(img)

        # 1. Dibujar máscara de material fino/fondo
        if mask_bg is not None:
            # Creamos una capa RGBA del tamaño de la imagen original
            mask_rgba = np.zeros((h_orig, w_orig, 4))
            
            # Aseguramos que la máscara coincida con el tamaño (por si hubo redondeos)
            if mask_bg.shape[:2] != (h_orig, w_orig):
                mask_bg = cv2.resize(mask_bg, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)
            
            mask_rgba[mask_bg > 0] = to_rgba(self.COLORS["FINE"], alpha=self.fill_alpha)
            ax.imshow(mask_rgba)

        # 2. Dibujar polígonos de rocas
        for rock in rocks:
            # Clasificación por color según diámetro
            if rock.diameter <= p20: color_key = "P00-P20"
            elif rock.diameter <= p40: color_key = "P20-P40"
            elif rock.diameter <= p60: color_key = "P40-P60"
            elif rock.diameter <= p80: color_key = "P60-P80"
            else: color_key = "P80-P100"
            
            base_color = self.COLORS[color_key]
            
            # Obtener coordenadas (ya vienen en escala original desde OnnxPostProcessingNode)
            x_coords, y_coords = rock.contorno.exterior.xy
            
            # Relleno y contorno
            ax.fill(x_coords, y_coords, color=to_rgba(base_color, alpha=self.fill_alpha))
            ax.plot(x_coords, y_coords, color=base_color, linewidth=self.contour_width)

        # 3. Leyenda y Estética
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', label='FINO', markerfacecolor=self.COLORS["FINE"], markersize=10),
            Line2D([0], [0], marker='o', color='w', label='< P20', markerfacecolor=self.COLORS["P00-P20"], markersize=10),
            Line2D([0], [0], marker='o', color='w', label='P20-P40', markerfacecolor=self.COLORS["P20-P40"], markersize=10),
            Line2D([0], [0], marker='o', color='w', label='P40-P60', markerfacecolor=self.COLORS["P40-P60"], markersize=10),
            Line2D([0], [0], marker='o', color='w', label='P60-P80', markerfacecolor=self.COLORS["P60-P80"], markersize=10),
            Line2D([0], [0], marker='o', color='w', label='> P80', markerfacecolor=self.COLORS["P80-P100"], markersize=10)
        ]
        
        ax.set_xlim(0, w_orig)
        ax.set_ylim(h_orig, 0) # Invertido para formato imagen
        ax.axis('off') # Limpiamos ejes para mejor visualización
        
        ax.legend(
            handles=legend_elements, 
            ncol=3, # En 2 filas para que no sea tan largo
            loc='upper center', 
            bbox_to_anchor=(0.5, -0.05), 
            frameon=False
        )
        
        plt.tight_layout()

        context['output_figure'] = fig
        logging.info("[%s] Visualización generada a resolución original (%dx%d).", self.name, w_orig, h_orig)
        
        return context