"""
Nodo del pipeline para la visualización de resultados.

Este módulo contiene el nodo responsable de generar una representación gráfica
de los resultados del análisis de granulometría.
"""
import logging
from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D

from src.nodes.base import PipelineNode


class ResultVisualizerNode(PipelineNode):
    """
    Genera una figura de Matplotlib que visualiza los resultados del análisis,
    superponiendo los polígonos de las rocas y las máscaras sobre la imagen original.
    """
    def __init__(
        self,
        fill_alpha: float = 0.5,
        contour_width: float = 0.8,
        name: str = "visualizer_node"
    ):
        """
        Inicializa el nodo de visualización.

        Args:
            fill_alpha (float): Nivel de transparencia para el relleno de los polígonos.
            contour_width (float): Ancho de línea para el contorno de los polígonos.
            name (str): Nombre del nodo.
        """
        super().__init__(name)
        self.fill_alpha = fill_alpha
        self.contour_width = contour_width
        
        # Define la paleta de colores para las categorías de tamaño.
        self.COLORS = {
            "P00-P20": (12/255, 143/255, 250/255),  # Azul
            "P20-P40": (12/255, 250/255, 96/255),   # Verde
            "P40-P60": (0/255, 250/255, 230/255),   # Cian
            "P60-P80": (250/255, 162/255, 25/255),  # Naranja
            "P80-P100": (250/255, 33/255, 25/255),  # Rojo
            "FINE": (0.792, 0, 1)                   # Violeta
        }

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Crea una figura de Matplotlib con los resultados del análisis.
        """
        logging.info("[%s] Iniciando generación de la visualización...", self.name)
        
        # Intentamos obtener la imagen original si existe; de lo contrario, usamos 'image'
        img = context.get('original_image', context.get('image'))
        rocks = context.get('rocks', [])
        mask_bg = context.get('mask_bg')
        granulometry_result = context.get('granulometry_result')

        if img is None:
            raise ValueError(f"[{self.name}] No se encontró 'image' en el contexto.")
        
        h, w = img.shape[:2]

        # --- INICIO HOT FIX 2160 CENTER CUT ---
        target_size = 2160
        
        # Calculamos el offset de forma independiente asegurando que no sea negativo.
        # Esto soluciona el problema cuando una dimensión ya es menor o igual a 2160.
        offset_x = max(0, (w - target_size) // 2)
        offset_y = max(0, (h - target_size) // 2)
        
        logging.info("[%s] Aplicando offset del Hot Fix: X=%d, Y=%d", self.name, offset_x, offset_y)
        # --- FIN HOT FIX ---

        # Prioriza los P-values del modelo ajustado; si no existen, usa los de la curva real.
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

        # Creación de la figura y ejes
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.imshow(img)

        # Dibuja la máscara de material fino.
        if mask_bg is not None:
            # Creamos un lienzo vacío del tamaño de la imagen original (4K, por ejemplo)
            mask_rgba = np.zeros((h, w, 4))
            h_crop, w_crop = mask_bg.shape[:2]
            
            # Seleccionamos la región de interés (ROI) correspondiente al recorte central
            roi = mask_rgba[offset_y:offset_y+h_crop, offset_x:offset_x+w_crop]
            
            # Aplicamos el color sobre la ROI usando la máscara binaria pequeña
            roi[mask_bg > 0] = to_rgba(self.COLORS["FINE"], alpha=self.fill_alpha)
            
            # Mostramos la máscara completa (transparente donde no hay material fino)
            ax.imshow(mask_rgba)

        # Itera sobre cada roca para dibujarla, coloreada según su tamaño.
        for rock in rocks:
            if rock.diameter <= p20: color_key = "P00-P20"
            elif rock.diameter <= p40: color_key = "P20-P40"
            elif rock.diameter <= p60: color_key = "P40-P60"
            elif rock.diameter <= p80: color_key = "P60-P80"
            else: color_key = "P80-P100"
            
            base_color = self.COLORS[color_key]
            
            # Obtenemos las coordenadas de la roca
            x_coords, y_coords = rock.contorno.exterior.xy
            
            # HOT FIX: Desplazamos las coordenadas sumando el offset
            x_coords = np.array(x_coords) + offset_x
            y_coords = np.array(y_coords) + offset_y
            
            # Dibuja el relleno y el contorno del polígono de la roca desplazado
            ax.fill(x_coords, y_coords, color=to_rgba(base_color, alpha=self.fill_alpha))
            ax.plot(x_coords, y_coords, color=base_color, linewidth=self.contour_width)

        # Configuración de la leyenda y apariencia del gráfico.
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', label='FINO', markerfacecolor=self.COLORS["FINE"], markersize=10),
            Line2D([0], [0], marker='o', color='w', label='< P20', markerfacecolor=self.COLORS["P00-P20"], markersize=10),
            Line2D([0], [0], marker='o', color='w', label='P20-P40', markerfacecolor=self.COLORS["P20-P40"], markersize=10),
            Line2D([0], [0], marker='o', color='w', label='P40-P60', markerfacecolor=self.COLORS["P40-P60"], markersize=10),
            Line2D([0], [0], marker='o', color='w', label='P60-P80', markerfacecolor=self.COLORS["P60-P80"], markersize=10),
            Line2D([0], [0], marker='o', color='w', label='> P80', markerfacecolor=self.COLORS["P80-P100"], markersize=10)
        ]
        ax.set_aspect('equal')
        ax.set_xlim(0, w)
        ax.set_ylim(h, 0)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.legend(handles=legend_elements, ncol=len(legend_elements), loc='lower center', bbox_to_anchor=(0.5, -0.1), frameon=False)
        
        plt.tight_layout()

        # Guarda la figura en el contexto
        context['output_figure'] = fig
        logging.info("[%s] Visualización generada y añadida al contexto.", self.name)
        
        return context