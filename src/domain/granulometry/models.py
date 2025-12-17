import numpy as np

from typing import Tuple
from src.domain.schemas.models import RosinRammlerResult, SwebrecResult

class RosinRammler:
    def __init__(self):
        self.name = "RosinRammler"

    def fit(self, bin_centers: np.ndarray, curve: np.ndarray, eps: float = 1e-6) -> RosinRammlerResult:
        """
        Ajusta la curva de distribución Rosin-Rammler a los datos proporcionados.
        
        Ecuación: Y = 1 - exp(-(x/Xc)^n)
        Linealización: ln(-ln(1-Y)) = n * ln(x) - n * ln(Xc)
        Donde pendiente (m) = n, intercepto (b_reg) = -n * ln(Xc)
        """
        
        # 1. Preprocesamiento para Regresión Lineal
        # Solo usamos puntos donde 0 < curve < 1 y x > 0 para evitar log(0) o log(negativo)
        valid_mask = (curve > eps) & (curve < (1 - eps)) & (bin_centers > 0)
        
        if np.sum(valid_mask) < 2:
            # Si no hay suficientes puntos para una regresión (ej. imagen vacía o muy pocas rocas)
            # Retornamos una identidad segura (todo ceros) para no romper el flujo
            return self._create_empty_result(bin_centers)

        x_reg = bin_centers[valid_mask]
        y_reg_raw = curve[valid_mask]

        # 2. Transformación Log-Log (Linealización)
        # X_lin = ln(x)
        x_log = np.log(x_reg)
        
        # Y_lin = ln(-ln(1 - y))
        # Nota: divisor = 1 - curve. Si curve es % Pasante.
        divisor = 1 - y_reg_raw
        y_log = np.log(-np.log(divisor))

        # 3. Regresión Lineal
        m, b_reg = self.linear_regression(x_log, y_log)

        # 4. Extracción de Parámetros (n y Xc)
        # m = n
        n_param = m
        
        # b_reg = -n * ln(Xc)  =>  ln(Xc) = -b_reg / n  =>  Xc = exp(-b_reg / n)
        # Evitamos división por cero si m es muy pequeño
        if abs(n_param) < 1e-9:
            xc_param = 0.0
        else:
            xc_param = np.exp(-b_reg / n_param)

        # 5. Cálculo de métricas extra (Lógica original mantenida)
        x_max = float(bin_centers.max())
        x_50 = float(np.median(bin_centers)) # Nota: Esto es mediana del eje X, no D50 real.
        
        # Cálculo de 'b' específico de tu lógica de negocio original
        # b = m * (2 * ln(2) * ln(x_max / x_50))
        # Protegemos x_50 para no dividir por cero
        safe_x50 = x_50 if x_50 > 1e-6 else 1.0
        safe_ratio = x_max / safe_x50 if safe_x50 > 0 else 1.0
        safe_ratio = safe_ratio if safe_ratio > 0 else 1.0 # Evitar log(negativo)
        
        b_metric = m * (2 * np.log(2) * np.log(safe_ratio))

        # 6. Generación de la Curva Modelada (Para TODOS los bin_centers)
        # Formula: 1 - exp(-(x / Xc)^n)
        if xc_param > 0:
            modeled_curve = 1 - np.exp(-np.power(bin_centers / xc_param, n_param))
        else:
            modeled_curve = np.zeros_like(bin_centers)

        # 7. Cálculo de P-Values (P10...P100)
        # x = Xc * (-ln(1 - P))^(1/n)
        p_values = {}
        target_ps = np.arange(0.1, 1.1, 0.1) # 0.1, 0.2 ... 1.0
        
        if xc_param > 0 and abs(n_param) > 1e-9:
            inv_n = 1 / n_param
            for idx, p in enumerate(target_ps, start=1):
                # Clamp p para evitar log(0) en P100 (p=1.0)
                p_safe = min(p, 1 - eps)
                term = -np.log(1 - p_safe)
                
                # Evitar raíz negativa si term < 0 (no debería pasar con el clamp)
                if term < 0: term = 0
                
                d_val = xc_param * (term ** inv_n)
                p_values[f"P{idx*10}"] = float(d_val)
        else:
            p_values = {f"P{x*10}": 0.0 for x in range(1, 11)}

        return RosinRammlerResult(
            curve=modeled_curve,
            p_values=p_values,
            granulometry_xc=xc_param,
            granulometry_n=n_param,
            granulometry_b=b_metric,
            granulometry_xmax=x_max,
            granulometry_x50=x_50
        )

    @staticmethod
    def linear_regression(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
        """
        Regresion lineal y = m*x + b
        Retorna (m, b)
        """
        # Solución de Mínimos Cuadrados
        A = np.vstack([x, np.ones(len(x))]).T
        m, b = np.linalg.lstsq(A, y, rcond=None)[0]
        return m, b

    def _create_empty_result(self, bin_centers: np.ndarray) -> RosinRammlerResult:
        """Helper para retornar un resultado vacío seguro."""
        return RosinRammlerResult(
            curve=np.zeros_like(bin_centers),
            p_values={f"P{x*10}": 0.0 for x in range(1, 11)},
            granulometry_xc=0.0,
            granulometry_n=0.0,
            granulometry_b=0.0,
            granulometry_xmax=0.0,
            granulometry_x50=0.0
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"

class Swebrec:
    name = "swebrec"

    def fit(self, bin_centers: np.ndarray, curve: np.ndarray, eps: float = 1e-6) -> SwebrecResult:
        """
        Ajusta la función Swebrec a los datos.
        
        La lógica original utiliza una linearización tipo Rosin-Rammler para estimar 'n',
        y luego calcula el parámetro 'b' de Swebrec basándose en x_max y x_50.
        """
        
        # 1. Validación y Máscara para Regresión (Igual que en RR)
        # Filtramos valores extremos para evitar log(0) o log(negativo)
        valid_mask = (curve > eps) & (curve < (1 - eps)) & (bin_centers > 0)
        
        if np.sum(valid_mask) < 2:
            return self._create_empty_result(bin_centers)

        x_reg = bin_centers[valid_mask]
        y_reg_raw = curve[valid_mask]

        # 2. Linearización (Log-Log) para obtener n y dprima
        # x = log(bin_centers)
        x_log = np.log(x_reg)
        
        # y = log(log(1 / (1 - curve))) -> Usamos 1 - curve como divisor
        divisor = 1 - y_reg_raw
        y_log = np.log(np.log(1 / divisor))

        # Regresión lineal
        n_param, intercept = self.linear_regression(x_log, y_log)
        
        # dprima = exp(-intercept / n)
        if abs(n_param) < 1e-9:
            dprima = 0.0
        else:
            dprima = np.exp(-intercept / n_param)

        # 3. Cálculo de Parámetros Swebrec (b, x_max, x_50)
        # x_max: Usamos el máximo de los centros de bins disponibles
        x_max = float(bin_centers.max()) if len(bin_centers) > 0 else 1.0
        
        # x_50: Mediana de los centros de bins (Lógica original)
        x_50 = float(np.median(bin_centers)) if len(bin_centers) > 0 else 0.5
        
        # Evitamos problemas matemáticos si x_50 es 0 o x_max/x_50 es problemático
        safe_x50 = x_50 if x_50 > 1e-6 else 1.0
        ratio_max_50 = x_max / safe_x50
        
        # log(1) es 0, evitamos que el logaritmo sea 0 si x_max == x_50
        if ratio_max_50 <= 1.0 + eps: 
            ratio_max_50 = 2.0 # Valor dummy para evitar división por cero en log

        log_ratio_max_50 = np.log(ratio_max_50)
        
        # b = n * (2 * ln(2) * ln(x_max / x_50))
        b_param = n_param * (2 * np.log(2) * log_ratio_max_50)

        # 4. Generación de la Curva Modelada Swebrec
        # Formula: 1 / (1 + ( log(x_max / x) / log(x_max / x_50) )^b )
        
        # Preparar array de salida
        modeled_curve = np.zeros_like(bin_centers, dtype=float)
        
        # Solo calculamos para x > 0 para evitar log(0)
        # Y para x <= x_max para mantener consistencia física (aunque log manejaria > x_max negativamente)
        calc_mask = (bin_centers > 1e-9)
        
        if np.any(calc_mask):
            x_vals = bin_centers[calc_mask]
            
            # Numerador del término interno: log(x_max / x)
            # Clip para evitar log(0) si x es muy grande o x es 0
            ratio_x = x_max / x_vals
            # Si x > x_max, ratio < 1, log es negativo. Swebrec suele definirse hasta x_max (donde P=100)
            # Forzamos que la curva sea 1.0 si x >= x_max
            valid_ratio_mask = ratio_x >= 1.0
            
            # Cálculo vectorizado parcial
            term_inner = np.zeros_like(x_vals)
            
            # Parte normal (x < x_max)
            if np.any(valid_ratio_mask):
                valid_x = x_vals[valid_ratio_mask]
                # log(x_max / x) / log(x_max / x_50)
                base = np.log(x_max / valid_x) / log_ratio_max_50
                term_inner[valid_ratio_mask] = np.power(base, b_param)
            
            # Aplicar formula final: 1 / (1 + term^b)
            # Manejo de overflow si term es muy grande
            with np.errstate(over='ignore'):
                res = 1.0 / (1.0 + term_inner)
            
            # Si x >= x_max, la curva debe ser 1 (o muy cercana)
            # En la formula, log(1) = 0 -> 1/(1+0) = 1. Funciona matemáticamente.
            
            modeled_curve[calc_mask] = res
            
            # Corrección explícita para valores que exceden x_max (asegurar 1.0)
            modeled_curve[bin_centers >= x_max] = 1.0

        # 5. Cálculo de P-Values (Usando la fórmula provista en tu código)
        # Formula: P_val = (x_max^2 * (x_max/x_50)^(-1/p)) / x_50
        # Donde p itera 0.1, 0.2 ... 1.0
        p_values = {}
        target_ps = np.arange(0.1, 1.1, 0.1)

        # Pre-calculamos constantes para el loop
        x_max_sq = x_max ** 2
        
        # Proteger divisiones
        safe_x50_div = safe_x50 if safe_x50 > 0 else 1.0
        
        for idx, p in enumerate(target_ps, start=1):
            # p va de 0.1 a 1.0. 
            # Exponente: -1/p. 
            # Nota: Si p es muy pequeño, exponente es grande negativo.
            exponent = -1.0 / p
            
            term = np.power(ratio_max_50, exponent)
            
            # (x_max^2 * term) / x_50
            d_val = (x_max_sq * term) / safe_x50_div
            
            p_values[f"P{idx*10}"] = float(d_val)

        return SwebrecResult(
            curve=modeled_curve,
            p_values=p_values,
            granulometry_xc=dprima, # Mapeamos dprima a xc para consistencia
            granulometry_n=n_param,
            granulometry_b=b_param,
            granulometry_xmax=x_max,
            granulometry_x50=x_50
        )

    @staticmethod
    def linear_regression(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
        """Regresion lineal y=m*x+b Retorna (m,b)"""
        A = np.vstack([x, np.ones(len(x))]).T
        m, b = np.linalg.lstsq(A, y, rcond=None)[0]
        return m, b

    def _create_empty_result(self, bin_centers: np.ndarray) -> SwebrecResult:
        return SwebrecResult(
            curve=np.zeros_like(bin_centers),
            p_values={f"P{x*10}": 0.0 for x in range(1, 11)},
            granulometry_xc=0.0,
            granulometry_n=0.0,
            granulometry_b=0.0,
            granulometry_xmax=0.0,
            granulometry_x50=0.0
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"