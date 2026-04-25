"""
clases.py
Sistema de Exploracion Neuroambiental - Parcial 2 Informatica II 2026-1
Clases principales y funciones de validacion numerica.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.io as sio

CARPETA_GRAFICAS = "graficas"
os.makedirs(CARPETA_GRAFICAS, exist_ok=True)


# ====================================================================
#  FUNCIONES DE VALIDACION NUMERICA
# ====================================================================

def validar_entero(mensaje, min_val=None, max_val=None):
    """Solicita un entero validando rango opcional."""
    while True:
        try:
            v = int(input(mensaje))
            if min_val is not None and v < min_val:
                print(f"  Error: el valor debe ser >= {min_val}.")
                continue
            if max_val is not None and v > max_val:
                print(f"  Error: el valor debe ser <= {max_val}.")
                continue
            return v
        except ValueError:
            print("  Error: ingrese un numero entero valido.")


def validar_opcion(mensaje, opciones):
    """Solicita una opcion dentro de una lista valida."""
    while True:
        r = input(mensaje).strip()
        if r in opciones:
            return r
        print(f"  Opcion no valida. Elija entre: {opciones}")


def validar_ruta(mensaje, extension):
    """Valida que el archivo exista y tenga la extension correcta."""
    while True:
        ruta = input(mensaje).strip().strip('"').strip("'")
        if not os.path.isfile(ruta):
            print(f"  Error: archivo no encontrado: '{ruta}'")
            continue
        if not ruta.lower().endswith(extension):
            print(f"  Error: se esperaba un archivo '{extension}'.")
            continue
        return ruta


def validar_columna(mensaje, columnas):
    """Valida que la columna ingresada exista en el DataFrame."""
    while True:
        col = input(mensaje).strip()
        if col in columnas:
            return col
        print(f"  Columna no encontrada. Disponibles: {list(columnas)}")


def _ruta_grafica(tipo, nombre_id, descripcion):
    """Construye la ruta de guardado: graficas/{tipo}_{id}_{descripcion}.png"""
    nombre_archivo = f"{tipo}_{nombre_id}_{descripcion}.png"
    return os.path.join(CARPETA_GRAFICAS, nombre_archivo)


# ====================================================================
#  CLASE AnalizadorSIATA
# ====================================================================

class AnalizadorSIATA:
    """Analiza archivos CSV de calidad del aire del sistema SIATA."""

    def __init__(self, ruta, nombre_id="siata"):
        self._ruta     = ruta
        self._nombre   = os.path.basename(ruta)
        self._id       = nombre_id
        self._df       = pd.read_csv(ruta, parse_dates=['fecha_hora'])
        self._df.set_index('fecha_hora', inplace=True)
        print(f"\n[SIATA] '{self._nombre}' cargado como objeto '{self._id}'.")
        print(f"  Registros: {len(self._df)}  |  Columnas: {list(self._df.columns)}")

    def mostrar_info(self):
        """Muestra info() y describe() del DataFrame."""
        sep = "-" * 55
        print(f"\n{sep}\n INFO - objeto '{self._id}' | archivo: {self._nombre}\n{sep}")
        self._df.info()
        print(f"\n{sep}\n ESTADISTICAS DESCRIPTIVAS\n{sep}")
        print(self._df.describe())

    def graficar_columna(self, columna):
        """Plot temporal, boxplot e histograma en 3 subplots."""
        datos = self._df[columna].dropna()

        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        fig.suptitle(
            f"SIATA | Objeto: '{self._id}' | Columna: '{columna}' | Archivo: {self._nombre}",
            fontsize=11
        )

        axes[0].plot(self._df.index, self._df[columna], lw=0.8, color='steelblue')
        axes[0].set_title("Serie temporal")
        axes[0].set_xlabel("Fecha")
        axes[0].set_ylabel(columna)
        axes[0].tick_params(axis='x', rotation=30)
        axes[0].grid(alpha=0.3)

        axes[1].boxplot(datos, patch_artist=True,
                        boxprops=dict(facecolor='lightcoral', alpha=0.75))
        axes[1].set_title("Boxplot")
        axes[1].set_xlabel(columna)
        axes[1].set_ylabel("Valor")

        axes[2].hist(datos, bins=30, color='seagreen', edgecolor='white', alpha=0.85)
        axes[2].set_title("Histograma")
        axes[2].set_xlabel(columna)
        axes[2].set_ylabel("Frecuencia")

        plt.tight_layout()
        archivo = _ruta_grafica("SIATA", self._id, f"graficos_{columna}")
        plt.savefig(archivo, dpi=150, bbox_inches='tight')
        print(f"[SIATA] Grafico guardado: '{archivo}'")
        plt.show()

    def operaciones(self, col1, col2, operacion='sumar'):
        """Tres operaciones: apply (norm. min-max), map (categoria), suma/resta."""
        sep = "-" * 55
        print(f"\n{sep}\n OPERACIONES | objeto '{self._id}' | {self._nombre}\n{sep}")

        mn, mx = self._df[col1].min(), self._df[col1].max()
        normalizada = self._df[col1].apply(
            lambda x: (x - mn) / (mx - mn) if pd.notna(x) else float('nan')
        )
        print(f"\n1. apply - Normalizacion min-max de '{col1}' (5 primeras filas):")
        print(normalizada.head().to_string())

        mediana = self._df[col1].median()
        categorizada = self._df[col1].map(
            lambda x: 'ALTO' if pd.notna(x) and x >= mediana else ('BAJO' if pd.notna(x) else None)
        )
        print(f"\n2. map - Categorizacion de '{col1}' (mediana={mediana:.3f}) (5 primeras):")
        print(categorizada.head().to_string())
        print(f"   Conteo: {categorizada.value_counts().to_dict()}")

        if operacion == 'sumar':
            resultado = self._df[col1] + self._df[col2]
            etiqueta  = f"{col1} + {col2}"
        else:
            resultado = self._df[col1] - self._df[col2]
            etiqueta  = f"{col1} - {col2}"
        print(f"\n3. {operacion.capitalize()} - {etiqueta} (5 primeras filas):")
        print(resultado.head().to_string())

        return normalizada, categorizada, resultado

    def graficar_remuestreado(self, columna):
        """Grafica la columna remuestreada a dias, meses y trimestres."""
        try:
            d = self._df[columna].resample('D').mean()
            m = self._df[columna].resample('ME').mean()
            q = self._df[columna].resample('QE').mean()
        except ValueError:
            d = self._df[columna].resample('D').mean()
            m = self._df[columna].resample('M').mean()
            q = self._df[columna].resample('Q').mean()

        fig, axes = plt.subplots(3, 1, figsize=(14, 10))
        fig.suptitle(
            f"SIATA | Objeto: '{self._id}' | Remuestreo: '{columna}' | Archivo: {self._nombre}",
            fontsize=11
        )
        specs = [
            (d, 'Diario',     'royalblue'),
            (m, 'Mensual',    'darkorange'),
            (q, 'Trimestral', 'forestgreen'),
        ]
        for ax, (serie, titulo, color) in zip(axes, specs):
            ax.plot(serie.index, serie.values, marker='o', ms=3,
                    lw=1.2, color=color, label=titulo)
            ax.set_title(titulo)
            ax.set_xlabel("Fecha")
            ax.set_ylabel(f"{columna} (promedio)")
            ax.legend()
            ax.grid(alpha=0.3)
            ax.tick_params(axis='x', rotation=30)

        plt.tight_layout()
        archivo = _ruta_grafica("SIATA", self._id, f"resample_{columna}")
        plt.savefig(archivo, dpi=150, bbox_inches='tight')
        print(f"[SIATA] Grafico de remuestreo guardado: '{archivo}'")
        plt.show()

    def columnas_numericas(self):
        """Devuelve lista de columnas numericas del DataFrame."""
        return list(self._df.select_dtypes(include='number').columns)


# ====================================================================
#  CLASE AnalizadorEEG
# ====================================================================

class AnalizadorEEG:
    """Analiza archivos MAT de EEG (Parkinson / control) a 1 kHz."""

    FS = 1000  # Hz

    def __init__(self, ruta, nombre_id="eeg"):
        self._ruta    = ruta
        self._nombre  = os.path.basename(ruta)
        self._id      = nombre_id
        self._mat     = sio.loadmat(ruta)
        self._data3d  = self._mat['data']          # (canales, muestras, epocas)
        self._data2d  = np.mean(self._data3d, axis=2)  # promedio epocas -> (canales, muestras)
        print(f"\n[EEG] '{self._nombre}' cargado como objeto '{self._id}'.")
        print(f"  Forma 3D: {self._data3d.shape}  (canales x muestras x epocas)")
        print(f"  Forma 2D (promedio epocas): {self._data2d.shape}")

    def mostrar_llaves(self):
        """Muestra las variables del archivo .mat usando whosmat."""
        sep = "-" * 55
        print(f"\n{sep}\n WHOSMAT | objeto '{self._id}' | {self._nombre}\n{sep}")
        for nombre, forma, tipo in sio.whosmat(self._ruta):
            print(f"  Variable: '{nombre}'  |  Forma: {forma}  |  Tipo: {tipo}")
