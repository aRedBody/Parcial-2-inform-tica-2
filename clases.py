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
