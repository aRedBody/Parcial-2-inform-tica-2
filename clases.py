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
