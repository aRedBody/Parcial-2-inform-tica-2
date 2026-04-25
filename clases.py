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

    def sumar_canales(self, canales, min_m, max_m):
        """Suma 3 canales en el rango dado y grafica en 2 subplots."""
        t    = np.arange(min_m, max_m) / self.FS
        suma = np.sum([self._data2d[ch, min_m:max_m] for ch in canales], axis=0)
        nombres_canales = [c + 1 for c in canales]

        fig, axes = plt.subplots(2, 1, figsize=(14, 8))
        fig.suptitle(
            f"EEG | Objeto: '{self._id}' | Suma canales {nombres_canales} | "
            f"Muestras [{min_m}-{max_m}] | Archivo: {self._nombre}",
            fontsize=10
        )

        for ch in canales:
            axes[0].plot(t, self._data2d[ch, min_m:max_m],
                         label=f"Canal {ch + 1}", lw=0.9)
        axes[0].set_title(f"Canales seleccionados: {nombres_canales}")
        axes[0].set_xlabel("Tiempo (s)")
        axes[0].set_ylabel("Amplitud (uV)")
        axes[0].legend()
        axes[0].grid(alpha=0.3)

        axes[1].plot(t, suma, color='crimson', lw=1.0,
                     label=f"Suma canales {nombres_canales}")
        axes[1].set_title(f"Resultado: suma de canales {nombres_canales}")
        axes[1].set_xlabel("Tiempo (s)")
        axes[1].set_ylabel("Amplitud (uV)")
        axes[1].legend()
        axes[1].grid(alpha=0.3)

        plt.tight_layout()
        desc    = f"suma_canales_{{'_'.join(str(c) for c in nombres_canales)}}_m{min_m}a{max_m}"
        archivo = _ruta_grafica("EEG", self._id, desc)
        plt.savefig(archivo, dpi=150, bbox_inches='tight')
        print(f"[EEG] Grafico guardado: '{archivo}'")
        plt.show()
        return suma

    def estadisticas_3d(self, eje):
        """Promedio y std de la matriz 3D con stem en 2 subplots."""
        prom = np.mean(self._data3d, axis=eje).flatten()
        std  = np.std(self._data3d,  axis=eje).flatten()

        MAX_PTS = 300
        if len(prom) > MAX_PTS:
            paso = len(prom) // MAX_PTS
            prom = prom[::paso]
            std  = std[::paso]
            print(f"  [EEG] Mostrando {len(prom)} puntos (1 de cada {paso}).")

        idx = np.arange(len(prom))

        fig, axes = plt.subplots(2, 1, figsize=(14, 8))
        fig.suptitle(
            f"EEG | Objeto: '{self._id}' | Estadisticas 3D - eje {eje} | "
            f"Archivo: {self._nombre}",
            fontsize=10
        )

        axes[0].stem(idx, prom, linefmt='C0-', markerfmt='C0o', basefmt='k-')
        axes[0].set_title(f"Promedio a lo largo del eje {eje}")
        axes[0].set_xlabel("Indice")
        axes[0].set_ylabel("Amplitud media (uV)")
        axes[0].grid(alpha=0.3)

        axes[1].stem(idx, std, linefmt='C1-', markerfmt='C1o', basefmt='k-')
        axes[1].set_title(f"Desviacion estandar a lo largo del eje {eje}")
        axes[1].set_xlabel("Indice")
        axes[1].set_ylabel("Desv. estandar (uV)")
        axes[1].grid(alpha=0.3)

        plt.tight_layout()
        archivo = _ruta_grafica("EEG", self._id, f"estadisticas_eje{eje}")
        plt.savefig(archivo, dpi=150, bbox_inches='tight')
        print(f"[EEG] Grafico estadisticas guardado: '{archivo}'")
        plt.show()

    @property
    def n_canales(self):
        return self._data3d.shape[0]

    @property
    def n_muestras(self):
        return self._data3d.shape[1]


# ====================================================================
#  CLASE AlmacenObjetos  (puntos extra)
# ====================================================================

class AlmacenObjetos:
    """Almacena objetos SIATA/EEG con nombre para busqueda posterior."""

    def __init__(self):
        self._registro = {}

    def agregar(self, nombre, objeto):
        self._registro[nombre] = objeto
        print(f"  [Almacen] '{nombre}' guardado ({type(objeto).__name__}).")

    def buscar(self, nombre):
        obj = self._registro.get(nombre)
        if obj is None:
            print(f"  [Almacen] No existe ningun objeto con nombre '{nombre}'.")
        return obj

    def listar(self):
        if not self._registro:
            print("  [Almacen] No hay objetos guardados.")
            return
        sep = "-" * 55
        print(f"\n{sep}\n OBJETOS EN EL ALMACEN\n{sep}")
        for nom, obj in self._registro.items():
            print(f"  * Nombre: '{nom}'  |  Tipo: {type(obj).__name__}  "
                  f"|  Archivo: {getattr(obj, '_nombre', '?')}")
