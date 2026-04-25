"""
main.py
Sistema de Exploracion Neuroambiental - Parcial 2 Informatica II 2026-1
Implementacion del menu principal con todas las funcionalidades del sistema.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from clases import (
    AnalizadorSIATA, AnalizadorEEG, AlmacenObjetos,
    validar_entero, validar_opcion, validar_ruta, validar_columna,
)

# Almacen global de objetos (puntos extra)
almacen = AlmacenObjetos()


# ====================================================================
#  SUBMENU SIATA
# ====================================================================

def menu_siata():
    ruta = validar_ruta("\nRuta del archivo CSV (SIATA): ", '.csv')
    nombre_id = input("Nombre para identificar este objeto: ").strip()
    if not nombre_id:
        nombre_id = "siata"

    obj = AnalizadorSIATA(ruta, nombre_id)
    almacen.agregar(nombre_id, obj)
    cols = obj.columnas_numericas()

    while True:
        print(f"""
+--------------------------------------------------+
|  SIATA - {nombre_id:<40}|
+--------------------------------------------------+
|  1. Informacion basica (info / describe)         |
|  2. Graficos de columna (plot / boxplot / hist)  |
|  3. Operaciones (apply, map, suma/resta)         |
|  4. Grafico de remuestreo (dia / mes / trimest.) |
|  0. Volver al menu principal                     |
+--------------------------------------------------+""")

        op = validar_opcion("Opcion: ", ['0', '1', '2', '3', '4'])

        if op == '1':
            obj.mostrar_info()

        elif op == '2':
            print(f"\n  Columnas disponibles: {cols}")
            col = validar_columna("  Columna a graficar: ", cols)
            obj.graficar_columna(col)

        elif op == '3':
            print(f"\n  Columnas disponibles: {cols}")
            c1 = validar_columna("  Primera columna: ", cols)
            c2 = validar_columna("  Segunda columna: ", cols)
            oper = validar_opcion("  Operacion (sumar/restar): ", ['sumar', 'restar'])
            obj.operaciones(c1, c2, oper)

        elif op == '4':
            print(f"\n  Columnas disponibles: {cols}")
            col = validar_columna("  Columna a remuestrear: ", cols)
            obj.graficar_remuestreado(col)

        elif op == '0':
            break


# ====================================================================
#  SUBMENU EEG
# ====================================================================

def menu_eeg():
    ruta = validar_ruta("\nRuta del archivo MAT (EEG): ", '.mat')
    nombre_id = input("Nombre para identificar este objeto: ").strip()
    if not nombre_id:
        nombre_id = "eeg"

    obj = AnalizadorEEG(ruta, nombre_id)
    almacen.agregar(nombre_id, obj)

    obj.mostrar_llaves()

    while True:
        print(f"""
+--------------------------------------------------+
|  EEG - {nombre_id:<42}|
+--------------------------------------------------+
|  1. Mostrar llaves del archivo (whosmat)         |
|  2. (a) Sumar 3 canales en rango de muestras     |
|  3. (b) Estadisticas 3D con stem                 |
|  0. Volver al menu principal                     |
+--------------------------------------------------+""")

        op = validar_opcion("Opcion: ", ['0', '1', '2', '3'])

        if op == '1':
            obj.mostrar_llaves()

        elif op == '2':
            n = obj.n_canales
            m = obj.n_muestras
            print(f"\n  Canales disponibles: 1 - {n}")
            print(f"  Muestras disponibles: 0 - {m - 1}  "
                  f"(duracion: {m / obj.FS:.2f} s a {obj.FS} Hz)")

            canales, vistos = [], set()
            for i in range(3):
                while True:
                    ch = validar_entero(f"  Canal {i + 1} (1-{n}): ", 1, n) - 1
                    if ch in vistos:
                        print("  Canal ya seleccionado, elija otro.")
                    else:
                        vistos.add(ch)
                        canales.append(ch)
                        break

            min_m = validar_entero(f"  Muestra minima (0 - {m - 2}): ", 0, m - 2)
            max_m = validar_entero(f"  Muestra maxima ({min_m + 1} - {m}): ", min_m + 1, m)
            obj.sumar_canales(canales, min_m, max_m)

        elif op == '3':
            print("\n  Ejes disponibles:")
            print(f"    0 -> a lo largo de canales  (resultado: muestras x epocas)")
            print(f"    1 -> a lo largo de muestras (resultado: canales x epocas)")
            print(f"    2 -> a lo largo de epocas   (resultado: canales x muestras)")
            eje = validar_entero("  Eje (0, 1 o 2): ", 0, 2)
            obj.estadisticas_3d(eje)

        elif op == '0':
            break


# ====================================================================
#  SUBMENU ALMACEN
# ====================================================================

def menu_almacen():
    while True:
        print("""
+--------------------------------------------------+
|            ALMACEN DE OBJETOS                    |
+--------------------------------------------------+
|  1. Listar todos los objetos                     |
|  2. Buscar objeto por nombre                     |
|  0. Volver al menu principal                     |
+--------------------------------------------------+""")

        op = validar_opcion("Opcion: ", ['0', '1', '2'])

        if op == '1':
            almacen.listar()

        elif op == '2':
            nombre = input("  Nombre a buscar: ").strip()
            obj = almacen.buscar(nombre)
            if obj:
                print(f"  Objeto encontrado: {type(obj).__name__} "
                      f"- archivo: {getattr(obj, '_nombre', '?')}")

        elif op == '0':
            break


# ====================================================================
#  MENU PRINCIPAL
# ====================================================================

def main():
    print("""
+==================================================================+
|        SISTEMA DE EXPLORACION NEUROAMBIENTAL                     |
|        Universidad de Antioquia - Bioingeniera                   |
|        Parcial 2  -  Informatica II  -  2026-1                   |
+==================================================================+""")

    while True:
        print("""
+--------------------------------------------------+
|  MENU PRINCIPAL                                  |
+--------------------------------------------------+
|  1. Cargar archivo CSV  - SIATA                  |
|  2. Cargar archivo MAT  - EEG                    |
|  3. Gestionar almacen de objetos                 |
|  0. Salir                                        |
+--------------------------------------------------+""")

        op = validar_opcion("Opcion: ", ['0', '1', '2', '3'])

        if op == '1':
            menu_siata()
        elif op == '2':
            menu_eeg()
        elif op == '3':
            menu_almacen()
        elif op == '0':
            print("\nCerrando el sistema. Hasta luego!\n")
            break


if __name__ == "__main__":
    main()
