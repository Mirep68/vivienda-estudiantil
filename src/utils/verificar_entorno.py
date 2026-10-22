"""
Verifica que el entorno de trabajo esté listo.

Ejecutar:  python -m src.utils.verificar_entorno
"""

import importlib
import platform
import shutil
import subprocess
import sys

# nombre para importar -> para qué la usamos en el proyecto
LIBRERIAS = {
    "requests": "descargar páginas web (scraping)",
    "bs4": "leer/recorrer HTML (BeautifulSoup)",
    "lxml": "motor rápido para interpretar HTML",
    "pandas": "tablas de datos",
    "numpy": "cálculo numérico con arreglos",
    "matplotlib": "gráficas",
    "seaborn": "gráficas estadísticas",
    "geopy": "geocodificación y distancias",
    "folium": "mapas interactivos",
    "tqdm": "barras de progreso",
    "dotenv": "leer configuración secreta (.env)",
}

print(f"Python {sys.version.split()[0]} en {platform.system()} {platform.release()}")
if sys.version_info < (3, 10):
    print("  AVISO: se recomienda Python 3.11 o 3.12")

print(f"Entorno virtual activo: {'sí' if sys.prefix != sys.base_prefix else 'NO (actívalo)'}\n")

faltan = []
for mod, uso in LIBRERIAS.items():
    try:
        m = importlib.import_module(mod)
        print(f"  OK  {mod:<11} {getattr(m, '__version__', ''):<10} {uso}")
    except ImportError:
        faltan.append(mod)
        print(f"  --  {mod:<11} {'FALTA':<10} {uso}")

print()
if shutil.which("nvidia-smi"):
    gpu = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
        capture_output=True, text=True,
    ).stdout.strip()
    print(f"GPU detectada: {gpu}")
else:
    print("No se encontró nvidia-smi (la GPU se configura más adelante, con PyTorch).")

print("\nTodo listo." if not faltan else f"\nFaltan: {', '.join(faltan)} -> pip install -r requirements.txt")
