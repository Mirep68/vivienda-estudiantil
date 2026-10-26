"""
Sub-paso 3a: extraer delitos por municipio (Policía / MinDefensa, datos.gov.co)
y la población municipal proyectada (DANE).

    data/raw/seguridad/<delito>_antioquia.csv        delitos de Antioquia, todos los años
    data/raw/poblacion/PPED-AreaMun-2018-2042_VP.xlsx población proyectada por municipio

Este script solo EXTRAE y muestra cómo vienen los datos (columnas, formatos, años,
cómo se escriben los municipios). El cálculo del índice de seguridad va en el 3b,
cuando sepamos exactamente el formato.

Ejecutar:  python -m src.scrapers.seguridad_municipios

CONCEPTO: SoQL (el "SQL" de datos.gov.co)
    La API de Socrata acepta filtros en la URL, como una consulta SQL:
      $where=upper(departamento)='ANTIOQUIA'   -> solo Antioquia (filtra el SERVIDOR, no Python)
      $limit / $offset                          -> paginación de a bloques
      $order=:id                                -> orden fijo para que la paginación no repita filas
    Así bajamos ~10 % de los datos en vez de los 650.000 registros de todo el país.
"""

import time
from pathlib import Path

import pandas as pd
import requests

RAIZ = Path(__file__).resolve().parents[2]
RAW_SEG = RAIZ / "data" / "raw" / "seguridad"
RAW_POB = RAIZ / "data" / "raw" / "poblacion"

SOCRATA = "https://www.datos.gov.co/resource"
DATASETS = {
    "hurto_personas": "4rxi-8m8d",
    "hurto_residencias_comercio": "6sqw-8cg5",
    "hurto_motos_carros": "9vha-vh9n",
}
DANE_POBLACION = ("https://www.dane.gov.co/files/censo2018/proyecciones-de-poblacion/"
                  "Municipal/PPED-AreaMun-2018-2042_VP.xlsx")
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) proyecto academico vivienda-estudiantil"}
BLOQUE = 50_000
VALLE = ["medellin", "bello", "itagui", "envigado", "sabaneta", "estrella", "caldas", "copacabana", "girardota", "barbosa"]


def descargar_socrata(id_: str) -> pd.DataFrame:
    filas, offset = [], 0
    while True:
        params = {"$where": "upper(departamento)='ANTIOQUIA'", "$limit": BLOQUE, "$offset": offset, "$order": ":id"}
        r = requests.get(f"{SOCRATA}/{id_}.json", params=params, headers=HEADERS, timeout=180)
        r.raise_for_status()
        bloque = r.json()
        filas += bloque
        print(f"    {len(filas):,} filas...")
        if len(bloque) < BLOQUE:
            break
        offset += BLOQUE
        time.sleep(1)
    return pd.DataFrame(filas)


def revisar_delitos(nombre: str, df: pd.DataFrame) -> None:
    print(f"  Filas: {len(df):,} | Columnas: {list(df.columns)}")
    mcol = "municipio"
    nombres = df[mcol].astype(str)
    del_valle = sorted({m for m in nombres.unique() if any(v in m.lower().replace("í", "i").replace("ü", "u") for v in VALLE)})
    print(f"  Cómo se escriben los municipios del Valle: {del_valle}")
    cod = "codigo_dane" if "codigo_dane" in df else "cod_muni"
    ejemplo = df.loc[nombres.str.lower().str.contains("medell"), cod].astype(str).unique()[:3]
    print(f"  Código de Medellín en la columna '{cod}': {list(ejemplo)}")
    f = df["fecha_hecho"].astype(str)
    print(f"  Formato de fecha (ejemplos): {f.head(3).tolist()}")
    anio = pd.to_datetime(f, errors="coerce", dayfirst=not f.iloc[0][:4].isdigit()).dt.year
    print(f"  Años: {int(anio.min())} a {int(anio.max())}  | fechas no interpretables: {int(anio.isna().sum())}")
    if "tipo_de_hurto" in df:
        print(f"  Tipos de hurto: {df['tipo_de_hurto'].value_counts().to_dict()}")


def main() -> None:
    RAW_SEG.mkdir(parents=True, exist_ok=True)
    RAW_POB.mkdir(parents=True, exist_ok=True)

    for nombre, id_ in DATASETS.items():
        print(f"\n=== {nombre} ({id_}) ===")
        try:
            df = descargar_socrata(id_)
            df.to_csv(RAW_SEG / f"{nombre}_antioquia.csv", index=False, encoding="utf-8-sig")
            revisar_delitos(nombre, df)
        except Exception as e:
            print(f"  ERROR: {e.__class__.__name__}: {e}")

    print("\n=== Población DANE por municipio ===")
    destino = RAW_POB / "PPED-AreaMun-2018-2042_VP.xlsx"
    try:
        r = requests.get(DANE_POBLACION, headers=HEADERS, timeout=300)
        r.raise_for_status()
        destino.write_bytes(r.content)
        print(f"  Descargado: {len(r.content):,} bytes -> {destino}")
        hojas = pd.ExcelFile(destino).sheet_names
        print(f"  Hojas: {hojas}")
        vista = pd.read_excel(destino, sheet_name=hojas[0], header=None, nrows=14)
        print("  Primeras 14 filas de la primera hoja (sin interpretar encabezados):")
        print(vista.to_string())
    except Exception as e:
        print(f"  ERROR: {e.__class__.__name__}: {e}")


if __name__ == "__main__":
    main()
