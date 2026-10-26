"""
Sub-paso 3c: homicidios por municipio (Policía Nacional, estadística delictiva).

La Policía publica un Excel por año con TODOS los delitos registro por registro
(https://www.policia.gov.co/Estadistica-delictiva). Este script:
  1. Descarga los archivos de 2024 y 2025 (si ya están en disco, no los vuelve a bajar).
  2. Detecta solo el encabezado y las columnas (municipio, código DANE, delito, fecha, cantidad).
  3. Se queda con los delitos que contienen "HOMICIDIO" en los 10 municipios del Valle de Aburrá.
  4. Muestra qué tipos de homicidio hay, para decidir cuáles cuentan como violencia.

    data/raw/seguridad/policia_delitos_<año>.xlsx    archivo original (raw)
    data/raw/seguridad/homicidios_valle_<año>.csv    solo homicidios del Valle

Ejecutar:  python -m src.scrapers.homicidios_policia
(Los archivos pueden pesar decenas de MB: la lectura del Excel puede tardar unos minutos.)
"""

from pathlib import Path

import pandas as pd
import requests

from src.limpieza.fincaraiz import sin_tildes

RAIZ = Path(__file__).resolve().parents[2]
RAW = RAIZ / "data" / "raw" / "seguridad"
BASE = "https://www.policia.gov.co/sites/default/files/"
ARCHIVOS = {
    2024: BASE + "INFORMACI%C3%93N_DE_DELITOS_A_NIVEL_DE_REGISTRO_A%C3%91O_2024_3.xlsx",
    2025: BASE + "INFORMACI%C3%93N_DE_DELITOS_A_NIVEL_DE_REGISTRO_A%C3%91O_2025_3.xlsx",
}
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/130.0 Safari/537.36"}
DANE_VALLE = {"05001", "05079", "05088", "05129", "05212", "05266", "05308", "05360", "05380", "05631"}


def descargar(anio: int, url: str) -> Path:
    destino = RAW / f"policia_delitos_{anio}.xlsx"
    if destino.exists():
        print(f"  Ya existe {destino.name} ({destino.stat().st_size:,} bytes), no se descarga de nuevo")
        return destino
    with requests.get(url, headers=HEADERS, timeout=600, stream=True) as r:   # stream: descarga por partes
        r.raise_for_status()
        with open(destino, "wb") as f:
            for parte in r.iter_content(chunk_size=1 << 20):
                f.write(parte)
    print(f"  Descargado {destino.name} ({destino.stat().st_size:,} bytes)")
    return destino


def leer(ruta: Path) -> pd.DataFrame:
    hoja = pd.ExcelFile(ruta).sheet_names
    print(f"  Hojas: {hoja}")
    crudo = pd.read_excel(ruta, sheet_name=hoja[0], header=None, nrows=30)
    fila = next(i for i, f in crudo.iterrows()
                if any("municipio" in sin_tildes(v) for v in f.dropna().astype(str)))
    df = pd.read_excel(ruta, sheet_name=hoja[0], header=fila, dtype=str)
    print(f"  Encabezado en fila {fila} | filas: {len(df):,} | columnas: {list(df.columns)}")
    return df


def detectar(df: pd.DataFrame) -> dict:
    cols = {c: sin_tildes(c) for c in df.columns}
    def busca(*palabras):
        return next((c for c, n in cols.items() if all(p in n for p in palabras)), None)
    codigo = next((c for c in df.columns
                   if df[c].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(8).str[:5].eq("05001").any()), None)
    delito = busca("delito") or next((c for c in df.columns
                                      if df[c].astype(str).str.upper().str.contains("HOMICIDIO").any()), None)
    return {"codigo": codigo, "delito": delito, "municipio": busca("municipio"),
            "fecha": busca("fecha"), "cantidad": busca("cantidad")}


def main() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    for anio, url in ARCHIVOS.items():
        print(f"\n=== {anio} ===")
        try:
            df = leer(descargar(anio, url))
        except Exception as e:
            print(f"  ERROR: {e.__class__.__name__}: {e}")
            continue

        c = detectar(df)
        print(f"  Columnas detectadas: {c}")
        if None in (c["codigo"], c["delito"]):
            print("  No pude detectar el código DANE o el delito. Pégame la lista de columnas.")
            continue

        cod5 = df[c["codigo"]].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(8).str[:5]
        es_hom = df[c["delito"]].astype(str).str.upper().str.contains("HOMICIDIO")
        hom = df[es_hom & cod5.isin(DANE_VALLE)].copy()
        hom["codigo5"] = cod5[hom.index]
        hom["casos"] = pd.to_numeric(hom[c["cantidad"]], errors="coerce").fillna(1) if c["cantidad"] else 1

        print(f"\n  Tipos de delito con 'HOMICIDIO' en el Valle (casos):")
        print(hom.groupby(c["delito"])["casos"].sum().sort_values(ascending=False).to_string())
        print(f"\n  Casos por municipio y tipo:")
        print(hom.pivot_table(index="codigo5", columns=c["delito"], values="casos", aggfunc="sum", fill_value=0).to_string())
        if c["fecha"]:
            print(f"\n  Fechas (ejemplos): {hom[c['fecha']].head(3).tolist()}")

        salida = RAW / f"homicidios_valle_{anio}.csv"
        hom.to_csv(salida, index=False, encoding="utf-8-sig")
        print(f"  Guardado: {salida} ({len(hom):,} filas)")


if __name__ == "__main__":
    main()
