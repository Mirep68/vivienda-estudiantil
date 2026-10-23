"""
Descarga la criminalidad por comuna de Medellín (MEData) y revisa el archivo.

    data/raw/seguridad/*.csv   <-  se guarda EXACTAMENTE como llega (raw no se toca)

Ejecutar:  python -m src.scrapers.seguridad_comunas

CONCEPTO: codificación de texto
    Un archivo de texto es una secuencia de bytes; la "codificación" dice cómo
    convertir esos bytes en letras. Si se lee con la equivocada, salen símbolos
    raros (Ã³ en vez de ó) o falla. Las más comunes: utf-8, latin-1 y utf-16.
    Por eso este script PRUEBA varias y muestra cuál funciona.
"""

import io
from pathlib import Path

import pandas as pd
import requests

RAIZ = Path(__file__).resolve().parents[2]
SALIDA = RAIZ / "data" / "raw" / "seguridad"

BASE = "https://medata.app.medellin.gov.co/sites/default/files/distribution"   # OJO: https
ARCHIVOS = {
    "criminalidad_comunas_anual.csv": f"{BASE}/1-027-23-000303/consolidado_cantidad_casos_criminalidad_en_comunas_por_anio.csv",
    "criminalidad_comunas_mensual.csv": f"{BASE}/1-027-23-000304/consolidado_cantidad_casos_criminalidad_en_comunas_por_anio_mes.csv",
}

CODIFICACIONES = ["utf-8-sig", "utf-16", "latin-1"]   # en este orden de probabilidad


def descargar(url: str) -> bytes:
    r = requests.get(url, timeout=120, headers={"User-Agent": "Mozilla/5.0 (proyecto academico vivienda-estudiantil)"})
    r.raise_for_status()
    return r.content


def leer_tabla(contenido: bytes):
    """Prueba codificaciones y separadores hasta que salgan >=3 columnas razonables."""
    for cod in CODIFICACIONES:
        try:
            texto = contenido.decode(cod)
        except UnicodeError:
            continue
        for sep in [",", ";", "\t"]:
            try:
                df = pd.read_csv(io.StringIO(texto), sep=sep)
            except Exception:
                continue
            if df.shape[1] >= 3:
                return df, cod, sep
    return None, None, None


def main() -> None:
    SALIDA.mkdir(parents=True, exist_ok=True)
    for nombre, url in ARCHIVOS.items():
        print("\n" + "=" * 78 + f"\n{nombre}\n" + "=" * 78)
        try:
            contenido = descargar(url)
        except Exception as e:
            print(f"ERROR al descargar: {e.__class__.__name__}: {e}")
            continue

        (SALIDA / nombre).write_bytes(contenido)
        print(f"Descargado: {len(contenido):,} bytes -> {SALIDA / nombre}")
        print(f"Primeros 60 bytes (crudos): {contenido[:60]!r}")

        df, cod, sep = leer_tabla(contenido)
        if df is None:
            print("No pude interpretarlo como tabla. Pégame la línea de 'Primeros 60 bytes'.")
            continue

        print(f"Codificación: {cod} | separador: {sep!r} | filas: {len(df):,} | columnas: {list(df.columns)}")
        print(df.head(8).to_string())
        for col in df.columns:
            # is_string_dtype detecta texto en cualquier versión de pandas
            # (en pandas 3 el texto ya no es 'object', por eso la prueba anterior no imprimía nada).
            if pd.api.types.is_string_dtype(df[col]) and df[col].nunique() <= 40:
                print(f"\nValores de '{col}': {sorted(df[col].dropna().astype(str).unique())}")
        for col in df.columns:
            if "fecha" in col.lower():
                print(f"\nRango de '{col}': {df[col].min()}  ->  {df[col].max()}")

        # Calidad: ¿cuántos casos no tienen comuna asignada ('SIN DATO')?
        sin = df["Codigo_comuna"].astype(str).str.strip().str.upper() == "SIN DATO"
        pct = df.loc[sin, "Cantidad_casos"].sum() / df["Cantidad_casos"].sum() * 100
        print(f"\nCasos SIN comuna asignada: {pct:.1f}% del total")

        # Casos por año (para ver si el último año está completo)
        anio = df["Fecha_hecho"].astype(str).str[:4]
        print("\nCasos por año (últimos 8):")
        print(df.groupby(anio)["Cantidad_casos"].sum().tail(8).to_string())


if __name__ == "__main__":
    main()
