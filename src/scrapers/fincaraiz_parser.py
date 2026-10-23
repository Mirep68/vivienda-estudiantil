"""
Parser de Finca Raíz: convierte los JSON crudos en UNA tabla ordenada.

    data/raw/fincaraiz/<fecha>/*.json   ->   data/interim/fincaraiz_<fecha>.csv

¿Por qué separado del scraper?
    Descargar es lento (horas); convertir es rápido (segundos). Si mañana queremos
    otra columna, re-ejecutamos solo el parser sobre los JSON ya descargados,
    sin volver a molestar al servidor.

Uso:
    python -m src.scrapers.fincaraiz_parser                 # usa la fecha más reciente
    python -m src.scrapers.fincaraiz_parser --fecha 2026-10-22
"""

import argparse
import json
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parents[2]
CARPETA_RAW = RAIZ / "data" / "raw" / "fincaraiz"
CARPETA_SALIDA = RAIZ / "data" / "interim"

# Finca Raíz codifica la antigüedad en rangos (lo verificamos contra su ficha técnica).
ANTIGUEDAD = {
    1: "menor a 1 año",
    2: "1 a 8 años",
    3: "9 a 15 años",
    4: "16 a 30 años",
    5: "más de 30 años",
}


def primero(lista, campo="name"):
    """Devuelve lista[0][campo] o None si la lista está vacía (evita errores)."""
    return lista[0].get(campo) if lista else None


def a_numero(valor):
    """Convierte a número; si viene vacío, None o '', devuelve None."""
    try:
        return float(valor) if valor not in (None, "", "None") else None
    except (TypeError, ValueError):
        return None


def aplanar(anuncio: dict, meta: dict) -> dict:
    """Convierte UN anuncio (JSON anidado) en UNA fila (diccionario plano)."""
    loc = anuncio.get("locations") or {}
    ficha = {t["field"]: t["value"] for t in anuncio.get("technicalSheet") or []}
    precio = anuncio.get("price") or {}
    admin = anuncio.get("commonExpenses") or {}
    amenidades = [f["name"] for f in anuncio.get("facilities") or []]

    return {
        "id": str(anuncio.get("id")),
        "fuente": "fincaraiz",
        "url": "https://www.fincaraiz.com.co" + (anuncio.get("link") or ""),
        "titulo": anuncio.get("title"),
        "tipo_inmueble": (anuncio.get("property_type") or {}).get("name"),
        "municipio": primero(loc.get("city")),
        "zona": primero(loc.get("zone")),
        "barrio": (loc.get("location_main") or {}).get("name"),
        "direccion": (anuncio.get("address") or "").strip() or None,
        "lat": a_numero(anuncio.get("latitude")),
        "lon": a_numero(anuncio.get("longitude")),
        "precio": a_numero(precio.get("amount")),
        "precio_con_admin": a_numero(precio.get("admin_included")),
        "administracion": a_numero(admin.get("amount")),
        "area_m2": a_numero(anuncio.get("m2apto")) or a_numero(anuncio.get("m2Built")),
        "habitaciones": a_numero(anuncio.get("bedrooms")),
        "banos": a_numero(anuncio.get("bathrooms")),
        "parqueaderos": a_numero(anuncio.get("garage")),
        "piso": a_numero(anuncio.get("floor")),
        "estrato": a_numero(anuncio.get("stratum")),
        "antiguedad_cod": anuncio.get("antiquity"),
        "antiguedad": ANTIGUEDAD.get(anuncio.get("antiquity")),
        "estado": ficha.get("construction_state_name") or None,
        "acepta_mascotas": ficha.get("allowPets") or None,
        "contrato_minimo": ficha.get("minimumContract") or None,
        "amenidades": "|".join(amenidades),      # separadas por | para caber en una celda
        "n_amenidades": len(amenidades),
        "descripcion": anuncio.get("description"),
        "tipo_anunciante": (anuncio.get("owner") or {}).get("type"),
        "inmobiliaria": (anuncio.get("owner") or {}).get("name"),
        "img": anuncio.get("img"),
        "n_imagenes": a_numero(anuncio.get("image_count")),
        "fecha_publicacion": anuncio.get("created_at"),
        "fecha_actualizacion": anuncio.get("updated_at"),
        "fecha_scraping": meta["fecha_scraping"],
        "municipio_busqueda": meta["municipio_busqueda"],
        "tipo_busqueda": meta["tipo_busqueda"],
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--fecha", help="carpeta de fecha a procesar (por defecto la más reciente)")
    args = p.parse_args()

    fechas = sorted(d.name for d in CARPETA_RAW.iterdir() if d.is_dir())
    if not fechas:
        raise SystemExit(f"No hay datos en {CARPETA_RAW}. Corre primero el scraper.")
    fecha = args.fecha or fechas[-1]
    archivos = sorted((CARPETA_RAW / fecha).glob("*.json"))
    print(f"Procesando {len(archivos)} páginas de {CARPETA_RAW / fecha}")

    filas = []
    for archivo in archivos:
        pagina = json.loads(archivo.read_text(encoding="utf-8"))
        filas.extend(aplanar(a, pagina) for a in pagina["data"])

    df = pd.DataFrame(filas)
    n_bruto = len(df)
    # Un mismo anuncio puede salir en varias páginas o búsquedas: nos quedamos con uno por id.
    df = df.drop_duplicates(subset="id", keep="first")
    print(f"Anuncios leídos: {n_bruto}  ->  únicos: {len(df)}  (duplicados quitados: {n_bruto - len(df)})")

    CARPETA_SALIDA.mkdir(parents=True, exist_ok=True)
    salida = CARPETA_SALIDA / f"fincaraiz_{fecha}.csv"
    # utf-8-sig: así Excel en Windows muestra bien las tildes si abres el CSV.
    df.to_csv(salida, index=False, encoding="utf-8-sig")
    print(f"Guardado: {salida}\n")

    # --- Resumen rápido (la verificación a fondo la hacemos en el notebook) ---
    print("Anuncios por municipio:")
    print(df["municipio"].value_counts().to_string(), "\n")
    print("Anuncios por tipo:")
    print(df["tipo_inmueble"].value_counts().to_string(), "\n")
    print("% de datos faltantes en columnas clave:")
    clave = ["precio", "area_m2", "habitaciones", "banos", "estrato", "lat", "lon", "piso", "antiguedad"]
    print((df[clave].isna().mean() * 100).round(1).to_string())


if __name__ == "__main__":
    main()
