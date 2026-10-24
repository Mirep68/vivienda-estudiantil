"""
Scraper de Finca Raíz — anuncios de arriendo en el Valle de Aburrá.

IDEA CLAVE
    Finca Raíz está hecha con Next.js. Cada página de resultados trae, escondido
    en un <script id="__NEXT_DATA__">, un JSON con TODOS los datos de los 21
    anuncios de esa página (precio, área, habitaciones, coordenadas, amenidades...).
    En vez de "leer" el HTML visual etiqueta por etiqueta, sacamos ese JSON:
    es más rápido, más completo y no se rompe si cambian el diseño de la página.

QUÉ HACE
    Recorre  municipio x tipo de inmueble x página  y guarda cada página
    en data/raw/fincaraiz/<fecha>/  como JSON, SIN transformar (regla: raw no se toca).
    La conversión a tabla la hace otro script: fincaraiz_parser.py

BUENAS PRÁCTICAS QUE APLICA
    - Pausa entre peticiones (no saturar el servidor).
    - Reintentos con espera creciente si el servidor responde con error.
    - Reanudable: si se corta, al volver a correrlo salta las páginas ya bajadas.
    - Minimización de datos personales: NO guarda teléfonos ni nombres de
      particulares (Ley 1581 de 2012, habeas data).
    - Solo visita rutas permitidas por robots.txt (/arriendo/<tipo>/<municipio>/...).

TOPE DEL SITIO
    Finca Raíz solo muestra los primeros ~10.000 resultados de una búsqueda.
    Si una búsqueda tiene más, se recorre además por precio (/baratos y /lujosos)
    y se para sola cuando el sitio empieza a repetir páginas.

USO (desde la raíz del proyecto)
    Prueba corta:   python -m src.scrapers.fincaraiz --municipios medellin --tipos apartaestudios --max-paginas 2
    Completo:       python -m src.scrapers.fincaraiz
    Completar una corrida de otro día:  python -m src.scrapers.fincaraiz --fecha 2026-10-23
"""

import argparse
import json
import logging
import random
import re
import time
from datetime import date
from pathlib import Path

import requests

# --- Configuración ---------------------------------------------------------
RAIZ = Path(__file__).resolve().parents[2]
BASE_URL = "https://www.fincaraiz.com.co"

# Los 10 municipios del Valle de Aburrá, con el "slug" que usa Finca Raíz en la URL.
MUNICIPIOS = [
    "medellin", "bello", "itagui", "envigado", "sabaneta",
    "la-estrella", "caldas", "copacabana", "girardota", "barbosa",
]

# Tipos de inmueble que le sirven a un estudiante (descartamos locales, lotes, oficinas).
TIPOS = ["apartaestudios", "apartamentos", "casas", "habitaciones"]

# Cabeceras HTTP: le dicen al servidor qué "navegador" hace la petición.
# Sin esto, muchos sitios bloquean a los scripts.
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/130.0 Safari/537.36"
    ),
    "Accept-Language": "es-CO,es;q=0.9",
}

# El sitio solo deja ver los primeros ~10.000 resultados de una búsqueda
# (lo descubrimos en el diagnóstico: desde la página 477 repetía la misma página).
# Si una búsqueda tiene más, la recorremos además ordenada por precio en ambos
# sentidos: los 10.000 más baratos + los 10.000 más caros cubren hasta 20.000.
TOPE_RESULTADOS = 10_000
ORDENES_EXTRA = ["baratos", "lujosos"]

PATRON_NEXT_DATA = re.compile(
    r'<script id="__NEXT_DATA__" type="application/json"[^>]*>(.*?)</script>', re.S
)

log = logging.getLogger("fincaraiz")


# --- Funciones -------------------------------------------------------------
def construir_url(municipio: str, tipo: str, pagina: int, orden: str | None = None) -> str:
    """
    Arma la URL de una página de resultados.
        orden None       -> orden por defecto del sitio (popularidad)
        orden "baratos"  -> menor precio primero
        orden "lujosos"  -> mayor precio primero
    La página 1 no lleva sufijo /paginaN.
    """
    url = f"{BASE_URL}/arriendo/{tipo}/{municipio}/antioquia"
    if orden:
        url += f"/{orden}"
    return url if pagina == 1 else f"{url}/pagina{pagina}"


def descargar(session: requests.Session, url: str, intentos: int = 4) -> str | None:
    """
    Descarga una URL y devuelve el HTML (texto).
    Si el servidor responde 429 (demasiadas peticiones) o 5xx (error del servidor),
    espera cada vez más (backoff exponencial: 10 s, 20 s, 40 s...) y reintenta.
    Devuelve None si la página no existe (404).
    """
    for intento in range(1, intentos + 1):
        try:
            r = session.get(url, headers=HEADERS, timeout=30)
        except requests.RequestException as e:
            log.warning("Error de conexión (%s), intento %d/%d", e.__class__.__name__, intento, intentos)
            time.sleep(10 * 2 ** (intento - 1))
            continue

        if r.status_code == 200:
            return r.text
        if r.status_code == 404:
            return None
        if r.status_code in (403, 429) or r.status_code >= 500:
            espera = 10 * 2 ** (intento - 1)
            log.warning("HTTP %d en %s -> espero %d s (intento %d/%d)", r.status_code, url, espera, intento, intentos)
            time.sleep(espera)
            continue
        log.warning("HTTP %d inesperado en %s", r.status_code, url)
        return None

    raise RuntimeError(
        f"El servidor sigue rechazando las peticiones ({url}). "
        "Detén el scraper, espera unos minutos y vuelve a correrlo (reanuda solo)."
    )


def extraer_resultados(html: str) -> dict | None:
    """Saca el JSON de __NEXT_DATA__ y devuelve la parte con los anuncios."""
    m = PATRON_NEXT_DATA.search(html)
    if not m:
        return None
    datos = json.loads(m.group(1))
    try:
        return datos["props"]["pageProps"]["fetchResult"]["searchFast"]
    except (KeyError, TypeError):
        return None


def limpiar_datos_personales(anuncio: dict) -> dict:
    """
    Quita datos personales del anunciante antes de guardar.
    De una inmobiliaria conservamos el nombre (es una empresa);
    de un particular, solo que es particular.
    También quitamos la lista completa de fotos (pesa mucho; 'img' ya trae la principal).
    """
    a = dict(anuncio)
    dueño = a.get("owner") or {}
    es_particular = bool(dueño.get("particular"))
    a["owner"] = {
        "type": dueño.get("type"),
        "particular": es_particular,
        "name": None if es_particular else dueño.get("name"),
    }
    a.pop("images", None)
    return a


def recorrer_busqueda(session, municipio, tipo, orden, max_paginas, pausa, carpeta: Path):
    """
    Recorre TODAS las páginas de una búsqueda (municipio + tipo + orden).
    Devuelve (total que reporta el sitio, anuncios nuevos descargados, ids únicos vistos).

    Protección contra el tope: si una página trae SOLO anuncios ya vistos en esta
    búsqueda, el sitio está repitiendo resultados -> paramos.
    """
    etiqueta = orden or "popularidad"
    prefijo = f"{municipio}__{tipo}" + (f"__{orden}" if orden else "")
    vistos: set[str] = set()
    total_sitio, nuevos = None, 0
    pagina = 1

    while max_paginas is None or pagina <= max_paginas:
        archivo = carpeta / f"{prefijo}__p{pagina:04d}.json"

        # Reanudar: si la página ya está en disco, la leemos en vez de bajarla.
        if archivo.exists():
            guardado = json.loads(archivo.read_text(encoding="utf-8"))
            origen = "disco"
        else:
            url = construir_url(municipio, tipo, pagina, orden)
            html = descargar(session, url)
            resultados = extraer_resultados(html) if html else None
            if not resultados or not resultados.get("data"):
                log.info("%-11s %-14s %-11s sin resultados en p%d -> fin", municipio, tipo, etiqueta, pagina)
                break
            guardado = {
                "fuente": "fincaraiz",
                "url": url,
                "municipio_busqueda": municipio,
                "tipo_busqueda": tipo,
                "orden": etiqueta,
                "pagina": pagina,
                "fecha_scraping": date.today().isoformat(),
                "paginatorInfo": resultados["paginatorInfo"],
                "data": [limpiar_datos_personales(a) for a in resultados["data"]],
            }
            archivo.write_text(json.dumps(guardado, ensure_ascii=False), encoding="utf-8")
            nuevos += len(guardado["data"])
            origen = "web"
            # Pausa aleatoria: no golpear el servidor a ritmo fijo.
            time.sleep(pausa + random.uniform(0, pausa))

        info = guardado["paginatorInfo"]
        total_sitio = info.get("total")
        ids = {str(a.get("id")) for a in guardado["data"]}
        ids_nuevos = ids - vistos

        if not ids_nuevos:
            log.warning(
                "%-11s %-14s %-11s p%d repite anuncios ya vistos -> el sitio llegó a su tope. Paro esta búsqueda.",
                municipio, tipo, etiqueta, pagina,
            )
            break
        vistos |= ids

        log.info(
            "%-11s %-14s %-11s p%-4d/%-4s %2d nuevos (%s) | únicos: %d de %s",
            municipio, tipo, etiqueta, pagina, info.get("lastPage"),
            len(ids_nuevos), origen, len(vistos), total_sitio,
        )
        if not info.get("hasMorePages"):
            break
        pagina += 1

    return total_sitio, nuevos, vistos


def scrapear(municipios, tipos, max_paginas, pausa, carpeta_salida: Path) -> None:
    carpeta_salida.mkdir(parents=True, exist_ok=True)
    session = requests.Session()   # reutiliza la conexión: más rápido y más amable con el servidor
    total_nuevos = 0
    resumen = []

    for municipio in municipios:
        for tipo in tipos:
            total, nuevos, vistos = recorrer_busqueda(session, municipio, tipo, None, max_paginas, pausa, carpeta_salida)
            total_nuevos += nuevos

            # Si la búsqueda supera el tope, la recorremos también ordenada por precio (ambos sentidos).
            if total and total > TOPE_RESULTADOS and max_paginas is None:
                log.info("%s/%s tiene %s resultados (> %d): recorro también por precio.", municipio, tipo, total, TOPE_RESULTADOS)
                for orden in ORDENES_EXTRA:
                    _, n, v = recorrer_busqueda(session, municipio, tipo, orden, max_paginas, pausa, carpeta_salida)
                    total_nuevos += n
                    vistos |= v

            if total:
                resumen.append((municipio, tipo, total, len(vistos)))

    # Resumen de cobertura: anuncios ÚNICOS obtenidos vs total que reporta el sitio.
    log.info("-" * 70)
    log.info("COBERTURA (únicos obtenidos / total del sitio)")
    for municipio, tipo, total, unicos in resumen:
        log.info("  %-11s %-14s %6d / %-6d  %5.1f%%", municipio, tipo, unicos, total, 100 * unicos / total)
    log.info("Anuncios descargados en esta corrida: %d", total_nuevos)
    log.info("Archivos en: %s", carpeta_salida)


def main() -> None:
    # argparse: permite pasar opciones desde la terminal (--municipios, --tipos...)
    p = argparse.ArgumentParser(description="Scraper de arriendos de Finca Raíz (Valle de Aburrá)")
    p.add_argument("--municipios", nargs="+", default=MUNICIPIOS, help="slugs de municipios")
    p.add_argument("--tipos", nargs="+", default=TIPOS, help="tipos de inmueble")
    p.add_argument("--max-paginas", type=int, default=None, help="límite de páginas por combinación (para pruebas)")
    p.add_argument("--pausa", type=float, default=2.0, help="segundos mínimos entre peticiones")
    p.add_argument(
        "--fecha", default=date.today().isoformat(),
        help="carpeta de corrida a usar/completar (AAAA-MM-DD). Por defecto, hoy.",
    )
    args = p.parse_args()

    # El log sale en la terminal Y se guarda en reports/logs/ (trazabilidad de cada corrida).
    carpeta_logs = RAIZ / "reports" / "logs"
    carpeta_logs.mkdir(parents=True, exist_ok=True)
    archivo_log = carpeta_logs / f"fincaraiz_{date.today().isoformat()}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(), logging.FileHandler(archivo_log, encoding="utf-8")],
    )
    log.info("Log guardándose en: %s", archivo_log)
    salida = RAIZ / "data" / "raw" / "fincaraiz" / args.fecha
    scrapear(args.municipios, args.tipos, args.max_paginas, args.pausa, salida)


if __name__ == "__main__":
    main()
