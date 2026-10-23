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

USO (desde la raíz del proyecto)
    Prueba corta:   python -m src.scrapers.fincaraiz --municipios medellin --tipos apartaestudios --max-paginas 2
    Completo:       python -m src.scrapers.fincaraiz
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

PATRON_NEXT_DATA = re.compile(
    r'<script id="__NEXT_DATA__" type="application/json"[^>]*>(.*?)</script>', re.S
)

log = logging.getLogger("fincaraiz")


# --- Funciones -------------------------------------------------------------
def construir_url(municipio: str, tipo: str, pagina: int) -> str:
    """Arma la URL de una página de resultados. La página 1 no lleva sufijo."""
    url = f"{BASE_URL}/arriendo/{tipo}/{municipio}/antioquia"
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


def scrapear(municipios, tipos, max_paginas, pausa, carpeta_salida: Path) -> None:
    carpeta_salida.mkdir(parents=True, exist_ok=True)
    session = requests.Session()   # reutiliza la conexión: más rápido y más amable con el servidor
    total_anuncios = 0

    for municipio in municipios:
        for tipo in tipos:
            pagina = 1
            while max_paginas is None or pagina <= max_paginas:
                archivo = carpeta_salida / f"{municipio}__{tipo}__p{pagina:04d}.json"

                # Reanudar: si ya bajamos esta página hoy, la leemos del disco.
                if archivo.exists():
                    guardado = json.loads(archivo.read_text(encoding="utf-8"))
                    info = guardado["paginatorInfo"]
                    log.info("%-11s %-14s p%-4d ya existe (salto)", municipio, tipo, pagina)
                else:
                    url = construir_url(municipio, tipo, pagina)
                    html = descargar(session, url)
                    resultados = extraer_resultados(html) if html else None

                    if not resultados or not resultados.get("data"):
                        log.info("%-11s %-14s sin resultados en p%d -> siguiente tipo", municipio, tipo, pagina)
                        break

                    info = resultados["paginatorInfo"]
                    anuncios = [limpiar_datos_personales(a) for a in resultados["data"]]
                    guardado = {
                        "fuente": "fincaraiz",
                        "url": url,
                        "municipio_busqueda": municipio,
                        "tipo_busqueda": tipo,
                        "pagina": pagina,
                        "fecha_scraping": date.today().isoformat(),
                        "paginatorInfo": info,
                        "data": anuncios,
                    }
                    archivo.write_text(json.dumps(guardado, ensure_ascii=False), encoding="utf-8")
                    total_anuncios += len(anuncios)
                    log.info(
                        "%-11s %-14s p%-4d/%-4s %2d anuncios (total en el sitio: %s)",
                        municipio, tipo, pagina, info.get("lastPage"), len(anuncios), info.get("total"),
                    )
                    # Pausa aleatoria: parecerse a una persona y no golpear el servidor a ritmo fijo.
                    time.sleep(pausa + random.uniform(0, pausa))

                if not info.get("hasMorePages"):
                    break
                pagina += 1

    log.info("Listo. Anuncios nuevos descargados en esta corrida: %d", total_anuncios)
    log.info("Archivos en: %s", carpeta_salida)


def main() -> None:
    # argparse: permite pasar opciones desde la terminal (--municipios, --tipos...)
    p = argparse.ArgumentParser(description="Scraper de arriendos de Finca Raíz (Valle de Aburrá)")
    p.add_argument("--municipios", nargs="+", default=MUNICIPIOS, help="slugs de municipios")
    p.add_argument("--tipos", nargs="+", default=TIPOS, help="tipos de inmueble")
    p.add_argument("--max-paginas", type=int, default=None, help="límite de páginas por combinación (para pruebas)")
    p.add_argument("--pausa", type=float, default=2.0, help="segundos mínimos entre peticiones")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
    salida = RAIZ / "data" / "raw" / "fincaraiz" / date.today().isoformat()
    scrapear(args.municipios, args.tipos, args.max_paginas, args.pausa, salida)


if __name__ == "__main__":
    main()
