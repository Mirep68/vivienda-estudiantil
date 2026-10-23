"""
Verifica las fuentes de datos abiertos del proyecto (solo LEE, no guarda nada).

Para cada fuente imprime: si responde, cuántos registros tiene, sus columnas
y una fila de ejemplo. Con esa salida decidimos qué usar y cómo.

Ejecutar:  python -m src.utils.verificar_fuentes

CONCEPTO: API REST
    Una API es una "puerta" para pedirle datos a un servidor con un programa.
    Le mandas una URL con parámetros (?filtro=valor) y responde con datos
    (JSON). Es lo contrario a leer una página visual: no hay que interpretar
    HTML, y se puede repetir cada semana para tener datos al día.
"""

import requests

# Dos tipos de servidores en este script:
#   1) datos.gov.co  -> usa "Socrata" (SODA API): /resource/<id>.json
#   2) Alcaldía      -> usa "ArcGIS REST": /MapServer/<capa>/query
SOCRATA = "https://www.datos.gov.co/resource"
ARCGIS = "https://www.medellin.gov.co/servidormapas/rest/services/ordenamiento_ter/VM_POT48_Base/MapServer"

DATASETS_SOCRATA = {
    "wtnd-h3vi": "Criminalidad por COMUNA de Medellín, anual",
    "gcyu-chif": "Criminalidad por COMUNA de Medellín, mensual",
    "4rxi-8m8d": "Hurto a personas, Colombia por municipio (MinDefensa)",
    "6sqw-8cg5": "Hurto a residencias y comercio, Colombia por municipio (Policía)",
    "9vha-vh9n": "Hurto de motos y carros, Colombia por municipio (Policía)",
}

CAPAS_ARCGIS = {3: "Comunas y corregimientos", 4: "Barrios y veredas"}


def titulo(texto: str) -> None:
    print("\n" + "=" * 78 + f"\n{texto}\n" + "=" * 78)


def pedir(url: str, params: dict):
    """GET con tiempo límite; devuelve el JSON o lanza el error."""
    r = requests.get(url, params=params, timeout=60)
    r.raise_for_status()          # si el servidor responde error (404, 500...), falla aquí
    return r.json()


def revisar_socrata() -> None:
    titulo("A) datos.gov.co  (Socrata)")
    for id_, nombre in DATASETS_SOCRATA.items():
        print(f"\n--- {id_}: {nombre}")
        try:
            total = pedir(f"{SOCRATA}/{id_}.json", {"$select": "count(*)"})[0]["count"]
            muestra = pedir(f"{SOCRATA}/{id_}.json", {"$limit": 3})
            print(f"  Registros totales: {total}")
            print(f"  Columnas: {list(muestra[0].keys())}")
            print(f"  Ejemplo : {muestra[0]}")
        except Exception as e:
            print(f"  ERROR: {e.__class__.__name__}: {e}")


def revisar_arcgis() -> None:
    titulo("B) Alcaldía de Medellín  (ArcGIS REST) - polígonos")
    for capa, nombre in CAPAS_ARCGIS.items():
        print(f"\n--- Capa {capa}: {nombre}")
        try:
            n = pedir(f"{ARCGIS}/{capa}/query", {"where": "1=1", "returnCountOnly": "true", "f": "json"})["count"]
            # outSR=4326 pide las coordenadas en latitud/longitud normales (por defecto vienen en otro sistema).
            geo = pedir(
                f"{ARCGIS}/{capa}/query",
                {"where": "1=1", "outFields": "*", "outSR": 4326, "f": "geojson", "resultRecordCount": 3},
            )
            props = geo["features"][0]["properties"]
            print(f"  Polígonos: {n}")
            print(f"  Campos  : {list(props.keys())}")
            print(f"  Ejemplo : {props}")
            print(f"  Tipo geometría: {geo['features'][0]['geometry']['type']}")
        except Exception as e:
            print(f"  ERROR: {e.__class__.__name__}: {e}")


if __name__ == "__main__":
    revisar_socrata()
    revisar_arcgis()
    print("\nListo. Copia TODA esta salida y pégamela.")
