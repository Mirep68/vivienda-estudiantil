"""
Extrae las capas geográficas de referencia del servidor de mapas de la Alcaldía de Medellín.

    data/reference/comunas.geojson            23 comunas y corregimientos (polígonos)
    data/reference/barrios.geojson            barrios y veredas (polígonos)
    data/reference/poblacion_comunas.csv      población proyectada 2018-2030 por comuna
    reports/figures/mapa_comunas_barrios.png  mapa para verificar visualmente

Ejecutar:  python -m src.scrapers.capas_alcaldia

CONCEPTO: sistemas de coordenadas (CRS)
    Una coordenada solo tiene sentido si se sabe en qué "sistema" está.
    - EPSG:4326 (WGS84): latitud/longitud en GRADOS. Es el del GPS y el de los anuncios.
    - EPSG:9377 (MAGNA-SIRGAS Origen Nacional): el oficial de Colombia, en METROS.
      Es el que usa la Alcaldía.
    Pedimos las capas en 4326 (outSR=4326) para que coincidan con los anuncios.
    Para medir distancias en metros, más adelante convertiremos todo a 9377.

CONCEPTO: paginación en ArcGIS
    El servidor entrega como máximo N registros por consulta (maxRecordCount).
    Pedimos de a bloques con resultOffset (desde dónde) y resultRecordCount (cuántos).
"""

import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")   # dibujar en archivo, sin abrir ventanas
import geopandas as gpd  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import requests  # noqa: E402

from src.limpieza.fincaraiz import sin_tildes  # noqa: E402

RAIZ = Path(__file__).resolve().parents[2]
REF = RAIZ / "data" / "reference"
FIG = RAIZ / "reports" / "figures"

SERVIDOR = "https://www.medellin.gov.co/servidormapas/rest/services"
CAPAS = {
    "comunas": f"{SERVIDOR}/ordenamiento_ter/VM_POT48_Base/MapServer/3",
    "barrios": f"{SERVIDOR}/ordenamiento_ter/VM_POT48_Base/MapServer/4",
    "poblacion": f"{SERVIDOR}/mapas_nacionales/VC_Distribucion_Poblacional/MapServer/1",
}
TAMANO_BLOQUE = 500
HEADERS = {"User-Agent": "Mozilla/5.0 (proyecto academico vivienda-estudiantil)"}


def pedir(url: str, params: dict) -> dict:
    r = requests.get(url, params=params, headers=HEADERS, timeout=120)
    r.raise_for_status()
    datos = r.json()
    if "error" in datos:   # ArcGIS responde 200 aunque haya error; el error viene en el JSON
        raise RuntimeError(f"ArcGIS respondió error: {datos['error']}")
    return datos


def descargar_features(url_capa: str) -> list[dict]:
    """Descarga TODAS las entidades (features) de una capa, de a bloques."""
    total = pedir(f"{url_capa}/query", {"where": "1=1", "returnCountOnly": "true", "f": "json"})["count"]
    features, offset = [], 0
    while offset < total:
        bloque = pedir(f"{url_capa}/query", {
            "where": "1=1", "outFields": "*", "outSR": 4326, "f": "geojson",
            "orderByFields": "objectid",          # orden fijo: sin esto, la paginación puede repetir/saltar
            "resultOffset": offset, "resultRecordCount": TAMANO_BLOQUE,
        })["features"]
        if not bloque:
            break
        features += bloque
        offset += len(bloque)
        time.sleep(1)                             # cortesía con el servidor
    if len(features) != total:
        raise RuntimeError(f"Se esperaban {total} entidades y llegaron {len(features)}")
    return features


def a_geodataframe(features: list[dict]) -> gpd.GeoDataFrame:
    """Lista de features GeoJSON -> GeoDataFrame (tabla con columna 'geometry')."""
    gdf = gpd.GeoDataFrame.from_features(features, crs="EPSG:4326")
    gdf = gdf.drop(columns=[c for c in gdf.columns if c.startswith("st_")], errors="ignore")
    return gdf


def clave_nombre(texto) -> str:
    """
    Clave para cruzar por nombre aunque esté escrito distinto:
    'Laureles - Estadio', 'Laureles Estadio' -> 'laurelesestadio';  'Villa Hermosa', 'Villahermosa' -> 'villahermosa'
    """
    t = sin_tildes(texto)
    # La capa de comunas dice "Corregimiento de San Cristóbal"; la de población solo "San Cristóbal".
    if t.startswith("corregimiento de "):
        t = t[len("corregimiento de "):]
    for ch in "-_.,()":
        t = t.replace(ch, " ")
    return "".join(t.split())


def resumen(nombre: str, gdf: gpd.GeoDataFrame) -> None:
    oeste, sur, este, norte = gdf.total_bounds
    print(f"\n--- {nombre}: {len(gdf)} polígonos")
    print(f"    Columnas: {[c for c in gdf.columns if c != 'geometry']}")
    print(f"    Extensión: lon {oeste:.3f} a {este:.3f} | lat {sur:.3f} a {norte:.3f}  (debe ser ~ -75.7/-75.4 y 6.1/6.4)")
    invalidas = int((~gdf.is_valid).sum())
    print(f"    Geometrías inválidas: {invalidas}")


def main() -> None:
    REF.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)

    capas = {}
    for nombre, url in CAPAS.items():
        print(f"Descargando {nombre}...")
        capas[nombre] = a_geodataframe(descargar_features(url))
        resumen(nombre, capas[nombre])

    comunas, barrios, pob = capas["comunas"], capas["barrios"], capas["poblacion"]

    # --- Comunas: código numérico para cruzar después con la criminalidad ('3' y '03' -> 3)
    comunas["codigo_num"] = pd.to_numeric(comunas["codigo"], errors="coerce").astype("Int64")
    print("\nComunas y corregimientos:")
    print(comunas[["codigo", "nombre", "identificacion"]].sort_values("codigo").to_string(index=False))

    # Polígonos sin nombre (códigos 'SN..'): ¿qué tan grandes son y dónde están?
    # Para medir área en km² hay que pasar a un sistema en METROS (EPSG:9377).
    sin_nombre = comunas[comunas["nombre"].isna()]
    if len(sin_nombre):
        area_km2 = sin_nombre.to_crs(9377).area / 1e6
        print("\nPolígonos SIN nombre:")
        for (_, f), a in zip(sin_nombre.iterrows(), area_km2):
            p = f.geometry.representative_point()
            print(f"  {f['codigo']}: área {a:.3f} km² | punto interior lat {p.y:.5f}, lon {p.x:.5f}")
        print(f"  (Para comparar: la comuna más pequeña mide {(comunas.dropna(subset=['nombre']).to_crs(9377).area / 1e6).min():.2f} km²)")

    # --- Barrios: tipos y polígonos institucionales (universidades, hospitales...)
    print("\nBarrios por subtipo (subtipo_barriovereda):")
    print(barrios["subtipo_barriovereda"].value_counts().to_string())
    inst = barrios["codigo"].astype(str).str.startswith("Inst")
    print(f"Polígonos institucionales (código 'Inst_...'): {int(inst.sum())}  ej.: {barrios.loc[inst, 'nombre'].head(6).tolist()}")

    # --- Población: cruzar con comunas por NOMBRE (esta capa no trae código)
    pob["clave"] = pob["nombre"].map(clave_nombre)
    comunas["clave"] = comunas["nombre"].map(clave_nombre)
    cruce = pob.drop(columns="geometry").merge(comunas[["clave", "codigo", "codigo_num"]], on="clave", how="left")
    sin_pareja = cruce[cruce["codigo"].isna()]
    print(f"\nPoblación: {len(pob)} filas | cruzadas con una comuna: {len(cruce) - len(sin_pareja)} | sin pareja: {len(sin_pareja)}")
    if len(sin_pareja):
        print("  Sin pareja (revisar nombres):", sin_pareja["nombre"].tolist())
        print("  Nombres en la capa de comunas:", sorted(comunas["nombre"].dropna().tolist()))
    cols_pob = ["codigo", "codigo_num", "nombre"] + sorted(c for c in cruce.columns if c.startswith("total_"))
    print("\nPoblación 2018, 2024 y 2026 (primeras filas):")
    print(cruce[[c for c in ["codigo", "nombre", "total_2018", "total_2024", "total_2026"] if c in cruce.columns]]
          .sort_values("codigo").head(25).to_string(index=False))
    if "total_2026" in cruce:
        print(f"Población total de Medellín 2026 según la capa: {int(cruce['total_2026'].sum()):,}")

    # --- Guardar
    comunas.drop(columns="clave").to_file(REF / "comunas.geojson", driver="GeoJSON")
    barrios.to_file(REF / "barrios.geojson", driver="GeoJSON")
    cruce[cols_pob].to_csv(REF / "poblacion_comunas.csv", index=False, encoding="utf-8-sig")
    print(f"\nGuardado en {REF}: comunas.geojson, barrios.geojson, poblacion_comunas.csv")

    # --- Mapa de verificación
    fig, ax = plt.subplots(figsize=(8, 10))
    barrios.boundary.plot(ax=ax, color="lightgray", linewidth=0.4)
    comunas.boundary.plot(ax=ax, color="black", linewidth=1)
    for _, f in comunas.iterrows():
        p = f.geometry.representative_point()   # un punto garantizado DENTRO del polígono
        ax.annotate(f["codigo"], (p.x, p.y), ha="center", fontsize=8, color="darkred")
    uni = REF / "universidades.csv"
    if uni.exists():
        u = pd.read_csv(uni)
        ax.scatter(u["lon"], u["lat"], s=18, c="blue", marker="^", label="Universidades", zorder=3)
        ax.legend(loc="lower left")
    ax.set_title("Medellín: comunas (negro), barrios (gris) y universidades")
    ax.set_xlabel("Longitud"); ax.set_ylabel("Latitud")
    salida = FIG / "mapa_comunas_barrios.png"
    fig.savefig(salida, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Mapa: {salida}")


if __name__ == "__main__":
    main()
