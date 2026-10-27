"""
Sub-paso 4: distancias y entorno de cada vivienda.

    data/interim/fincaraiz_seguridad_<fecha>.csv  ->  data/processed/viviendas_<fecha>.csv
    data/reference/osm_metro.geojson, osm_parques.geojson, osm_servicios_<tipo>.geojson
    reports/figures/mapa_entorno.png

Variables nuevas (todas en KILÓMETROS, distancia en línea recta):
    dist_<SIGLA>_km          distancia a cada una de las 25 sedes universitarias
    dist_universidad_min_km  distancia a la sede más cercana  (+ universidad_cercana)
    dist_metro_km            estación de Metro, tranvía o metrocable más cercana (+ metro_cercano)
    dist_parque_km           parque más cercano (>= 0,5 ha)
    servicios_500m           servicios a <= 500 m (supermercados, farmacias y, si se pudieron
                             descargar, hospitales y clínicas) + una columna por tipo

CONCEPTOS
  - Distancias en METROS: todo se pasa a EPSG:9377 (MAGNA-SIRGAS, en metros).
    En grados no se puede medir: 1° de longitud no mide lo mismo que 1° de latitud.
  - Línea recta vs caminando: caminando se recorre ~1,2-1,4 veces la línea recta
    (las calles no son rectas). Es una aproximación que se declara en el artículo.
  - OSMnx + caché: la primera vez descarga de OpenStreetMap (Overpass API) y guarda
    una copia; las siguientes corridas la reutilizan.

Ejecutar:  python -u -m src.features.entorno              (sin servicios: usa solo lo ya descargado)
           python -u -m src.features.entorno --servicios  (intenta descargar también los servicios)
  -u = "unbuffered": muestra cada mensaje al instante (sin -u, al usar | Tee-Object
       Python acumula los mensajes y parece que el programa estuviera congelado).
"""

import argparse
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import geopandas as gpd  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import osmnx as ox  # noqa: E402
import pandas as pd  # noqa: E402
from shapely.geometry import box  # noqa: E402

RAIZ = Path(__file__).resolve().parents[2]
INTERIM = RAIZ / "data" / "interim"
PROCESSED = RAIZ / "data" / "processed"
REF = RAIZ / "data" / "reference"
FIG = RAIZ / "reports" / "figures"
GPS, METROS = "EPSG:4326", "EPSG:9377"
RADIO_SERVICIOS_M = 500
AREA_MIN_PARQUE_M2 = 5_000

ox.settings.use_cache = True
ox.settings.requests_timeout = 600     # consultas grandes: hasta 10 minutos por respuesta
ox.settings.cache_folder = str(RAIZ / "data" / "raw" / "osm_cache")


def distancias_matriz(xy_puntos: np.ndarray, xy_destinos: np.ndarray) -> np.ndarray:
    """
    Distancia euclidiana (en las unidades de entrada) entre CADA punto y CADA destino.
    Usa 'broadcasting' de NumPy: (n,1,2) - (1,m,2) -> (n,m,2), sin bucles de Python.
    """
    dif = xy_puntos[:, None, :] - xy_destinos[None, :, :]
    return np.sqrt((dif ** 2).sum(axis=2))


# Servidores públicos de Overpass (la API de consultas de OpenStreetMap).
# Si el principal está saturado, se prueban los otros.
SERVIDORES_OVERPASS = [
    "https://overpass-api.de/api",
    "https://overpass.kumi.systems/api",
    "https://overpass.private.coffee/api",
]


def consultar_overpass(poligono, tags: dict):
    """Pide datos a OSM con reintentos: espera 30 s, 60 s, 120 s... y rota entre servidores."""
    espera, ultimo_error = 30, None
    for intento in range(6):
        servidor = SERVIDORES_OVERPASS[intento % len(SERVIDORES_OVERPASS)]
        ox.settings.overpass_url = servidor
        try:
            return ox.features_from_polygon(poligono, tags)
        except Exception as e:
            ultimo_error = e
            print(f"    {servidor} falló ({e.__class__.__name__}). Espero {espera} s y reintento ({intento + 1}/6)...")
            time.sleep(espera)
            espera = min(espera * 2, 300)
    raise RuntimeError(f"OpenStreetMap no respondió tras 6 intentos: {ultimo_error}")


def descargar_osm(poligono, tags: dict, nombre: str) -> gpd.GeoDataFrame:
    destino = REF / f"osm_{nombre}.geojson"
    if destino.exists():
        print(f"  {nombre}: usando {destino.name}")
        return gpd.read_file(destino)
    g = consultar_overpass(poligono, tags).reset_index()
    cols = [c for c in ["osmid", "name", *tags.keys(), "geometry"] if c in g.columns]
    g = g[cols]
    for c in g.columns:                      # GeoJSON no admite listas en las columnas
        if c != "geometry" and g[c].map(lambda v: isinstance(v, list)).any():
            g[c] = g[c].astype(str)
    g.to_file(destino, driver="GeoJSON")
    print(f"  {nombre}: {len(g):,} elementos descargados de OpenStreetMap")
    return g


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--servicios", action="store_true", help="intentar descargar servicios de OSM")
    intentar_servicios = ap.parse_args().servicios
    inicio = time.time()

    entrada = sorted(INTERIM.glob("fincaraiz_seguridad_*.csv"))[-1]
    fecha = entrada.stem.replace("fincaraiz_seguridad_", "")
    df = pd.read_csv(entrada, dtype={"id": str, "comuna_codigo": str, "barrio_codigo": str})
    print(f"Anuncios: {len(df):,}")

    municipios = gpd.read_file(REF / "municipios_valle.geojson")
    valle = municipios.to_crs(GPS).geometry.union_all()
    # Para consultar OSM usamos el RECTÁNGULO que encierra el valle: el contorno detallado
    # tiene miles de vértices y hace la consulta mucho más pesada para el servidor.
    rectangulo = box(*municipios.to_crs(GPS).total_bounds)

    # --- Puntos de las viviendas en metros -------------------------------------
    con = df["lat"].notna()
    viv = gpd.GeoDataFrame(df.loc[con, ["id"]], geometry=gpd.points_from_xy(df.loc[con, "lon"], df.loc[con, "lat"]),
                           crs=GPS).to_crs(METROS)
    xy_viv = np.column_stack([viv.geometry.x, viv.geometry.y])

    # --- 1) Universidades ------------------------------------------------------------
    print("\n1) Universidades")
    uni = pd.read_csv(REF / "universidades.csv")
    # Una sola universidad sin coordenadas daña la distancia mínima de TODAS las viviendas
    # (en NumPy, el mínimo de algo con NaN es NaN). Mejor detenerse y avisar que seguir con datos falsos.
    sin_coord = uni[uni["lat"].isna() | uni["lon"].isna()]
    if len(sin_coord):
        raise SystemExit(
            f"\nERROR: {len(sin_coord)} sedes sin coordenadas: {sin_coord['sigla'].tolist()}\n"
            "Pon su lat/lon en config/universidades.csv y corre: python -m src.utils.geocodificar_universidades"
        )
    print(f"  {len(uni)} sedes, todas con coordenadas")
    uni_g = gpd.GeoDataFrame(uni, geometry=gpd.points_from_xy(uni["lon"], uni["lat"]), crs=GPS).to_crs(METROS)
    d_uni = distancias_matriz(xy_viv, np.column_stack([uni_g.geometry.x, uni_g.geometry.y])) / 1000
    nuevas = pd.DataFrame(d_uni.round(2), index=viv.index, columns=[f"dist_{s}_km" for s in uni["sigla"]])
    nuevas["dist_universidad_min_km"] = d_uni.min(axis=1).round(2)
    nuevas["universidad_cercana"] = uni["sigla"].to_numpy()[d_uni.argmin(axis=1)]

    # --- 2) Metro, tranvía y metrocable ---------------------------------------------------
    print("\n2) OpenStreetMap")
    metro = descargar_osm(valle, {"railway": ["station", "tram_stop"], "aerialway": "station"}, "metro")
    metro = metro.to_crs(METROS)
    metro["geometry"] = metro.geometry.centroid          # algunas estaciones vienen como polígonos
    historicas = metro["name"].fillna("").str.lower().str.contains("ferrocarril")
    print(f"  Excluidas (no son del sistema Metro): {metro.loc[historicas, 'name'].tolist()}")
    metro = metro[~historicas]
    tipo = metro["railway"].fillna("") if "railway" in metro else pd.Series("", index=metro.index)
    if "aerialway" in metro:
        tipo = tipo.where(tipo != "", "metrocable:" + metro["aerialway"].fillna(""))
    print(f"  Estaciones por tipo: {tipo.value_counts().to_dict()}")
    print(f"  Ejemplos: {metro['name'].dropna().sort_values().unique()[:15].tolist()}")
    print(f"  Sin nombre: {int(metro['name'].isna().sum())}")
    cerca = gpd.sjoin_nearest(viv[["geometry"]], metro[["name", "geometry"]], how="left", distance_col="d")
    cerca = cerca[~cerca.index.duplicated()]
    nuevas["dist_metro_km"] = (cerca["d"] / 1000).round(2)
    nuevas["metro_cercano"] = cerca["name"]

    # --- 3) Parques ---------------------------------------------------------------------
    parques = descargar_osm(valle, {"leisure": "park"}, "parques").to_crs(METROS)
    parques = parques[parques.geom_type.isin(["Polygon", "MultiPolygon"]) & (parques.area >= AREA_MIN_PARQUE_M2)]
    print(f"  parques de al menos {AREA_MIN_PARQUE_M2 / 10_000:.1f} ha: {len(parques):,}")
    cp = gpd.sjoin_nearest(viv[["geometry"]], parques[["geometry"]], how="left", distance_col="d")
    nuevas["dist_parque_km"] = (cp[~cp.index.duplicated()]["d"] / 1000).round(2)   # 0 si está dentro del parque

    # --- 4) Servicios básicos a <= 500 m (opcional) --------------------------------------------
    # Depende de los servidores públicos de OSM, que pueden estar saturados por horas.
    # Sin --servicios solo se usan los tipos que YA estén descargados en data/reference.
    tipos_servicio = {"supermercados": {"shop": "supermarket"}, "farmacias": {"amenity": "pharmacy"},
                      "hospitales": {"amenity": "hospital"}, "clinicas": {"amenity": "clinic"}}
    partes, faltan = [], []
    for nombre, tags in tipos_servicio.items():
        en_disco = (REF / f"osm_servicios_{nombre}.geojson").exists()
        if not (en_disco or intentar_servicios):
            faltan.append(nombre)
            continue
        try:
            g = descargar_osm(rectangulo, tags, f"servicios_{nombre}")
            partes.append(g.assign(tipo_servicio=nombre)[["tipo_servicio", "geometry"]])
        except Exception as e:
            faltan.append(nombre)
            print(f"  AVISO: no se pudo descargar {nombre} ({e.__class__.__name__}).")
    if partes:
        # Se usan los tipos disponibles; la columna total dice exactamente cuáles incluye.
        zonas = viv[["geometry"]].copy()
        zonas["geometry"] = zonas.buffer(RADIO_SERVICIOS_M)    # círculo de 500 m alrededor de cada vivienda
        serv = gpd.GeoDataFrame(pd.concat(partes, ignore_index=True), crs=partes[0].crs).to_crs(METROS)
        serv["geometry"] = serv.geometry.centroid
        incluidos = sorted(serv["tipo_servicio"].unique())
        print(f"  Servicios incluidos: {serv['tipo_servicio'].value_counts().to_dict()}")
        if faltan:
            print(f"  (sin datos de: {faltan}; para intentarlo: --servicios)")
        unidos = gpd.sjoin(serv, zonas, how="inner", predicate="within")
        nuevas["servicios_500m"] = unidos.groupby("index_right").size().reindex(viv.index, fill_value=0)
        por_tipo = unidos.groupby(["index_right", "tipo_servicio"]).size().unstack(fill_value=0)
        for t in incluidos:
            nuevas[f"{t}_500m"] = por_tipo.get(t, pd.Series(dtype=int)).reindex(viv.index, fill_value=0)
    else:
        print(f"  Servicios NO incluidos (faltan: {faltan}). Para intentarlo: --servicios")

    df = df.join(nuevas)

    # --- Verificación -------------------------------------------------------------
    print("\n3) Resumen (km, excepto servicios):")
    cols = [c for c in ["dist_universidad_min_km", "dist_metro_km", "dist_parque_km", "servicios_500m"] if c in df]
    print(df[cols].describe(percentiles=[0.1, 0.5, 0.9]).T.round(2).to_string())
    print(f"\nViviendas a <= 1 km del Metro: {(df['dist_metro_km'] <= 1).mean() * 100:.1f} %")
    print(f"Viviendas a <= 2 km de alguna universidad: {(df['dist_universidad_min_km'] <= 2).mean() * 100:.1f} %")
    print("\nUniversidad más cercana (cuántas viviendas):")
    print(df["universidad_cercana"].value_counts().head(12).to_string())
    print("\nEstaciones más cercanas más frecuentes:")
    print(df["metro_cercano"].value_counts().head(10).to_string())
    print("\nDistancia mediana al Metro por municipio (km):")
    print(df.groupby("municipio")["dist_metro_km"].median().sort_values().round(2).to_string())

    PROCESSED.mkdir(parents=True, exist_ok=True)
    salida = PROCESSED / f"viviendas_{fecha}.csv"
    df.to_csv(salida, index=False, encoding="utf-8-sig")
    print(f"\nGuardado: {salida}  ({len(df):,} viviendas, {df.shape[1]} columnas)")

    # --- Mapa ---------------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 10))
    municipios.to_crs(METROS).boundary.plot(ax=ax, color="black", linewidth=0.6)
    parques.plot(ax=ax, color="green", alpha=0.4)
    sc = ax.scatter(xy_viv[:, 0], xy_viv[:, 1], c=nuevas["dist_metro_km"].clip(upper=5), s=1, cmap="viridis_r")
    metro.plot(ax=ax, color="red", markersize=8, label="Metro / tranvía / metrocable")
    uni_g.plot(ax=ax, color="blue", marker="^", markersize=30, label="Universidades")
    plt.colorbar(sc, ax=ax, label="Distancia al Metro (km, tope 5)")
    ax.legend(loc="lower left")
    ax.set_title("Viviendas, Metro, parques (verde) y universidades")
    ax.set_axis_off()
    fig.savefig(FIG / "mapa_entorno.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Mapa: {FIG / 'mapa_entorno.png'}")
    print(f"Tiempo total: {time.time() - inicio:.0f} s")


if __name__ == "__main__":
    main()
