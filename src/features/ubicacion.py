"""
Sub-paso 2: ubicación oficial de cada anuncio (cruce espacial punto-en-polígono).

    data/interim/fincaraiz_limpio_<fecha>.csv  ->  data/interim/fincaraiz_ubicado_<fecha>.csv
    reports/figures/mapa_inconsistencias_ubicacion.png

Hace tres cosas:
  1. MUNICIPIO REAL: en qué municipio caen las coordenadas (límites de OSM).
     El municipio del anuncio se corrige con el real; el original queda en 'municipio_anuncio'.
     Anuncios cuyas coordenadas caen FUERA de los 10 municipios se eliminan
     (están fuera del área de estudio, igual que la regla 1 de limpieza).
  2. COMUNA (solo Medellín): polígonos de la Alcaldía. Si un punto cae en un retazo
     sin nombre (SN) o justo en un borde sin comuna, se asigna la más cercana.
  3. BARRIO OFICIAL (solo Medellín): reemplaza a los barrios genéricos del portal.

Requisito: haber corrido antes  python -m src.scrapers.limites_municipios
Ejecutar:  python -m src.features.ubicacion

CONCEPTO: cruce espacial (spatial join)
    Como un merge de pandas, pero la condición es geométrica:
    "el punto (vivienda) está DENTRO del polígono".  gpd.sjoin(..., predicate="within")
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import geopandas as gpd  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

RAIZ = Path(__file__).resolve().parents[2]
INTERIM = RAIZ / "data" / "interim"
REF = RAIZ / "data" / "reference"
FIG = RAIZ / "reports" / "figures"
CRS_GPS, CRS_METROS = "EPSG:4326", "EPSG:9377"


def dentro(puntos, poligonos):
    """sjoin 'within' sin duplicados (un punto en un borde exacto podría caer en dos polígonos)."""
    r = gpd.sjoin(puntos, poligonos, how="left", predicate="within")
    return r[~r.index.duplicated()]


def main() -> None:
    entrada = sorted(INTERIM.glob("fincaraiz_limpio_*.csv"))[-1]
    fecha = entrada.stem.replace("fincaraiz_limpio_", "")
    df = pd.read_csv(entrada, dtype={"id": str})
    print(f"Anuncios: {len(df):,}  ({entrada.name})")

    municipios = gpd.read_file(REF / "municipios_valle.geojson")[["municipio", "geometry"]]
    comunas = gpd.read_file(REF / "comunas.geojson")[["codigo", "nombre", "geometry"]]
    barrios = gpd.read_file(REF / "barrios.geojson")[["codigo", "nombre", "subtipo_barriovereda", "geometry"]]

    # Puntos (x = LONGITUD, y = latitud)
    con_coord = df.dropna(subset=["lat", "lon"])
    puntos = gpd.GeoDataFrame(con_coord[["id"]].copy(),
                              geometry=gpd.points_from_xy(con_coord["lon"], con_coord["lat"]), crs=CRS_GPS)
    for col in ["municipio_real", "comuna_codigo", "comuna_nombre", "barrio_codigo", "barrio_oficial", "subtipo"]:
        puntos[col] = pd.NA

    # 1) Municipio real
    m = dentro(puntos[["geometry"]], municipios.rename(columns={"municipio": "mr"}))
    puntos["municipio_real"] = m["mr"]

    # 2) Comuna y 3) barrio: solo para los puntos que caen en Medellín
    en_med = puntos["municipio_real"].eq("Medellín")
    pm = puntos.loc[en_med, ["geometry"]]

    c = dentro(pm, comunas.rename(columns={"codigo": "cc", "nombre": "cn"}))
    sin_comuna = c["cc"].isna() | c["cc"].astype(str).str.startswith("SN")
    if sin_comuna.any():
        reales = comunas[~comunas["codigo"].str.startswith("SN")].rename(columns={"codigo": "cc2", "nombre": "cn2"})
        cerca = gpd.sjoin_nearest(pm[sin_comuna].to_crs(CRS_METROS), reales.to_crs(CRS_METROS), how="left")
        cerca = cerca[~cerca.index.duplicated()]
        c.loc[sin_comuna, "cc"] = cerca["cc2"]
        c.loc[sin_comuna, "cn"] = cerca["cn2"]
    print(f"Puntos de Medellín sin comuna directa (retazo SN o borde) -> comuna más cercana: {int(sin_comuna.sum())}")
    puntos.loc[en_med, "comuna_codigo"] = c["cc"]
    puntos.loc[en_med, "comuna_nombre"] = c["cn"]

    b = dentro(pm, barrios.rename(columns={"codigo": "bc", "nombre": "bn", "subtipo_barriovereda": "st"}))
    puntos.loc[en_med, "barrio_codigo"] = b["bc"]
    puntos.loc[en_med, "barrio_oficial"] = b["bn"]
    puntos.loc[en_med, "subtipo"] = b["st"]

    df = df.merge(pd.DataFrame(puntos.drop(columns="geometry")), on="id", how="left")
    df["es_vereda"] = pd.to_numeric(df["subtipo"], errors="coerce").eq(2)
    df["barrio_institucional"] = df["barrio_codigo"].astype(str).str.startswith("Inst")
    df = df.drop(columns="subtipo")

    # --- Coherencia: municipio del anuncio vs municipio real ---------------------
    df["municipio_anuncio"] = df["municipio"]
    real = df["municipio_real"].fillna("FUERA DEL VALLE").where(df["lat"].notna(), "SIN COORDENADAS")
    print("\nMunicipio del ANUNCIO (filas) vs municipio REAL según coordenadas (columnas):")
    print(pd.crosstab(df["municipio_anuncio"], real, margins=True, margins_name="Total").to_string())

    corregido = df["municipio_real"].notna() & (df["municipio_real"] != df["municipio_anuncio"])
    fuera = df["lat"].notna() & df["municipio_real"].isna()
    print(f"\nMunicipio corregido con las coordenadas: {int(corregido.sum()):,}")
    print(f"Coordenadas fuera de los 10 municipios (se eliminan): {int(fuera.sum()):,}")
    print(f"Sin coordenadas (conservan el municipio del anuncio): {int(df['lat'].isna().sum()):,}")

    df.loc[df["municipio_real"].notna(), "municipio"] = df["municipio_real"]
    df["municipio_corregido"] = corregido
    mapa_df = df.copy()
    df = df[~fuera].drop(columns="municipio_real")

    # --- Resultados en Medellín -------------------------------------------------
    med = df[df["municipio"].eq("Medellín") & df["comuna_codigo"].notna()]
    print(f"\nAnuncios en Medellín con comuna: {len(med):,} | con barrio oficial: {med['barrio_oficial'].notna().sum():,}")
    if "barrio_generico" in df:
        g = df["barrio_generico"].fillna(False).astype(bool) & df["municipio"].eq("Medellín")
        print(f"Barrios genéricos en Medellín: {int(g.sum()):,} | ahora con barrio oficial: "
              f"{int((g & df['barrio_oficial'].notna()).sum()):,}")
    print(f"En veredas (zona rural): {int(med['es_vereda'].sum()):,}")
    print(f"En polígonos institucionales: {int(med['barrio_institucional'].sum()):,}")
    print("\nAnuncios por municipio (corregido):")
    print(df["municipio"].value_counts().to_string())
    print("\nAnuncios por comuna de Medellín:")
    print(med.groupby(["comuna_codigo", "comuna_nombre"]).size().rename("anuncios").to_string())

    salida = INTERIM / f"fincaraiz_ubicado_{fecha}.csv"
    df.to_csv(salida, index=False, encoding="utf-8-sig")
    print(f"\nGuardado: {salida}  ({len(df):,} anuncios)")

    # --- Mapa ---------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 10))
    municipios.boundary.plot(ax=ax, color="black", linewidth=0.9)
    comunas.boundary.plot(ax=ax, color="gray", linewidth=0.3)
    ok = mapa_df[mapa_df["municipio_real"].notna() & ~mapa_df["municipio_corregido"]]
    cor = mapa_df[mapa_df["municipio_corregido"]]
    fue = mapa_df[fuera.reindex(mapa_df.index, fill_value=False)]
    ax.scatter(ok["lon"], ok["lat"], s=1, c="lightgray", label=f"Coherente ({len(ok):,})")
    ax.scatter(cor["lon"], cor["lat"], s=8, c="orange", label=f"Municipio corregido ({len(cor):,})")
    ax.scatter(fue["lon"], fue["lat"], s=12, c="red", label=f"Fuera del Valle, eliminado ({len(fue):,})")
    ax.legend(loc="lower left", markerscale=3)
    ax.set_title("Municipio del anuncio vs municipio real (coordenadas)")
    fig.savefig(FIG / "mapa_inconsistencias_ubicacion.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Mapa: {FIG / 'mapa_inconsistencias_ubicacion.png'}")


if __name__ == "__main__":
    main()
