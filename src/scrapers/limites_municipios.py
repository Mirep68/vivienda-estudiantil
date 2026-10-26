"""
Límites de los 10 municipios del Valle de Aburrá desde OpenStreetMap (OSM).

    data/reference/municipios_valle.geojson
    reports/figures/mapa_municipios_valle.png

Ejecutar:  python -m src.scrapers.limites_municipios

LIBRERÍA: OSMnx
    Permite descargar datos de OpenStreetMap (el "Wikipedia de los mapas") con
    Python: límites administrativos, calles, estaciones, parques...
    geocode_to_gdf("Envigado, Antioquia, Colombia") busca el lugar y devuelve
    su POLÍGONO como GeoDataFrame. Usamos la misma fuente para los 10 municipios
    para que los límites sean coherentes entre sí.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import geopandas as gpd  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import osmnx as ox  # noqa: E402
import pandas as pd  # noqa: E402

RAIZ = Path(__file__).resolve().parents[2]
REF = RAIZ / "data" / "reference"
FIG = RAIZ / "reports" / "figures"

# Nombre oficial -> texto de búsqueda (con departamento y país para no confundir:
# existe también el departamento de Caldas, por ejemplo).
MUNICIPIOS = {
    "Medellín": "Medellín, Antioquia, Colombia",
    "Bello": "Bello, Antioquia, Colombia",
    "Itagüí": "Itagüí, Antioquia, Colombia",
    "Envigado": "Envigado, Antioquia, Colombia",
    "Sabaneta": "Sabaneta, Antioquia, Colombia",
    "La Estrella": "La Estrella, Antioquia, Colombia",
    "Caldas": "Caldas, Antioquia, Colombia",
    "Copacabana": "Copacabana, Antioquia, Colombia",
    "Girardota": "Girardota, Antioquia, Colombia",
    "Barbosa": "Barbosa, Antioquia, Colombia",
}


def main() -> None:
    REF.mkdir(parents=True, exist_ok=True)
    partes = []
    for nombre, consulta in MUNICIPIOS.items():
        g = ox.geocode_to_gdf(consulta)
        tipo = g.geometry.iloc[0].geom_type
        if tipo not in ("Polygon", "MultiPolygon"):
            raise RuntimeError(f"{nombre}: OSM devolvió un {tipo}, no un polígono")
        partes.append(gpd.GeoDataFrame({"municipio": [nombre], "osm_nombre": [g["display_name"].iloc[0]]},
                                       geometry=[g.geometry.iloc[0]], crs="EPSG:4326"))
        print(f"OK  {nombre:<12} <- {g['display_name'].iloc[0][:70]}")

    mun = gpd.GeoDataFrame(pd.concat(partes, ignore_index=True), crs="EPSG:4326")

    # Áreas en km² (hay que pasar a metros: EPSG:9377)
    mun["area_km2"] = (mun.to_crs(9377).area / 1e6).round(1)
    print("\nÁrea de cada municipio (km²) — comparar con valores conocidos:")
    print("  (aprox.: Medellín ~380, Bello ~142, Barbosa ~206, Caldas ~133-150, Envigado ~79,")
    print("   Girardota ~82, Copacabana ~70, La Estrella ~35, Itagüí ~17-21, Sabaneta ~15)")
    print(mun[["municipio", "area_km2"]].to_string(index=False))

    # Los municipios no deberían superponerse (solo compartir bordes)
    m = mun.to_crs(9377)
    for i in range(len(m)):
        for j in range(i + 1, len(m)):
            inter = m.geometry.iloc[i].intersection(m.geometry.iloc[j]).area / 1e6
            if inter > 0.05:
                print(f"  AVISO: {m['municipio'].iloc[i]} y {m['municipio'].iloc[j]} se superponen {inter:.2f} km²")

    mun.to_file(REF / "municipios_valle.geojson", driver="GeoJSON")
    print(f"\nGuardado: {REF / 'municipios_valle.geojson'}")

    fig, ax = plt.subplots(figsize=(7, 10))
    mun.plot(ax=ax, column="municipio", alpha=0.35, edgecolor="black", linewidth=0.8)
    for _, f in mun.iterrows():
        p = f.geometry.representative_point()
        ax.annotate(f["municipio"], (p.x, p.y), ha="center", fontsize=8)
    ax.set_title("Municipios del Valle de Aburrá (OpenStreetMap)")
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "mapa_municipios_valle.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Mapa: {FIG / 'mapa_municipios_valle.png'}")


if __name__ == "__main__":
    main()
