"""
Geocodificación de universidades del Valle de Aburrá.

¿Qué hace este script?
    1. Lee la lista de universidades de config/universidades.csv
    2. Busca la latitud y longitud de cada sede usando OpenStreetMap (Nominatim)
    3. Guarda el resultado en data/reference/universidades.csv
    4. Dibuja un mapa interactivo (reports/figures/mapa_universidades.html)
       para que TÚ verifiques visualmente que cada punto quedó bien ubicado.

¿Por qué lo necesitamos?
    Uno de los parámetros críticos del reto es la "distancia estratégica"
    (vivienda -> universidad). Para calcular distancias necesitamos coordenadas.

Cómo ejecutarlo (desde la raíz del proyecto, con el entorno virtual activo):
    python -m src.utils.geocodificar_universidades
"""

from pathlib import Path

import folium                                   # mapas interactivos en HTML
import pandas as pd                             # tablas de datos (DataFrames)
from geopy.extra.rate_limiter import RateLimiter
from geopy.geocoders import Nominatim           # cliente del geocodificador de OpenStreetMap

from src.limpieza.fincaraiz import sin_tildes   # 'Politécnico' -> 'politecnico' (comparar sin tildes)

# --- Rutas del proyecto ---------------------------------------------------
# Path(__file__) es la ruta de ESTE archivo; subimos 3 niveles hasta la raíz.
RAIZ = Path(__file__).resolve().parents[2]
ENTRADA = RAIZ / "config" / "universidades.csv"
SALIDA = RAIZ / "data" / "reference" / "universidades.csv"
MAPA = RAIZ / "reports" / "figures" / "mapa_universidades.html"

# --- Caja geográfica del Valle de Aburrá ----------------------------------
# (latitud, longitud) de dos esquinas: noroeste y sureste.
# Sirve para que Nominatim NO nos devuelva un "Universidad de Medellín" en otro país,
# y para validar después que todos los puntos caen dentro del valle.
VALLE_NO = (6.45, -75.75)   # norte de Barbosa/Girardota, oeste de San Antonio de Prado
VALLE_SE = (6.00, -75.40)   # sur de Caldas, este de Las Palmas


def dentro_del_valle(lat: float, lon: float) -> bool:
    """True si el punto cae dentro de la caja del Valle de Aburrá."""
    return VALLE_SE[0] <= lat <= VALLE_NO[0] and VALLE_NO[1] <= lon <= VALLE_SE[1]


def main() -> None:
    df = pd.read_csv(ENTRADA)
    print(f"Universidades/sedes a procesar: {len(df)}")

    # Nominatim es un servicio gratuito y su política exige:
    #   - un user_agent que identifique la aplicación
    #   - máximo 1 petición por segundo  -> RateLimiter se encarga de esperar
    geolocator = Nominatim(user_agent="vivienda-estudiantil-vda (proyecto academico)")
    geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1.1, max_retries=2)

    fuentes = []
    for i, fila in df.iterrows():
        # Si ya pusiste coordenadas a mano en el CSV, se respetan.
        if pd.notna(fila["lat"]) and pd.notna(fila["lon"]):
            fuentes.append("manual")
            continue

        # Varias consultas posibles separadas por " | " (se prueban en orden) + nombre y municipio.
        consultas = [q.strip() for q in str(fila["consulta"]).split("|")]
        consultas.append(f'{fila["nombre"]}, {fila["municipio"]}, Antioquia')
        # 'debe_contener': palabras que el resultado DEBE tener (separadas por |), para rechazar
        # coincidencias equivocadas (p. ej. "Politécnico mayor" cuando buscamos "Colegio Mayor").
        claves = [sin_tildes(k) for k in str(fila.get("debe_contener", "") or "").split("|") if k.strip() and k != "nan"]
        ubicacion = None
        for q in consultas:
            candidatos = geocode(q, country_codes="co", viewbox=[VALLE_NO, VALLE_SE], bounded=True,
                                 language="es", exactly_one=False, limit=5) or []
            for c in candidatos:
                texto = sin_tildes(c.address)
                if claves and not any(k in texto for k in claves):
                    print(f"    descartado para {fila['sigla']}: {c.address[:60]}")
                    continue
                ubicacion = c
                break
            if ubicacion:
                break

        if ubicacion and dentro_del_valle(ubicacion.latitude, ubicacion.longitude):
            df.at[i, "lat"] = round(ubicacion.latitude, 6)
            df.at[i, "lon"] = round(ubicacion.longitude, 6)
            fuentes.append("nominatim")
            print(f"  OK   {fila['sigla']:<16} -> {ubicacion.address[:80]}")
        else:
            fuentes.append("no_encontrado")
            print(f"  FALLA {fila['sigla']:<15} -> no se encontró; ponla a mano en config/universidades.csv")

    df["fuente_coord"] = fuentes

    SALIDA.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(SALIDA, index=False, encoding="utf-8")
    print(f"\nGuardado: {SALIDA}")

    # --- Mapa de verificación --------------------------------------------
    ok = df.dropna(subset=["lat", "lon"])
    # Fondo Esri: el de OpenStreetMap bloquea mapas abiertos como archivo local ("Access blocked")
    # y CartoDB ahora exige una clave de API. Esri no pide clave.
    mapa = folium.Map(location=[6.25, -75.58], zoom_start=12, tiles=None)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}",
        attr="Tiles &copy; Esri", name="Esri World Street Map",
    ).add_to(mapa)
    colores = {"Pública": "blue", "Privada": "red"}
    for _, f in ok.iterrows():
        folium.Marker(
            location=[f["lat"], f["lon"]],
            popup=f"<b>{f['nombre']}</b><br>{f['sede']} ({f['municipio']})<br>fuente: {f['fuente_coord']}",
            tooltip=f["sigla"],
            icon=folium.Icon(color=colores.get(f["tipo"], "gray"), icon="graduation-cap", prefix="fa"),
        ).add_to(mapa)
    MAPA.parent.mkdir(parents=True, exist_ok=True)
    mapa.save(MAPA)
    print(f"Mapa: {MAPA}  (ábrelo en el navegador y revisa cada punto)")

    faltan = df[df["fuente_coord"] == "no_encontrado"]
    print(f"\nResumen: {len(ok)} ubicadas, {len(faltan)} pendientes.")
    if len(faltan):
        print("Pendientes:", ", ".join(faltan["sigla"]))


if __name__ == "__main__":
    main()
