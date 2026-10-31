"""
Paso 6: cargar todos los datos del proyecto a PostgreSQL (con PostGIS si está instalado).

    data/processed/viviendas_<fecha>.csv     -> tabla viviendas          (1 fila por anuncio)
    data/reference/universidades.csv         -> tabla universidades      (25 sedes)
    reports/seguridad_unidades.csv           -> tabla seguridad_unidades (9 municipios + 21 comunas)
    data/reference/municipios_valle.geojson  -> tabla municipios         (polígonos)
    data/reference/comunas.geojson           -> tabla comunas            (polígonos)
    data/reference/barrios.geojson           -> tabla barrios            (polígonos)
    data/reference/poblacion_comunas.csv     -> tabla poblacion_comunas
    data/processed/valoracion_<fecha>.csv    -> tabla valoracion         (precio justo, clase, segmento)
    (registro de cada ejecución)             -> tabla cargas

¿POR QUÉ UNA BASE DE DATOS SI YA TENEMOS CSV?
    - El documento lo pide ("motor de base de datos") y es lo que hace una empresa real.
    - La API y la página web consultan aquí (filtros por precio, municipio, distancia...).
    - Varias tablas relacionadas: cada vivienda apunta a su unidad de seguridad, su valoración...
    - Con PostGIS, la base de datos entiende geometría: "viviendas a menos de 1 km de la USB"
      es una sola consulta SQL que usa un índice espacial.

LOCAL O EN LA NUBE
    Local: variables PG_* del .env (la base se crea si no existe).
    Nube (Neon, etc.): DATABASE_URL en el .env; la base ya la creó el servicio, solo se llenan las tablas.
    La conexión está en src/db/conexion.py.

RENOVABLE
    Cada ejecución reemplaza las tablas con los CSV más recientes y deja una fila en 'cargas'.

Ejecutar:  python -u -m src.db.cargar_postgres
"""

import os
from datetime import datetime

import geopandas as gpd
import pandas as pd
from sqlalchemy import create_engine, text

from src.db.conexion import RAIZ, cargar_entorno, nombre_columna, url

PROCESSED = RAIZ / "data" / "processed"
REF = RAIZ / "data" / "reference"
REPORTS = RAIZ / "reports"

TABLAS = ["valoracion", "viviendas", "universidades", "seguridad_unidades", "municipios", "comunas", "barrios",
          "poblacion_comunas"]


# --------------------------------------------------------------------------------------------
# Conexión
# --------------------------------------------------------------------------------------------
def crear_base_si_no_existe(nombre: str) -> None:
    """Solo en local: se conecta a la base 'postgres' (siempre existe) y crea la nuestra si falta.
    CREATE DATABASE no puede ir dentro de una transacción, por eso AUTOCOMMIT."""
    motor = create_engine(url("postgres"), isolation_level="AUTOCOMMIT")
    with motor.connect() as c:
        existe = c.execute(text("SELECT 1 FROM pg_database WHERE datname = :n"), {"n": nombre}).scalar()
        if not existe:
            c.execute(text(f'CREATE DATABASE "{nombre}" ENCODING \'UTF8\''))
            print(f"Base de datos '{nombre}' creada.")
        else:
            print(f"Base de datos '{nombre}' ya existe.")
    motor.dispose()


def activar_postgis(motor) -> bool:
    """Intenta activar PostGIS. Si no está instalado, seguimos sin geometría (lat/lon y WKT en texto)."""
    try:
        with motor.begin() as c:
            c.execute(text("CREATE EXTENSION IF NOT EXISTS postgis"))
            version = c.execute(text("SELECT postgis_version()")).scalar()
        print(f"PostGIS activo: {version}")
        return True
    except Exception as e:  # noqa: BLE001
        print("AVISO: PostGIS no está instalado; se carga sin columnas de geometría.\n"
              f"       Instálalo con Stack Builder (Spatial Extensions -> PostGIS). Detalle: {str(e).splitlines()[0]}")
        return False


# --------------------------------------------------------------------------------------------
# Utilidades
# --------------------------------------------------------------------------------------------
def subir(motor, df: pd.DataFrame, tabla: str, origen: str, clave: str | None = None) -> None:
    df = df.copy()
    df.columns = [nombre_columna(c) for c in df.columns]
    repetidas = df.columns[df.columns.duplicated()].tolist()
    if repetidas:
        raise SystemExit(f"{tabla}: columnas repetidas tras normalizar nombres: {repetidas}")
    # method="multi" manda muchas filas en un solo INSERT; cada celda es un "parámetro" y
    # PostgreSQL acepta máximo 65.535 por consulta -> filas por lote = 30.000 / columnas.
    lote = max(1, 30_000 // df.shape[1])
    df.to_sql(tabla, motor, if_exists="replace", index=False, chunksize=lote, method="multi")
    with motor.begin() as c:
        if clave:
            c.execute(text(f"ALTER TABLE {tabla} ADD PRIMARY KEY ({clave})"))
        c.execute(text("INSERT INTO cargas (fecha, tabla, filas, origen) VALUES (:f, :t, :n, :o)"),
                  {"f": datetime.now(), "t": tabla, "n": len(df), "o": origen})
    print(f"  {tabla:<20} {len(df):>7,} filas  {df.shape[1]:>3} columnas  <- {origen}")


def geometria_puntos(motor, tabla: str) -> None:
    """Columna geom (punto WGS84) a partir de lat/lon + índice espacial GIST."""
    with motor.begin() as c:
        c.execute(text(f"ALTER TABLE {tabla} ADD COLUMN geom geometry(Point, 4326)"))
        c.execute(text(f"UPDATE {tabla} SET geom = ST_SetSRID(ST_MakePoint(lon, lat), 4326) "
                       "WHERE lat IS NOT NULL AND lon IS NOT NULL"))
        c.execute(text(f"CREATE INDEX {tabla}_geom_idx ON {tabla} USING GIST (geom)"))


def subir_capa(motor, archivo, tabla: str, postgis: bool, clave: str | None = None) -> None:
    """Capa de polígonos: la geometría viaja como texto WKT y PostGIS la convierte."""
    gdf = gpd.read_file(archivo).to_crs(4326)
    df = pd.DataFrame(gdf.drop(columns="geometry"))
    df["wkt"] = gdf.geometry.to_wkt()
    subir(motor, df, tabla, archivo.name, clave)
    if postgis:
        with motor.begin() as c:
            c.execute(text(f"ALTER TABLE {tabla} ADD COLUMN geom geometry(MultiPolygon, 4326)"))
            c.execute(text(f"UPDATE {tabla} SET geom = ST_Multi(ST_GeomFromText(wkt, 4326))"))
            c.execute(text(f"ALTER TABLE {tabla} DROP COLUMN wkt"))
            c.execute(text(f"CREATE INDEX {tabla}_geom_idx ON {tabla} USING GIST (geom)"))


# --------------------------------------------------------------------------------------------
def main() -> None:
    cargar_entorno()
    en_la_nube = bool(os.getenv("DATABASE_URL"))
    if not en_la_nube:
        if not os.getenv("PG_CLAVE"):
            raise SystemExit("Falta PG_CLAVE (o DATABASE_URL) en el archivo .env.")
        crear_base_si_no_existe(os.getenv("PG_BASE", "vivienda_estudiantil"))
    motor = create_engine(url())
    print(f"Conectado a: {motor.url.host} / {motor.url.database}" + (" (nube)" if en_la_nube else " (local)"))
    postgis = activar_postgis(motor)

    with motor.begin() as c:
        # CASCADE: también borra vistas/llaves que dependan de las tablas (carga limpia y repetible)
        for t in TABLAS:
            c.execute(text(f"DROP TABLE IF EXISTS {t} CASCADE"))
        c.execute(text("""CREATE TABLE IF NOT EXISTS cargas (
                            id serial PRIMARY KEY, fecha timestamp, tabla text, filas integer, origen text)"""))

    print("\nCargando tablas:")
    # 1) Viviendas (la tabla principal)
    archivo = sorted(PROCESSED.glob("viviendas_*.csv"))[-1]
    viv = pd.read_csv(archivo, dtype={"id": str, "comuna_codigo": str}, encoding="utf-8-sig", low_memory=False)
    viv["fecha_datos"] = archivo.stem.replace("viviendas_", "")
    subir(motor, viv, "viviendas", archivo.name, clave="id")

    # 2) Universidades
    uni = pd.read_csv(REF / "universidades.csv", encoding="utf-8-sig")
    subir(motor, uni, "universidades", "universidades.csv", clave="id")

    # 3) Unidades de seguridad (el índice de la fila es 'municipio:Bello', 'comuna:10'...)
    seg = pd.read_csv(REPORTS / "seguridad_unidades.csv", index_col=0, encoding="utf-8-sig")
    seg = seg.rename_axis("unidad").reset_index()
    subir(motor, seg, "seguridad_unidades", "seguridad_unidades.csv", clave="unidad")

    # 4) Capas geográficas y población
    subir_capa(motor, REF / "municipios_valle.geojson", "municipios", postgis)
    subir_capa(motor, REF / "comunas.geojson", "comunas", postgis)
    subir_capa(motor, REF / "barrios.geojson", "barrios", postgis)
    subir(motor, pd.read_csv(REF / "poblacion_comunas.csv", encoding="utf-8-sig"),
          "poblacion_comunas", "poblacion_comunas.csv")

    # 5) Valoración (precio justo, ganga/justo/caro, segmento), si ya se generó con src.models.valoracion
    archivos_val = sorted(PROCESSED.glob("valoracion_*.csv"))
    if archivos_val:
        val = pd.read_csv(archivos_val[-1], dtype={"id": str}, encoding="utf-8-sig")
        subir(motor, val, "valoracion", archivos_val[-1].name, clave="id")
        with motor.begin() as c:
            c.execute(text("""ALTER TABLE valoracion ADD CONSTRAINT valoracion_vivienda_fk
                              FOREIGN KEY (id) REFERENCES viviendas (id)"""))
            c.execute(text("CREATE INDEX valoracion_clase_idx ON valoracion (clase_precio)"))
    else:
        print("  (valoracion: aún no existe el archivo; corre primero python -m src.models.valoracion)")

    # --- Geometría e índices para las búsquedas de la web -----------------------------------
    if postgis:
        geometria_puntos(motor, "viviendas")
        geometria_puntos(motor, "universidades")
    with motor.begin() as c:
        for col in ["municipio", "tipo_inmueble", "precio", "estrato", "habitaciones"]:
            c.execute(text(f"CREATE INDEX viviendas_{col}_idx ON viviendas ({col})"))
        # Llave foránea: cada vivienda apunta a una unidad de seguridad que debe existir.
        # 'municipio:Medellín' (anuncios sin comuna) no está entre las 30 unidades -> se agrega,
        # con sus tasas e índices copiados de las viviendas (son iguales en toda la unidad).
        faltan = c.execute(text("""SELECT DISTINCT unidad_seguridad FROM viviendas
                                   WHERE unidad_seguridad IS NOT NULL
                                     AND unidad_seguridad NOT IN (SELECT unidad FROM seguridad_unidades)""")).scalars().all()
        for u in faltan:
            c.execute(text("""
                INSERT INTO seguridad_unidades (unidad, nombre, unidad_tipo,
                       hurto_personas, hurto_residencias, hurto_motos, hurto_carros, homicidios,
                       indice_seguridad, indice_violencia, indice_hurto)
                SELECT :u, :n, 'municipio',
                       max(seg_hurto_personas), max(seg_hurto_residencias), max(seg_hurto_motos),
                       max(seg_hurto_carros), max(seg_homicidios),
                       max(indice_seguridad), max(indice_violencia), max(indice_hurto)
                FROM viviendas WHERE unidad_seguridad = :u"""), {"u": u, "n": u.split(":", 1)[1]})
        c.execute(text("""ALTER TABLE viviendas ADD CONSTRAINT viviendas_seguridad_fk
                          FOREIGN KEY (unidad_seguridad) REFERENCES seguridad_unidades (unidad)"""))
    if faltan:
        print(f"\nUnidades agregadas a seguridad_unidades para la llave foránea: {faltan}")

    verificar(motor, postgis, len(viv))


# --------------------------------------------------------------------------------------------
def verificar(motor, postgis: bool, filas_csv: int) -> None:
    """Consultas de verificación: lo mismo que puedes correr en pgAdmin (sql/verificacion.sql)."""
    with motor.connect() as c:
        print("\nVERIFICACIÓN")
        n = c.execute(text("SELECT count(*) FROM viviendas")).scalar()
        print(f"  Viviendas en la base: {n:,}  | en el CSV: {filas_csv:,}  -> {'OK' if n == filas_csv else 'NO COINCIDE'}")

        print("\n  Precio mediano por municipio (calculado por PostgreSQL):")
        q = text("""SELECT municipio, count(*) AS n,
                           percentile_cont(0.5) WITHIN GROUP (ORDER BY precio) AS mediana
                    FROM viviendas GROUP BY municipio ORDER BY n DESC""")
        for fila in c.execute(q):
            print(f"    {fila.municipio:<14} n={fila.n:>6,}  mediana ${fila.mediana:,.0f}")

        if postgis:
            print("\n  Prueba espacial: viviendas a menos de 1 km de cada universidad (top 5):")
            q = text("""SELECT u.sigla, count(v.id) AS n,
                               percentile_cont(0.5) WITHIN GROUP (ORDER BY v.precio) AS mediana
                        FROM universidades u
                        LEFT JOIN viviendas v ON ST_DWithin(u.geom::geography, v.geom::geography, 1000)
                        GROUP BY u.sigla ORDER BY n DESC LIMIT 5""")
            for fila in c.execute(q):
                med = f"${fila.mediana:,.0f}" if fila.mediana else "-"
                print(f"    {fila.sigla:<15} {fila.n:>6,} viviendas  mediana {med}")
            q = text("""SELECT avg(abs(ST_Distance(v.geom::geography, m.geom::geography) / 1000 - v.dist_universidad_min_km))
                        FROM viviendas v
                        CROSS JOIN LATERAL (SELECT geom FROM universidades u
                                            ORDER BY u.geom <-> v.geom LIMIT 1) m
                        WHERE v.geom IS NOT NULL""")
            dif = c.execute(q).scalar()
            print(f"\n  Distancia a la universidad más cercana, PostGIS vs Python: diferencia media {dif * 1000:.1f} m")

        print("\n  Registro de cargas (últimas):")
        for fila in c.execute(text("SELECT fecha, tabla, filas FROM cargas ORDER BY id DESC LIMIT 8")):
            print(f"    {fila.fecha:%Y-%m-%d %H:%M}  {fila.tabla:<20} {fila.filas:>7,}")


if __name__ == "__main__":
    main()
