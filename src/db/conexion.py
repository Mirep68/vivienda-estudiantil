"""
Conexión a PostgreSQL, sin dependencias pesadas (no importa geopandas): la usan la API y el motor,
también cuando corren en un servidor en la nube.

DOS FORMAS DE CONFIGURARLA
  1. DATABASE_URL (la usan los servicios en la nube como Neon o Render):
        postgresql://usuario:clave@servidor/base?sslmode=require
  2. Variables PG_* del archivo .env (la forma local que usamos desde el principio):
        PG_HOST, PG_PUERTO, PG_USUARIO, PG_CLAVE, PG_BASE
  Si existe DATABASE_URL, tiene prioridad.
"""

import os
import re
import unicodedata
from pathlib import Path

from sqlalchemy.engine import URL, make_url

RAIZ = Path(__file__).resolve().parents[2]


def cargar_entorno() -> None:
    """Lee .env si existe (en la nube no hay .env: las variables las pone el servicio)."""
    try:
        from dotenv import load_dotenv
        load_dotenv(RAIZ / ".env")
    except ImportError:
        pass


def url(base: str | None = None) -> URL:
    """Dirección de conexión para SQLAlchemy con el driver psycopg (versión 3)."""
    cargar_entorno()
    completa = os.getenv("DATABASE_URL")
    if completa:
        u = make_url(completa)
        u = u.set(drivername="postgresql+psycopg")       # postgres:// o postgresql:// -> driver psycopg
        return u.set(database=base) if base else u
    return URL.create(
        "postgresql+psycopg",
        username=os.getenv("PG_USUARIO", "postgres"),
        password=os.getenv("PG_CLAVE"),
        host=os.getenv("PG_HOST", "localhost"),
        port=int(os.getenv("PG_PUERTO", "5432")),
        database=base or os.getenv("PG_BASE", "vivienda_estudiantil"),
    )


def sin_tildes(texto: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", texto) if not unicodedata.combining(c))


def nombre_columna(c: str) -> str:
    """'dist_UdeA-Salud_km' -> 'dist_udea_salud_km' (minúsculas, sin tildes, con '_')."""
    return re.sub(r"[^a-z0-9]+", "_", sin_tildes(str(c)).lower()).strip("_")
