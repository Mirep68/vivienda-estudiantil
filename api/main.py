"""
API de ViviU: expone el recomendador para la página web (y para cualquier otra aplicación).

LIBRERÍAS
    FastAPI   framework para construir APIs: cada función decorada con @app.get/@app.post se vuelve
              una dirección web (endpoint). Valida solo los datos que llegan (con Pydantic) y genera
              la documentación interactiva en /docs.
    Uvicorn   el servidor que recibe las peticiones HTTP y se las pasa a FastAPI.
    Pydantic  describe la forma de los datos (tipos, rangos). Si la web manda un presupuesto
              negativo, la API responde un error claro sin llegar al motor.

ENDPOINTS
    GET  /api/resumen                 cifras del sistema (viviendas, fecha de los datos, precisión del modelo)
    GET  /api/universidades           lista de sedes para el formulario
    POST /api/recomendar              perfil del estudiante -> recomendaciones explicadas + embudo
    GET  /api/similares/{id}          "más como esta" (vector de contenido de la red)
    GET  /api/mercado                 panorama del mercado (precios por municipio, universidad, tipo, segmentos)
    GET  /api/metodologia             métricas reales del modelo y del recomendador (lee reports/)
    GET  /                            la página web (carpeta web/)

Los datos se leen UNA vez de PostgreSQL al arrancar (31 mil filas caben de sobra en memoria);
cada búsqueda filtra y puntúa en milisegundos. Al renovar los datos, se reinicia la API.

Ejecutar (desde la raíz del proyecto):
    uvicorn api.main:app --reload
y abrir http://127.0.0.1:8000  (documentación: http://127.0.0.1:8000/docs)
"""

import json
import os
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.models.comun import RAIZ, REPORTES
from src.recomendador.motor import CRITERIOS, Perfil, cargar_datos, cargar_vectores, columna_universidad, recomendar, similares

ESTADO: dict = {}
COLUMNAS_SALIDA = ["id", "url", "img", "municipio", "barrio", "tipo_inmueble", "precio", "precio_justo",
                   "diferencia_pct", "clase_precio", "segmento_nombre", "area_m2", "habitaciones", "banos",
                   "estrato", "dist_univ_km", "indice_seguridad", "dist_metro_km", "metro_cercano",
                   "lat", "lon", "puntaje", "por_que", "similitud", "parqueaderos", "piso", "antiguedad",
                   "amenidades", "dist_parque_km", "servicios_500m", "inmobiliaria", "fecha_publicacion"]


@asynccontextmanager
async def ciclo_de_vida(app: FastAPI):
    """Se ejecuta al arrancar: carga datos, vectores y universidades una sola vez."""
    df = cargar_datos()
    ESTADO["df"] = df
    try:
        ESTADO["vectores"] = cargar_vectores()
    except FileNotFoundError:
        ESTADO["vectores"] = None
    from sqlalchemy import create_engine

    from src.db.conexion import url
    motor = create_engine(url())
    with motor.connect() as c:
        ESTADO["universidades"] = pd.read_sql("SELECT sigla, nombre, sede, municipio, lat, lon FROM universidades "
                                              "ORDER BY nombre, sede", c)
    print(f"ViviU listo: {len(df):,} viviendas, {len(ESTADO['universidades'])} sedes")
    yield
    ESTADO.clear()


app = FastAPI(title="ViviU API", version="1.0",
              description="Recomendador de vivienda estudiantil para el Valle de Aburrá", lifespan=ciclo_de_vida)


def a_registros(d: pd.DataFrame) -> list[dict]:
    """DataFrame -> lista de dicts lista para JSON (NaN -> null; solo columnas existentes)."""
    cols = [c for c in COLUMNAS_SALIDA if c in d.columns]
    out = d[cols].copy()
    for c in ["dist_univ_km", "dist_metro_km", "puntaje", "diferencia_pct", "similitud"]:
        if c in out:
            out[c] = out[c].astype(float).round(3)
    return json.loads(out.to_json(orient="records", force_ascii=False))


# --------------------------------------------------------------------------------------------
class PerfilEntrada(BaseModel):
    """Lo que el estudiante pide. Cada campo trae su descripción para que /docs se entienda solo."""
    universidad: str = Field(..., description="Sigla de la sede (ver /api/universidades), p. ej. USB, UdeA, EAFIT")
    presupuesto_max: float = Field(..., gt=0, description="Arriendo máximo mensual en pesos")
    distancia_max_km: float = Field(3.0, gt=0, le=40, description="Distancia máxima en línea recta a la sede, en km")
    habitaciones_min: int = Field(0, ge=0, le=10, description="Habitaciones mínimas (0 = cualquiera)")
    tipos: list[str] = Field(["Apartaestudio", "Apartamento", "Casa", "Habitación"], description="Tipos de inmueble aceptados")
    municipios: list[str] | None = Field(None, description="Municipios aceptados; null = todo el valle")
    incluir_atipicos: bool = Field(False, description="Incluir anuncios con precio por verificar")
    pesos: dict[str, float] = Field({"cercania": 3, "seguridad": 3, "precio": 3, "valor": 2},
                                    description="Prioridades de 0 a 5: cercania, seguridad, precio, valor, espacio, metro, "
                                                "parques, servicios, amenidades")
    me_gustan: list[str] = Field([], description="Ids de viviendas guardadas para recomendar parecidas")
    n: int = Field(30, ge=1, le=60, description="Cuántos resultados devolver")

    model_config = {"json_schema_extra": {"examples": [{
        "universidad": "USB", "presupuesto_max": 2000000, "distancia_max_km": 3, "habitaciones_min": 1,
        "tipos": ["Apartaestudio", "Apartamento"], "municipios": None, "incluir_atipicos": False,
        "pesos": {"seguridad": 5, "cercania": 4, "valor": 3, "precio": 2}, "me_gustan": [], "n": 5}]}}


class Vivienda(BaseModel):
    """Una recomendación (se muestran los campos principales; la respuesta trae más)."""
    model_config = {"extra": "allow"}
    id: str
    url: str | None = None
    municipio: str | None = None
    barrio: str | None = None
    tipo_inmueble: str | None = None
    precio: float | None = Field(None, description="Arriendo mensual publicado")
    precio_justo: float | None = Field(None, description="Precio estimado por el modelo (red neuronal + XGBoost)")
    diferencia_pct: float | None = Field(None, description="% del precio publicado frente al justo (negativo = más barata)")
    clase_precio: str | None = Field(None, description="ganga, justo, caro, atipico o sin_valoracion")
    dist_univ_km: float | None = None
    indice_seguridad: float | None = Field(None, description="0-100, relativo a las 30 zonas del valle (100 = más segura)")
    puntaje: float | None = Field(None, description="0-100, qué tan bien encaja con las prioridades del estudiante")
    por_que: str | None = Field(None, description="Explicación en palabras")


class RespuestaRecomendar(BaseModel):
    embudo: dict[str, int] = Field(..., description="Cuántas viviendas quedan tras cada criterio obligatorio")
    resultados: list[Vivienda]
    universidad: dict | None = None


@app.get("/api/resumen", summary="Cifras generales del sistema")
def resumen():
    df = ESTADO["df"]
    met = {}
    ruta = REPORTES / "metricas_red.json"
    if ruta.exists():
        m = json.loads(ruta.read_text(encoding="utf-8")).get("ensemble_50_50", {})
        met = {"r2_log": round(m.get("R2_log", 0), 3), "error_tipico_pct": round(m.get("MdAPE_%", 0), 1)}
    return {"viviendas": len(df), "fecha_datos": str(df["fecha_datos"].iloc[0]) if "fecha_datos" in df else None,
            "municipios": sorted(df["municipio"].dropna().unique().tolist()),
            "precio_mediano": float(df["precio"].median()), "modelo": met,
            "criterios": list(CRITERIOS)}


@app.get("/api/universidades", summary="Sedes universitarias disponibles")
def universidades():
    return json.loads(ESTADO["universidades"].to_json(orient="records", force_ascii=False))


@app.post("/api/recomendar", response_model=RespuestaRecomendar,
          summary="Recomienda viviendas para un perfil de estudiante")
def api_recomendar(p: PerfilEntrada):
    try:
        perfil = Perfil(**p.model_dump(exclude={"n"}))
        top, embudo = recomendar(ESTADO["df"], perfil, ESTADO["vectores"], n=p.n)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    unis = ESTADO["universidades"]
    u = unis[unis["sigla"] == p.universidad]
    return {"embudo": embudo, "resultados": a_registros(top),
            "universidad": json.loads(u.to_json(orient="records", force_ascii=False))[0] if len(u) else None}


@app.get("/api/similares/{vivienda_id}", summary="Más como esta: viviendas parecidas según la red")
def api_similares(vivienda_id: str, presupuesto_max: float | None = None, universidad: str | None = None, n: int = 6):
    if ESTADO["vectores"] is None:
        raise HTTPException(status_code=503, detail="No hay vectores de contenido (corre src.recomendador.vectores)")
    df = ESTADO["df"]
    if universidad:
        try:
            df = df.assign(dist_univ_km=df[columna_universidad(universidad)])
        except KeyError:
            raise HTTPException(status_code=400, detail="Universidad desconocida")
    try:
        sim = similares(df, ESTADO["vectores"], vivienda_id, n=n, presupuesto_max=presupuesto_max)
    except (ValueError, IndexError):
        raise HTTPException(status_code=404, detail="Vivienda no encontrada")
    return {"resultados": a_registros(sim)}


def _leer_json(nombre: str) -> dict:
    ruta = REPORTES / nombre
    return json.loads(ruta.read_text(encoding="utf-8")) if ruta.exists() else {}


def _leer_csv(nombre: str, **kw) -> list[dict]:
    ruta = REPORTES / nombre
    if not ruta.exists():
        return []
    return json.loads(pd.read_csv(ruta, encoding="utf-8-sig", **kw).to_json(orient="records", force_ascii=False))


@app.get("/api/mercado", summary="Panorama del mercado de arriendos")
def mercado():
    """Panorama del mercado calculado sobre los anuncios reales (sin habitaciones: casi no hay oferta)."""
    df = ESTADO["df"]
    d = df.assign(precio_m2=df["precio"] / df["area_m2"], ganga=df["clase_precio"].eq("ganga"))
    por_mun = (d.groupby("municipio").agg(viviendas=("id", "size"), precio_mediano=("precio", "median"),
                                          precio_m2=("precio_m2", "median"), pct_gangas=("ganga", "mean"))
               .query("viviendas >= 20").sort_values("precio_mediano"))
    por_mun["pct_gangas"] *= 100
    por_tipo = d.groupby("tipo_inmueble").agg(viviendas=("id", "size"), precio_mediano=("precio", "median"),
                                              area_mediana=("area_m2", "median"))
    cortes = [0, 1e6, 1.5e6, 2e6, 3e6, 5e6, 10e6, float("inf")]
    nombres = ["< $1M", "$1M–1,5M", "$1,5M–2M", "$2M–3M", "$3M–5M", "$5M–10M", "> $10M"]
    rangos = pd.cut(d["precio"], cortes, labels=nombres, right=False).value_counts().reindex(nombres)
    unis = []
    for _, u in ESTADO["universidades"].iterrows():
        col = columna_universidad(u["sigla"])
        if col in d:
            cerca = d[d[col] <= 2]
            unis.append({"sigla": u["sigla"], "nombre": u["nombre"], "sede": u["sede"], "viviendas_2km": len(cerca),
                         "hasta_2M_2km": int((cerca["precio"] <= 2e6).sum()),
                         "precio_mediano_2km": float(cerca["precio"].median()) if len(cerca) else None})
    reparto = (d["clase_precio"].value_counts(normalize=True) * 100).round(1).to_dict()
    to = lambda x: json.loads(x.reset_index().to_json(orient="records", force_ascii=False))
    return {"por_municipio": to(por_mun.round(1)), "por_tipo": to(por_tipo.round(1)),
            "rangos_precio": [{"rango": k, "viviendas": int(v)} for k, v in rangos.items()],
            "universidades": sorted(unis, key=lambda x: -x["viviendas_2km"]), "clases": reparto,
            "segmentos": _leer_csv("segmentos_perfil.csv")}


@app.get("/api/metodologia", summary="Métricas reales del modelo y del recomendador")
def metodologia():
    """Métricas tal como las dejaron los scripts en reports/ (nada escrito a mano)."""
    hist = []
    ruta = REPORTES / "historial_red.csv"
    if ruta.exists():
        h = pd.read_csv(ruta)
        hist = json.loads(h[["epoca", "rmse_train", "rmse_val"]].round(4).to_json(orient="records"))
    val = _leer_json("valoracion_resumen.json")
    return {"modelos": _leer_json("metricas_red.json"), "xgboost": _leer_json("metricas_xgboost.json"),
            "valoracion": {k: val.get(k) for k in ["metricas_oof", "sigma_log", "umbral_ganga_pct", "umbral_caro_pct",
                                                   "umbral_atipico_pct", "reparto_clases_pct", "k_segmentos", "siluetas"]},
            "vectores": _leer_csv("evaluacion_vectores.csv"), "historial_red": hist,
            "viviendas": len(ESTADO["df"])}


# La página web: se sirve al final para que /api/... tenga prioridad
app.mount("/", StaticFiles(directory=RAIZ / "web", html=True), name="web")
