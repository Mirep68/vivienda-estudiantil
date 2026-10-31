"""
Paso 1 del modelo: preparar la matriz de criterios (features) y dividir los datos.

    data/processed/viviendas_<fecha>.csv  ->  data/processed/modelo/train.csv, val.csv, test.csv
                                           ->  data/processed/modelo/metadata.json

Etapas del documento que cubre:
  - "Selección y generación de matriz y vectores de criterios"
  - "Normalización escalar, centralización, one hot encoding y embeddings" (la estandarización
    y los embeddings se aplican dentro de cada modelo con los parámetros guardados aquí)

DECISIONES
  1. Objetivo = log(precio): el modelo aprende errores RELATIVOS (en %).
  2. Faltantes: mediana del TRAIN + columna indicadora '<col>_falta' (el "no informó" también informa).
  3. One-hot de las amenidades más frecuentes (ascensor, piscina, vigilancia...).
  4. Categóricas con muchos valores (barrio, comuna, municipio, tipo) -> índices enteros para embeddings.
     Categorías con menos de MIN_FRECUENCIA anuncios en train -> "OTRO".
  5. División 70/15/15 POR GRUPOS de anuncios duplicados (mismo inmueble publicado por varias
     inmobiliarias), para que una copia no quede en train y otra en test (fuga de datos).
  6. Todo lo que se "aprende" de los datos (medianas, categorías, medias) sale SOLO del train.

CORRECCIÓN (v2): en pandas 3, .astype(str) deja los faltantes como NaN, y pd.factorize les da el
código -1. Resultado: los ~1.560 anuncios con área, habitaciones o baños faltantes quedaban TODOS en
un único "grupo de duplicados" y caían juntos en un solo conjunto. Ahora el faltante se escribe "NA"
dentro de la clave, y se verifica que ningún grupo sea anormalmente grande.

Ejecutar:  python -m src.models.preparar_datos
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from src.limpieza.fincaraiz import sin_tildes

RAIZ = Path(__file__).resolve().parents[2]
PROCESSED = RAIZ / "data" / "processed"
SALIDA = PROCESSED / "modelo"
SEMILLA = 42
TOP_AMENIDADES = 30
MIN_FRECUENCIA = 5
MAX_TAM_GRUPO = 50     # un mismo inmueble rara vez se publica más de unas decenas de veces

NUMERICAS = [
    "area_m2", "habitaciones", "banos", "parqueaderos", "piso", "estrato", "antiguedad_cod",
    "n_amenidades", "lat", "lon",
    "seg_hurto_personas", "seg_hurto_residencias", "seg_hurto_motos", "seg_hurto_carros", "seg_homicidios",
    "dist_universidad_min_km", "dist_metro_km", "dist_parque_km",
    "servicios_500m", "supermercados_500m", "farmacias_500m",
]
CATEGORICAS = ["municipio", "tipo_inmueble", "comuna", "barrio_modelo", "estado"]


def _texto(s: pd.Series) -> pd.Series:
    """Convierte a texto escribiendo 'NA' en los faltantes (astype(str) en pandas 3 los deja como NaN)."""
    return s.map(lambda v: "NA" if pd.isna(v) else str(v))


def grupos_duplicados(df: pd.DataFrame) -> pd.Series:
    """Mismo inmueble = misma ubicación (~10 m), área, habitaciones y baños."""
    clave = (_texto(df["lat"].round(4)) + "|" + _texto(df["lon"].round(4)) + "|" +
             _texto(df["area_m2"].round(0)) + "|" + _texto(df["habitaciones"]) + "|" + _texto(df["banos"]))
    codigos = pd.factorize(clave)[0]
    assert (codigos >= 0).all(), "hay claves de duplicado vacías"
    return pd.Series(codigos, index=df.index)


def main() -> None:
    entrada = sorted(PROCESSED.glob("viviendas_*.csv"))[-1]
    df = pd.read_csv(entrada, dtype={"id": str, "comuna_codigo": str})
    print(f"Viviendas: {len(df):,}  ({entrada.name})")

    # --- Filas utilizables ----------------------------------------------------------
    sin_coord = df["lat"].isna()
    print(f"Sin coordenadas (no tienen distancias ni entorno, se excluyen): {int(sin_coord.sum())}")
    df = df[~sin_coord].copy()

    # --- Variables derivadas ------------------------------------------------------------
    df["log_precio"] = np.log(df["precio"])
    df["antiguedad_cod"] = df["antiguedad_cod"].replace(0, np.nan)        # 0 = no informada
    df["comuna"] = np.where(df["municipio"].eq("Medellín"), "C" + df["comuna_codigo"].fillna("NA"), "fuera")
    # Barrio: el oficial (polígonos) en Medellín; el del anuncio en los demás municipios.
    barrio_txt = df["barrio"].fillna("").map(sin_tildes)
    df["barrio_modelo"] = np.where(df["municipio"].eq("Medellín"), df["barrio_oficial"].fillna("NA"),
                                   df["municipio"] + ":" + barrio_txt)
    df["estado"] = df["estado"].fillna("NA")
    print(f"Variable objetivo log(precio): media {df['log_precio'].mean():.3f}, "
          f"desv. {df['log_precio'].std():.3f}")

    # --- División por grupos (sin fuga de datos) ----------------------------------------
    df["grupo"] = grupos_duplicados(df)
    tam = df.groupby("grupo").size()
    print(f"\nPosibles duplicados: {int((tam > 1).sum()):,} grupos con 2+ anuncios "
          f"({int(tam[tam > 1].sum()):,} anuncios, {tam[tam > 1].sum() / len(df) * 100:.1f} %)")
    print(f"Grupo más grande: {int(tam.max())} anuncios (debe ser pequeño; tope de alerta {MAX_TAM_GRUPO})")
    assert tam.max() <= MAX_TAM_GRUPO, "hay un grupo de duplicados anormalmente grande: revisar la clave"

    g1 = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=SEMILLA)
    idx_train, idx_resto = next(g1.split(df, groups=df["grupo"]))
    resto = df.iloc[idx_resto]
    g2 = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=SEMILLA)
    idx_val, idx_test = next(g2.split(resto, groups=resto["grupo"]))
    df["conjunto"] = "train"
    df.loc[resto.index[idx_val], "conjunto"] = "val"
    df.loc[resto.index[idx_test], "conjunto"] = "test"
    tr = df["conjunto"].eq("train")
    print("Tamaños:", df["conjunto"].value_counts().to_dict())
    cruzados = df.groupby("grupo")["conjunto"].nunique().gt(1).sum()
    print(f"Grupos repartidos en más de un conjunto (debe ser 0): {int(cruzados)}")
    falta_area = df["area_m2"].isna()
    print("Anuncios SIN área por conjunto (deben repartirse ~70/15/15):",
          df.loc[falta_area, "conjunto"].value_counts().to_dict())

    # --- Amenidades -> one-hot (vocabulario del TRAIN) -----------------------------------
    listas = df["amenidades"].fillna("").str.split("|")
    frec = listas[tr].explode().loc[lambda s: s != ""].value_counts()
    top = frec.head(TOP_AMENIDADES).index.tolist()
    cols_amen = []
    for i, a in enumerate(top):
        col = f"am_{i:02d}"
        df[col] = listas.map(lambda l, a=a: int(a in l))
        cols_amen.append(col)
    print(f"\nAmenidades one-hot ({len(top)}): {top[:10]} ...")

    # --- Numéricas: faltantes -> mediana del train + indicador ----------------------------
    medianas, cols_falta = {}, []
    for c in NUMERICAS:
        falta = df[c].isna()
        if falta[tr].any():
            df[f"{c}_falta"] = falta.astype(int)
            cols_falta.append(f"{c}_falta")
        medianas[c] = float(df.loc[tr, c].median())
        df[c] = df[c].fillna(medianas[c])
    print("% faltante (train) -> se rellena con la mediana:",
          {c.replace("_falta", ""): round(float(df.loc[tr, c].mean()) * 100, 1) for c in cols_falta})

    # Media y desviación del TRAIN (las usará la red para estandarizar)
    estandar = {c: {"media": float(df.loc[tr, c].mean()), "desv": float(df.loc[tr, c].std() or 1.0)}
                for c in NUMERICAS}

    # --- Categóricas -> índices (0 = "OTRO" / no vista en train) ---------------------------
    vocab = {}
    for c in CATEGORICAS:
        cuenta = df.loc[tr, c].value_counts()
        categorias = sorted(cuenta[cuenta >= MIN_FRECUENCIA].index.tolist())
        vocab[c] = ["OTRO"] + categorias
        mapa = {v: i for i, v in enumerate(vocab[c])}
        df[f"{c}_idx"] = df[c].map(mapa).fillna(0).astype(int)
        otros = (df[f"{c}_idx"] == 0).mean() * 100
        print(f"  {c:<14} {len(vocab[c]):>4} categorías (incl. OTRO) | en OTRO: {otros:.1f} %")

    # --- Guardar -----------------------------------------------------------------------------
    SALIDA.mkdir(parents=True, exist_ok=True)
    cols_idx = [f"{c}_idx" for c in CATEGORICAS]
    columnas = ["id", "precio", "log_precio", "conjunto"] + NUMERICAS + cols_falta + cols_amen + cols_idx + CATEGORICAS
    for nombre in ["train", "val", "test"]:
        df.loc[df["conjunto"] == nombre, columnas].to_csv(SALIDA / f"{nombre}.csv", index=False, encoding="utf-8-sig")

    metadata = {
        "fuente": entrada.name, "semilla": SEMILLA, "objetivo": "log_precio",
        "numericas": NUMERICAS, "indicadores_falta": cols_falta,
        "amenidades": dict(zip(cols_amen, top)), "categoricas": CATEGORICAS,
        "vocabulario": vocab, "medianas": medianas, "estandarizacion": estandar,
        "tamanos": df["conjunto"].value_counts().to_dict(),
    }
    (SALIDA / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=1), encoding="utf-8")
    n_feat = len(NUMERICAS) + len(cols_falta) + len(cols_amen)
    print(f"\nVariables de entrada: {n_feat} numéricas/binarias + {len(CATEGORICAS)} categóricas (embeddings)")
    print(f"Guardado en {SALIDA}: train.csv, val.csv, test.csv, metadata.json")

    # --- Verificación: los 3 conjuntos deben parecerse -------------------------------------------
    print("\nPrecio mediano por conjunto (deben ser parecidos):")
    print(df.groupby("conjunto")["precio"].median().map("${:,.0f}".format).to_string())
    print("\nProporción por municipio en cada conjunto (%):")
    print((pd.crosstab(df["municipio"], df["conjunto"], normalize="columns") * 100).round(1).to_string())


if __name__ == "__main__":
    main()
