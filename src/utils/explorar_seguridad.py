"""
Exploración rápida del archivo de criminalidad por comuna ya descargado.
Solo LEE data/raw/seguridad/criminalidad_comunas_anual.csv (no descarga ni modifica nada).

Ejecutar:  python -m src.utils.explorar_seguridad
"""

from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parents[2]
ARCHIVO = RAIZ / "data" / "raw" / "seguridad" / "criminalidad_comunas_anual.csv"

# Dejamos que pandas muestre todas las filas de las tablas que imprimimos.
pd.set_option("display.max_rows", 200)

df = pd.read_csv(ARCHIVO, encoding="utf-8-sig", dtype={"Codigo_comuna": str})
df["Codigo_comuna"] = df["Codigo_comuna"].str.strip()

# 1) ¿Qué códigos de comuna existen y cuántos casos tiene cada uno?
print("=" * 70 + "\n1) Códigos de comuna distintos:", df["Codigo_comuna"].nunique())
por_codigo = (
    df.groupby("Codigo_comuna")
    .agg(filas=("Cantidad_casos", "size"), casos=("Cantidad_casos", "sum"))
    .reset_index()
)
# orden: primero los numéricos (por valor), luego los textos
por_codigo["orden"] = pd.to_numeric(por_codigo["Codigo_comuna"], errors="coerce")
print(por_codigo.sort_values(["orden", "Codigo_comuna"]).drop(columns="orden").to_string(index=False))

# 2) ¿Qué porcentaje de casos no tiene comuna, por tipo de delito?
print("\n" + "=" * 70 + "\n2) % de casos SIN DATO por delito")
sin = df["Codigo_comuna"].str.upper() == "SIN DATO"
tot = df.groupby("Conducta")["Cantidad_casos"].sum()
sin_tot = df[sin].groupby("Conducta")["Cantidad_casos"].sum().reindex(tot.index).fillna(0)
print((sin_tot / tot * 100).round(1).rename("% sin comuna").to_string())

# 3) Casos por delito y año (2019-2023)
print("\n" + "=" * 70 + "\n3) Casos por delito y año")
rec = df[df["Fecha_hecho"].between(2019, 2023)]
print(rec.pivot_table(index="Conducta", columns="Fecha_hecho", values="Cantidad_casos", aggfunc="sum").to_string())
