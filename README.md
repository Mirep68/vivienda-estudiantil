# Vivienda Estudiantil — Valle de Aburrá

Sistema de recomendación de vivienda para estudiantes universitarios del Valle de Aburrá,
basado en datos reales (scraping de portales inmobiliarios + datos abiertos) y deep learning.

## Estructura

```
config/            Archivos de configuración editables (universidades, parámetros)
data/raw/          Datos tal cual salen del scraper (no se modifican nunca)
data/interim/      Datos en proceso de limpieza
data/processed/    Datos limpios, listos para el modelo
data/reference/    Datos de referencia (universidades, estaciones, barrios)
notebooks/         Exploración y gráficas (Jupyter)
src/scrapers/      Un scraper por portal
src/utils/         Utilidades (geocodificación, verificación del entorno)
src/features/      Construcción de variables (distancias, seguridad, etc.)
src/models/        Modelos de deep learning
models/            Modelos entrenados guardados
api/               Backend (API)
web/               Plataforma web
reports/figures/   Gráficas para el artículo y la presentación
docs/              Artículo, presentación, tabla herramientas/conceptos
```

## Instalación (Windows)

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python -m src.utils.verificar_entorno
```
