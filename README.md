# Estimación del caudal medio diario en cuencas de Colombia

Proyecto del curso MINE-4101 Ciencia de Datos Aplicada (Universidad de los Andes, 2026-20).

## Grupo 6 - Integrantes

- Juan Sebastián Rodríguez Avella
- Andrés Felipe Méndez Antolínez
- Rafael Yecid Porras Morales
- Juan David Guzmán Sarmiento

## Alcance

Producto de datos (Caudal-COL) que estima el caudal medio diario de cuencas colombianas, incluidas cuencas sin
estación de aforo, para apoyar la operación de embalses, el despacho energético y la gestión del riesgo por
crecientes. La organización de referencia es XM S.A. E.S.P. La primera entrega cubre el entendimiento del negocio,
la ideación del producto, los aspectos éticos y legales, el enfoque analítico, la recolección de datos y el
análisis exploratorio y de calidad. El reporte está en el documento PDF: `Reporte_entrega_1.pdf`).

## Principales hallazgos (primera entrega)

- Los datos son suficientes para un modelo regional: 346 cuencas con atributos completos y 252 con al menos un
  año de registro en el periodo de prueba (2016–2022).
- Las series no tienen celdas vacías, pero los días sin dato no aparecen como filas: 18.281 vacíos de fechas;
  148 cuencas tienen al menos un vacío de un año o más.
- Magdalena-Cauca tiene menor caudal específico que el resto del país (3,70 frente a 6,15 mm/día; t de Welch,
  p < 0,001), resultado que se mantiene en escala logarítmica y sin valores atípicos.
- La suficiencia del registro depende de la región (chi-cuadrado, p = 0,005); el Orinoco es la región más débil.
- La aridez (ρ = −0,74) y el porcentaje de bosque (ρ = 0,54) son los atributos más asociados con el caudal.

## Organización del repositorio

| Ruta | Contenido |
|---|---|
| `Camels col/` | Datos de CAMELS-COL v3 (no se versionan; ver "Datos") |
| `explorar_camels_col.ipynb` | Exploración inicial: inventario, tablas, series y gráficos |
| `eda_fase1.py` | Análisis de calidad, pruebas P1–P3 y Figura 2 del reporte |
| `mockup_producto.py` | Mockup (Figura 1) de la pantalla principal |
| `diccionario_datos.md` | Diccionario de datos: definición, unidad e interpretación de cada variable |
| `salidas/` | Resultados generados: tablas CSV, `eda_resultados.json` y figuras |
| `referencias/` | Artículo de datos de CAMELS-COL (preprint ESSD) |
| `environment.yml` | Definición del entorno de software |
| `USO_IA.md` | Declaración del uso de inteligencia artificial |
| `Video Ciencia de Datos` | Video introductorio del producto de datos |

## Datos

Descargar CAMELS-COL versión 3 desde Zenodo (DOI [10.5281/zenodo.18794895](https://doi.org/10.5281/zenodo.18794895))
y descomprimir los archivos en la carpeta `Camels col/`, conservando sus nombres originales (`02_...csv`,
`04_CAMELS_COL_Hydrometeorological_data/`, etc.). Los datos tienen licencia CC BY 4.0; la temperatura y la
evapotranspiración provienen de MSWX, con licencia CC BY-NC 4.0 (uso no comercial).

## Entorno

Requiere Anaconda o Miniconda. Desde la raíz del repositorio:

```bash
conda env create -f environment.yml     # crea el entorno "proyecto" (solo la primera vez)
conda activate proyecto
python -m ipykernel install --user --name proyecto --display-name "Proyecto (CAMELS-COL)"   # kernel para Jupyter
```

Si se modifica `environment.yml`: `conda env update -f environment.yml --prune`.

## Ejecución (en este orden)

```bash
conda activate proyecto
jupyter nbconvert --to notebook --execute explorar_camels_col.ipynb --inplace   # 1. exploración inicial
python eda_fase1.py                                                             # 2. calidad, P1–P3 y Figura 2
python mockup_producto.py                                                       # 3. mockup (Figura 1)
```

El paso 1 genera `salidas/atributos_unidos.csv` y `salidas/resumen_series.csv`. El paso 2 imprime todos los
resultados citados en la sección 6 del reporte y los guarda en `salidas/eda_resultados.json`.

## Dependencias

Python 3.12 con pandas, numpy, scipy, statsmodels, scikit-learn, matplotlib, seaborn, plotly, geopandas,
rasterio, pyproj, openpyxl, pypdf, python-docx y JupyterLab, todas definidas en `environment.yml` (canal
`defaults` de Anaconda).
