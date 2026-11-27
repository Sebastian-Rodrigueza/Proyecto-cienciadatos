"""Estrategia de validación y selección de modelo (ítem 9) para el modelo de caudal diario (CAMELS-COL).

Complementa a `preparar_datos.py` (que ya asigna la partición temporal 60/20/20 dentro de cada cuenca) con:
  - cuencas anidadas a partir de los polígonos de CAMELS-COL (archivo 03) y grupos de cuencas conectadas
  - validación cruzada espacial: StratifiedGroupKFold estratificado por región y agrupado por cuencas conectadas
  - cobertura residual: qué parte de una cuenca de evaluación sigue drenada por cuencas de entrenamiento
  - comparación de cada subconjunto contra el conjunto completo (estadístico KS y PSI)
  - búsqueda de hiperparámetros con la partición de validación y la mediana del NSE por cuenca como criterio
  - fechas compartidas entre cuencas de entrenamiento y de evaluación en la validación espacial
  - intervalos de confianza por bootstrap (por conglomerados de cuencas anidadas) para P1, P2 y P3

Uso: `notebooks/validacion_particion.ipynb`. Salida: salidas/particion/ (fechas de corte, grupos y folds).
"""
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy import stats
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.base import clone
from sklearn.model_selection import ParameterGrid, StratifiedGroupKFold

import preparar_datos as pdat

BASE = Path(__file__).resolve().parent.parent
LIMITES = BASE / "Camels col/03_CAMELS_COL_Basin_boundary/03_CAMELS_COL_Basin_boundary/CAMELS_COL_catchments_boundaries.shp"
SALIDA = BASE / "salidas/particion"
CRS_PLANO = "EPSG:9377"          # MAGNA-SIRGAS Origen Nacional, en metros; las áreas solo se usan como proporciones
UMBRAL_ANIDADA = 0.20            # dos cuencas anidadas van al mismo fold si la pequeña es >= 20 % del área de la grande
N_FOLDS = 5
SEMILLA = 42
N_BOOTSTRAP = 10_000
PERIODOS = ["entrenamiento", "validacion", "prueba"]


# ---------------------------------------------------------------- cuencas anidadas
def poligonos(ids):
    """Polígonos de las cuencas `ids`, en coordenadas planas."""
    b = gpd.read_file(LIMITES)
    b["gauge_id"] = b.IDEAM_CODE.astype(str)
    b = b.set_index("gauge_id").loc[list(ids), ["geometry"]].to_crs(CRS_PLANO)
    b["geometry"] = b.geometry.make_valid()
    b["area_km2"] = b.area / 1e6
    return b


def pares_anidados(b):
    """Todos los pares de cuencas que se tocan: fracción del área de la pequeña que está dentro de la grande.

    `rel_area` = área de la pequeña / área de la grande: qué tanto de la grande drena la pequeña.
    """
    sj = gpd.sjoin(b.reset_index()[["gauge_id", "geometry"]], b.reset_index()[["gauge_id", "geometry"]], predicate="intersects")
    sj = sj[sj.gauge_id_left != sj.gauge_id_right]
    area = b.geometry.area
    filas = []
    for g, p in zip(sj.gauge_id_left, sj.gauge_id_right):
        if area[g] <= area[p]:
            continue
        filas.append(dict(grande=g, pequena=p, frac_dentro=b.geometry[g].intersection(b.geometry[p]).area / area[p],
                          rel_area=area[p] / area[g]))
    return pd.DataFrame(filas)


def grupos_conectados(ids, pares):
    """Número de grupo de cada cuenca: componentes conexas del grafo de `pares` (0 = el grupo más grande)."""
    pos = {g: i for i, g in enumerate(ids)}
    n = len(ids)
    a = coo_matrix((np.ones(len(pares)), ([pos[g] for g in pares.grande], [pos[g] for g in pares.pequena])), shape=(n, n))
    _, etiqueta = connected_components(a, directed=False)
    g = pd.Series(etiqueta, index=list(ids))
    orden = {e: i for i, e in enumerate(g.value_counts().index)}
    return g.map(orden).rename("grupo")


def enlaces(pares, umbral=UMBRAL_ANIDADA):
    """Pares que obligan a ir juntos: la pequeña está dentro de la grande y drena al menos `umbral` de su área."""
    return pares[(pares.frac_dentro > 0.5) & (pares.rel_area >= umbral)]


# ---------------------------------------------------------------- validación cruzada espacial
def folds_espaciales(grupo, region, k=N_FOLDS, semilla=SEMILLA):
    """Fold (0..k-1) de cada cuenca: los grupos no se parten y la proporción de regiones se reparte pareja."""
    sgkf = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=semilla)
    fold = pd.Series(-1, index=grupo.index, name="fold")
    for i, (_, ev) in enumerate(sgkf.split(np.zeros(len(grupo)), region.loc[grupo.index], grupo)):
        fold.iloc[ev] = i
    return fold


def escoger_semilla(grupo, region, atributos, semillas=range(50), k=N_FOLDS):
    """Folds con varias semillas y el desequilibrio de atributos de cada uno, sin mirar ningún modelo.

    KS_max: el mayor estadístico KS entre las cuencas de un fold y todas, sobre todos los atributos y folds.
    """
    filas = []
    for s in semillas:
        fold = folds_espaciales(grupo, region, k, s)
        ks = [stats.ks_2samp(atributos.loc[fold == f, c], atributos[c]).statistic for c in atributos.columns for f in range(k)]
        tam = fold.value_counts()
        filas.append(dict(semilla=s, KS_max=max(ks), KS_medio=float(np.mean(ks)), cuencas_min=tam.min(), cuencas_max=tam.max()))
    return pd.DataFrame(filas).set_index("semilla").sort_values("KS_max")


def cobertura_residual(b, pares, fold):
    """Para cada cuenca, lo que comparte con cuencas de entrenamiento cuando su fold se evalúa.

    aguas_arriba: fracción de su área drenada por cuencas de entrenamiento anidadas dentro de ella (unión de polígonos).
    aguas_abajo:  la mayor fracción que ella representa de una cuenca de entrenamiento que la contiene.
    """
    dentro = pares[pares.frac_dentro > 0.5]
    filas = []
    for g in fold.index:
        arriba = dentro[(dentro.grande == g) & dentro.pequena.map(fold).ne(fold[g])].pequena
        cub = b.geometry.loc[arriba].union_all().intersection(b.geometry[g]).area / b.geometry[g].area if len(arriba) else 0.0
        abajo = dentro[(dentro.pequena == g) & dentro.grande.map(fold).ne(fold[g])].rel_area
        filas.append(dict(gauge_id=g, aguas_arriba=cub, aguas_abajo=abajo.max() if len(abajo) else 0.0))
    return pd.DataFrame(filas).set_index("gauge_id")


def resumen_umbral(ids, region, b, pares, umbrales, k=N_FOLDS, semilla=SEMILLA):
    """Grupos, equilibrio de los folds y cobertura residual para varios umbrales de `rel_area`."""
    filas = []
    for u in umbrales:
        grupo = grupos_conectados(ids, enlaces(pares, u))
        fold = folds_espaciales(grupo, region, k, semilla)
        cob = cobertura_residual(b, pares, fold)
        tam = fold.value_counts()
        filas.append({"umbral rel_area": u, "grupos": grupo.nunique(), "grupo más grande": int(grupo.value_counts().iloc[0]),
                      "cuencas por fold (mín-máx)": f"{tam.min()}-{tam.max()}",
                      "aguas arriba mediana": cob.aguas_arriba.median(), "aguas arriba p90": cob.aguas_arriba.quantile(.9),
                      "cuencas con aguas arriba >= 0,5": int((cob.aguas_arriba >= .5).sum()),
                      "cuencas con aguas abajo >= 0,2": int((cob.aguas_abajo >= .2).sum())})
    return pd.DataFrame(filas).set_index("umbral rel_area")


# ---------------------------------------------------------------- partición temporal
def fechas_corte(filas):
    """Primer día de validación y de prueba, y filas de cada periodo, por cuenca."""
    f = filas.groupby(["gauge_id", "particion"]).Fecha.agg(["min", "max", "size"]).unstack("particion")
    t = pd.DataFrame({"inicio_entrenamiento": f[("min", "entrenamiento")], "inicio_validacion": f[("min", "validacion")],
                      "inicio_prueba": f[("min", "prueba")], "fin_prueba": f[("max", "prueba")]})
    for p in PERIODOS:
        t[f"filas_{p}"] = f[("size", p)].astype(int)
    return t


def cobertura_fecha_fija(filas, inicio, fin="2022-12-31"):
    """Días utilizables de cada cuenca en un periodo de prueba fijo (como el del primer reporte)."""
    m = (filas.Fecha >= inicio) & (filas.Fecha <= fin)
    return filas[m].groupby("gauge_id").size().reindex(filas.gauge_id.unique(), fill_value=0)


# ---------------------------------------------------------------- distribuciones
def psi(ref, x, bins=10):
    """Índice de estabilidad poblacional de `x` frente a `ref`, con los deciles de `ref` como intervalos."""
    cortes = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    cortes[0], cortes[-1] = -np.inf, np.inf
    a = np.clip(np.histogram(ref, cortes)[0] / len(ref), 1e-6, None)
    c = np.clip(np.histogram(x, cortes)[0] / len(x), 1e-6, None)
    return float(((c - a) * np.log(c / a)).sum())


def comparar_con_total(df, columna, variables):
    """Cada subconjunto (`columna`) contra todas las filas: media, cuantiles, estadístico KS y PSI."""
    filas = []
    for v in variables:
        tot = df[v].dropna().to_numpy()
        filas.append(dict(variable=v, subconjunto="total", n=len(tot), media=tot.mean(), p05=np.quantile(tot, .05),
                          mediana=np.median(tot), p95=np.quantile(tot, .95), KS=0.0, PSI=0.0))
        for s, x in df.groupby(columna, observed=True)[v]:
            x = x.dropna().to_numpy()
            filas.append(dict(variable=v, subconjunto=s, n=len(x), media=x.mean(), p05=np.quantile(x, .05), mediana=np.median(x),
                              p95=np.quantile(x, .95), KS=stats.ks_2samp(x, tot, method="asymp").statistic, PSI=psi(tot, x)))
    return pd.DataFrame(filas).set_index(["variable", "subconjunto"])


# ---------------------------------------------------------------- selección de modelo
def resumir(m, region):
    """Mediana nacional y por región de las métricas por cuenca."""
    m = m.join(region)
    tab = m.groupby("region")[["NSE", "KGE", "RMSE", "MAE"]].median()
    tab.loc["Nacional"] = m[["NSE", "KGE", "RMSE", "MAE"]].median()
    tab["cuencas"] = m.groupby("region").size().reindex(tab.index).fillna(len(m)).astype(int)
    return tab


def buscar_hiperparametros(modelo, grilla, X_tr, y_tr, X_val, y_val, g_val):
    """Entrena cada configuración con entrenamiento y la califica en validación con la mediana del NSE por cuenca.

    Equivale a GridSearchCV con PredefinedSplit (-1 entrenamiento, 0 validación), pero el criterio se calcula por
    cuenca: el R² o el error de todas las filas juntas los dominan las cuencas húmedas.
    """
    filas = []
    for params in ParameterGrid(grilla):
        m = clone(modelo).set_params(**params).fit(X_tr, y_tr)
        met = pdat.metricas_por_cuenca(g_val, y_val, m.predict(X_val))
        filas.append({**params, "NSE mediano": met.NSE.median(), "KGE mediano": met.KGE.median(),
                      "cuencas con NSE > 0": int((met.NSE > 0).sum())})
    return pd.DataFrame(filas).sort_values("NSE mediano", ascending=False)


def evaluar_espacial(modelo, filas, fold, columnas_x, periodo_eval="prueba", periodos_entrenamiento=("entrenamiento",)):
    """Validación cruzada espacial: entrena con `periodos_entrenamiento` de las cuencas de los otros folds y predice
    `periodo_eval` de las cuencas que quedaron fuera. Devuelve métricas por cuenca con su fold."""
    trozos = []
    for f in sorted(fold.unique()):
        fuera = fold.index[fold == f]
        tr = filas[~filas.gauge_id.isin(fuera) & filas.particion.isin(periodos_entrenamiento)]
        ev = filas[filas.gauge_id.isin(fuera) & (filas.particion == periodo_eval)]
        m = clone(modelo).fit(tr[columnas_x], tr.q)
        met = pdat.metricas_por_cuenca(ev.gauge_id, ev.q, m.predict(ev[columnas_x]))
        trozos.append(met.assign(fold=f))
    return pd.concat(trozos)


# ---------------------------------------------------------------- fechas compartidas en la validación espacial
def fechas_compartidas(filas, fold, periodo_eval="validacion", periodos_entrenamiento=("entrenamiento",)):
    """Por cuenca evaluada: fracción de sus días de `periodo_eval` cuya fecha también está en el entrenamiento de las
    cuencas de los otros folds, y número mediano de cuencas de entrenamiento con dato en esas fechas."""
    trozos = []
    for f in sorted(fold.unique()):
        fuera = fold.index[fold == f]
        tr = filas[~filas.gauge_id.isin(fuera) & filas.particion.isin(periodos_entrenamiento)]
        por_fecha = tr.groupby("Fecha").gauge_id.nunique()
        ev = filas.loc[filas.gauge_id.isin(fuera) & (filas.particion == periodo_eval), ["gauge_id", "Fecha"]]
        n = ev.Fecha.map(por_fecha).fillna(0)
        trozos.append(pd.DataFrame({"compartida": n > 0, "cuencas_entrenamiento": n, "gauge_id": ev.gauge_id})
                      .groupby("gauge_id").agg(fraccion_compartida=("compartida", "mean"),
                                               cuencas_misma_fecha=("cuencas_entrenamiento", "median")))
    return pd.concat(trozos)


def experimento_corte_comun(modelos, filas, fold, columnas_x, corte, n_por_cuenca=2000, min_dias_eval=365, semilla=0):
    """¿Compartir fechas entre cuencas de entrenamiento y de evaluación infla el NSE espacial?

    Las cuencas de cada fold se evalúan en sus días desde `corte`. Las de los otros folds entrenan con el mismo número
    de días por cuenca tomados de tres lugares: antes de `corte` (ninguna fecha compartida), desde `corte` (las mismas
    fechas que la evaluación) o de cualquier fecha. Solo cambia de dónde salen los días; el conjunto de evaluación,
    las cuencas y el tamaño del entrenamiento son los mismos.
    """
    antes = (filas.Fecha < corte).to_numpy()
    ids = filas.gauge_id.unique()
    n_antes = filas[antes].groupby("gauge_id").size().reindex(ids, fill_value=0)
    n_despues = filas[~antes].groupby("gauge_id").size().reindex(ids, fill_value=0)
    n = np.minimum(n_antes, n_despues).clip(upper=n_por_cuenca)
    r = pd.Series(np.random.default_rng(semilla).random(len(filas)), index=filas.index)
    variantes = {"antes del corte (sin fechas compartidas)": antes, "desde el corte (mismas fechas)": ~antes,
                 "cualquier fecha": np.ones(len(filas), bool)}
    muestras = {}
    for v, en in variantes.items():
        rango = r[en].groupby(filas.gauge_id[en]).rank(method="first")
        muestras[v] = rango.index[rango.to_numpy() <= filas.gauge_id[en].map(n).to_numpy()]
    trozos = []
    for f in sorted(fold.unique()):
        fuera = fold.index[fold == f]
        ev = filas[~antes & filas.gauge_id.isin(fuera)]
        ev = ev[ev.gauge_id.map(ev.gauge_id.value_counts()) >= min_dias_eval]
        for v, idx in muestras.items():
            tr = filas.loc[idx]
            tr = tr[~tr.gauge_id.isin(fuera)]
            for nombre, modelo in modelos.items():
                m = clone(modelo).fit(tr[columnas_x], tr.q)
                met = pdat.metricas_por_cuenca(ev.gauge_id, ev.q, m.predict(ev[columnas_x]))
                trozos.append(met.assign(variante=v, modelo=nombre, fold=f, filas_entrenamiento=len(tr)))
    return pd.concat(trozos)


# ---------------------------------------------------------------- intervalos de confianza por bootstrap
def _remuestreo(grupos, n, semilla):
    """Índices de `n` remuestreos por conglomerados: se sortean grupos de cuencas conectadas con reemplazo."""
    codigos, unicos = pd.factorize(grupos)
    miembros = [np.flatnonzero(codigos == k) for k in range(len(unicos))]
    rng = np.random.default_rng(semilla)
    for _ in range(n):
        yield np.concatenate([miembros[k] for k in rng.integers(0, len(unicos), len(unicos))])


def bootstrap_mediana(x, grupos=None, n=N_BOOTSTRAP, semilla=0):
    """Mediana de `x` (una métrica por cuenca) con su intervalo del 95 % por bootstrap percentil. Si se dan `grupos`,
    las cuencas anidadas se sortean juntas, porque no son independientes."""
    x = x.dropna()
    g = np.arange(len(x)) if grupos is None else grupos.reindex(x.index).to_numpy()
    v = x.to_numpy()
    meds = np.array([np.median(v[i]) for i in _remuestreo(g, n, semilla)])
    return pd.Series({"mediana": np.median(v), "IC95_inf": np.percentile(meds, 2.5), "IC95_sup": np.percentile(meds, 97.5),
                      "cuencas": len(v)})


def bootstrap_pareado(a, b, grupos=None, n=N_BOOTSTRAP, semilla=0):
    """P3: diferencia de NSE por cuenca (`a` − `b`, mismas cuencas). Devuelve la mejora mediana, su intervalo del
    95 % por bootstrap pareado y el % de cuencas que mejoran. Criterio: el intervalo excluye 0 y la mejora es ≥ 0,10."""
    d = (a - b).dropna()
    r = bootstrap_mediana(d, grupos, n, semilla)
    return pd.Series({"mejora mediana": r.mediana, "IC95_inf": r.IC95_inf, "IC95_sup": r.IC95_sup,
                      "% de cuencas que mejoran": 100 * (d > 0).mean(), "cuencas": len(d),
                      "cumple": bool(r.IC95_inf > 0 and r.mediana >= 0.10)})


def resumir_ic(m, region, grupos=None, metrica="NSE", n=N_BOOTSTRAP):
    """Mediana nacional y por región de una métrica con su intervalo del 95 % por bootstrap."""
    x = m[metrica].dropna()
    reg = region.reindex(x.index)
    tab = {r: bootstrap_mediana(x[reg == r], grupos, n) for r in reg.dropna().unique()}
    tab["Nacional"] = bootstrap_mediana(x, grupos, n)
    return pd.DataFrame(tab).T
