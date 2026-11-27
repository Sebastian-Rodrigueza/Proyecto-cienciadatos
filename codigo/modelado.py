"""Construcción y evaluación de los modelos de caudal diario (ítem 10).

Aplica el protocolo de `notebooks/validacion_particion.ipynb` (ítem 9) a tres algoritmos (Ridge, Random Forest y
gradient boosting):
  1. hiperparámetros: cada configuración se entrena con el periodo de entrenamiento (hasta N_BUSQUEDA días por cuenca)
     y se califica en validación con la mediana del NSE por cuenca
  2. algoritmo: validación cruzada espacial sobre el periodo de validación, con la mejor configuración de cada uno;
     gana el mayor NSE mediano espacial y, si otro queda a menos de MARGEN, el más simple
  3. prueba, una sola vez: se reentrena con entrenamiento + validación y se predice el periodo de prueba en las
     cuencas vistas (P4) y en las no vistas (P6, con varias semillas de folds)

Uso (desde la raíz del repositorio, tarda alrededor de una hora): python codigo/modelado.py
Cada paso guarda su resultado en salidas/modelos/ y no se repite si ya existe; la búsqueda se guarda configuración
por configuración. Los resultados los presenta notebooks/modelado.ipynb.
"""
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import ParameterGrid
from sklearn.pipeline import Pipeline

import preparar_datos as pdat
import validacion as val

BASE = Path(__file__).resolve().parent.parent
SALIDA = BASE / "salidas/modelos"
FOLDS = BASE / "salidas/particion/folds_espaciales.csv"
N_BUSQUEDA = 2000                # días de entrenamiento por cuenca en la búsqueda y en la selección
MARGEN = 0.02                    # diferencia de NSE mediano por debajo de la cual se prefiere el algoritmo más simple
N_SEMILLAS_P6 = 3                # semillas de folds para el KPI espacial del modelo escogido
SEMILLA = 0
ARBOLES_FINAL = 300              # el número de árboles no se ajusta: más árboles solo bajan la varianza

# del más simple al más complejo (tamaño del modelo y costo de predecir)
ALGORITMOS = {
    "Ridge": (Ridge(), {"alpha": [0.1, 1, 10, 100]}),
    "Gradient boosting": (
        HistGradientBoostingRegressor(max_iter=500, early_stopping=False, random_state=SEMILLA),
        {"learning_rate": [0.05, 0.1], "max_leaf_nodes": [31, 127], "min_samples_leaf": [20, 200], "l2_regularization": [0, 1]}),
    "Random Forest": (
        RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=SEMILLA),
        {"max_depth": [20, None], "min_samples_leaf": [5, 50], "max_features": [0.33, 0.66]}),
}


def tuberia(modelo, col_num, col_cat):
    return Pipeline([("pre", pdat.crear_preprocesamiento(col_num, col_cat)), ("modelo", modelo)])


def configurar(nombre, params, final=False, n_busqueda=None):
    """Modelo del algoritmo `nombre` con `params`. En el ajuste final el Random Forest usa más árboles y cada árbol
    ve `n_busqueda` filas, las mismas que en la búsqueda, para que los hiperparámetros sigan valiendo."""
    m = clone(ALGORITMOS[nombre][0]).set_params(**params)
    if final and nombre == "Random Forest":
        m.set_params(n_estimators=ARBOLES_FINAL, max_samples=n_busqueda)
    return m


# ---------------------------------------------------------------- datos
def muestra_por_cuenca(filas, periodos, n, semilla=SEMILLA):
    """Máscara con hasta `n` filas al azar por cuenca dentro de `periodos`."""
    en = filas.particion.isin(periodos).to_numpy()
    r = pd.Series(np.random.default_rng(semilla).random(len(filas)), index=filas.index)
    rango = r[en].groupby(filas.gauge_id[en]).rank(method="first")
    m = pd.Series(False, index=filas.index)
    m[rango.index[rango <= n]] = True
    return m


def preparar():
    """Filas de las 248 cuencas con atributos, la marca de la muestra de búsqueda y las columnas de entrada."""
    est = pdat.atributos_estaticos()
    filas, _ = pdat.construir_filas(est, pdat.areas_cuencas(est.index))
    X = pdat.agregar_estaticos(filas, est)
    X["busqueda"] = muestra_por_cuenca(X, ["entrenamiento"], N_BUSQUEDA)
    col_num, col_cat = pdat.columnas(est)
    return est, X, col_num, col_cat


def folds_p6(est):
    """Folds del ítem 9 y, para repetir la evaluación espacial, los de las semillas siguientes mejor equilibradas."""
    f = pd.read_csv(FOLDS, dtype={"gauge_id": str}).set_index("gauge_id")
    region = est.region.loc[f.index]
    sem = val.escoger_semilla(f.grupo, region, est.loc[f.index, ["aridez", "log_area", "mean_ele", "forest_perc"]])
    semillas = [int(s) for s in sem.index[:N_SEMILLAS_P6]]
    folds = {s: val.folds_espaciales(f.grupo, region, semilla=s) for s in semillas}
    assert (folds[semillas[0]] == f.fold).all(), "la primera semilla debe reproducir los folds del ítem 9"
    return folds, f.fold


# ---------------------------------------------------------------- paso 1: hiperparámetros
def paso1(X, col_num, col_cat):
    ruta = SALIDA / "busqueda.csv"
    hecho = pd.read_csv(ruta) if ruta.exists() else pd.DataFrame(columns=["algoritmo", "parametros"])
    tr, va = X[X.busqueda], X[X.particion == "validacion"]
    cols = col_num + col_cat
    for nombre, (_, grilla) in ALGORITMOS.items():
        for params in ParameterGrid(grilla):
            clave = json.dumps(params, sort_keys=True)
            if ((hecho.algoritmo == nombre) & (hecho.parametros == clave)).any():
                continue
            t = time.time()
            m = tuberia(configurar(nombre, params), col_num, col_cat).fit(tr[cols], tr.q)
            met = pdat.metricas_por_cuenca(va.gauge_id, va.q, m.predict(va[cols]))
            fila = dict(algoritmo=nombre, parametros=clave, NSE_mediano=met.NSE.median(), KGE_mediano=met.KGE.median(),
                        RMSE_mediano=met.RMSE.median(), cuencas_NSE_positivo=int((met.NSE > 0).sum()), segundos=round(time.time() - t))
            hecho = pd.concat([hecho, pd.DataFrame([fila])], ignore_index=True)
            hecho.to_csv(ruta, index=False)
            print(f"  [1] {nombre:18s} {clave:90s} NSE {fila['NSE_mediano']:.3f}  ({fila['segundos']} s)", flush=True)
    return hecho


def mejores(busqueda):
    """Mejor configuración de cada algoritmo (mayor NSE mediano en validación)."""
    b = busqueda.sort_values("NSE_mediano", ascending=False).drop_duplicates("algoritmo")
    return {r.algoritmo: json.loads(r.parametros) for r in b.itertuples()}


# ---------------------------------------------------------------- paso 2: algoritmo
def paso2(X, col_num, col_cat, fold, mej):
    ruta = SALIDA / "seleccion_espacial.csv"
    if ruta.exists():
        return pd.read_csv(ruta, dtype={"gauge_id": str})
    cols, trozos = col_num + col_cat, []
    for nombre, params in mej.items():
        for f in sorted(fold.unique()):
            fuera = fold.index[fold == f]
            tr = X[X.busqueda & ~X.gauge_id.isin(fuera)]
            ev = X[(X.particion == "validacion") & X.gauge_id.isin(fuera)]
            m = tuberia(configurar(nombre, params), col_num, col_cat).fit(tr[cols], tr.q)
            trozos.append(pdat.metricas_por_cuenca(ev.gauge_id, ev.q, m.predict(ev[cols])).assign(algoritmo=nombre, fold=f))
        print(f"  [2] {nombre:18s} NSE mediano espacial {pd.concat(trozos).query('algoritmo == @nombre').NSE.median():.3f}", flush=True)
    res = pd.concat(trozos).reset_index()
    res.to_csv(ruta, index=False)
    return res


def elegir(seleccion):
    """Mayor NSE mediano espacial; entre los que quedan a menos de MARGEN del mejor, el más simple."""
    nse = seleccion.groupby("algoritmo").NSE.median()
    cerca = [a for a in ALGORITMOS if a in nse and nse[a] >= nse.max() - MARGEN]
    return cerca[0], nse


# ---------------------------------------------------------------- paso 3: prueba, una sola vez
def paso3_temporal(X, col_num, col_cat, mej, elegido):
    ruta = SALIDA / "prediccion_prueba_temporal.csv.gz"
    if ruta.exists():
        return
    cols, n_b = col_num + col_cat, int(X.busqueda.sum())
    tr, te = X[X.particion != "prueba"], X[X.particion == "prueba"]
    pred = te[["gauge_id", "Fecha", "q"]].copy()
    for nombre, params in mej.items():
        t = time.time()
        m = tuberia(configurar(nombre, params, final=True, n_busqueda=n_b), col_num, col_cat).fit(tr[cols], tr.q)
        pred[nombre] = m.predict(te[cols]).astype("float32")
        print(f"  [3] temporal {nombre:18s} ({time.time() - t:.0f} s)", flush=True)
        if nombre == elegido:
            joblib.dump({"modelo": m, "columnas": cols, "algoritmo": nombre, "parametros": params,
                         "entrenado_con": "entrenamiento + validación (las 248 cuencas)"}, SALIDA / "modelo_final.joblib", compress=3)
        del m
    pred.to_csv(ruta, index=False)


def paso3_espacial(X, col_num, col_cat, mej, elegido, folds):
    cols, n_b = col_num + col_cat, int(X.busqueda.sum())
    for i, (s, fold) in enumerate(folds.items()):
        ruta = SALIDA / f"prediccion_prueba_espacial_s{s}.csv.gz"
        if ruta.exists():
            continue
        algs = list(mej) if i == 0 else [elegido]
        trozos = []
        for f in sorted(fold.unique()):
            fuera = fold.index[fold == f]
            tr = X[(X.particion != "prueba") & ~X.gauge_id.isin(fuera)]
            te = X[(X.particion == "prueba") & X.gauge_id.isin(fuera)]
            p = te[["gauge_id", "Fecha", "q"]].assign(fold=f)
            for nombre in algs:
                m = tuberia(configurar(nombre, mej[nombre], final=True, n_busqueda=n_b), col_num, col_cat).fit(tr[cols], tr.q)
                p[nombre] = m.predict(te[cols]).astype("float32")
                del m
            trozos.append(p)
            print(f"  [3] espacial semilla {s} fold {f}: {', '.join(algs)}", flush=True)
        pd.concat(trozos).to_csv(ruta, index=False)


# ---------------------------------------------------------------- evaluación (la usa notebooks/modelado.ipynb)
def leer_predicciones(nombre):
    return pd.read_csv(SALIDA / nombre, dtype={"gauge_id": str}, parse_dates=["Fecha"])


def metricas(pred, algoritmos):
    """NSE, KGE, RMSE y MAE por cuenca de cada algoritmo (índice: algoritmo, gauge_id)."""
    return pd.concat({a: pdat.metricas_por_cuenca(pred.gauge_id, pred.q, pred[a]) for a in algoritmos}, names=["algoritmo"])


def umbral_creciente(X, cuantil=0.95):
    """Percentil 95 del caudal observado de cada cuenca en el periodo de entrenamiento."""
    return X[X.particion == "entrenamiento"].groupby("gauge_id").q.quantile(cuantil).rename("umbral")


def crecientes(pred, col, umbral):
    """Recall y precisión por cuenca en los días de creciente (caudal sobre el umbral de la cuenca)."""
    d = pred.join(umbral, on="gauge_id")
    obs, est = d.q > d.umbral, d[col] > d.umbral
    g = pd.DataFrame({"gauge_id": d.gauge_id, "vp": obs & est, "obs": obs, "est": est}).groupby("gauge_id").sum()
    return pd.DataFrame({"recall": g.vp / g.obs.replace(0, np.nan), "precision": g.vp / g.est.replace(0, np.nan),
                         "dias_creciente": g.obs})


def error_volumen_mensual(pred, col, min_dias=25):
    """Mediana por cuenca de |volumen mensual estimado / observado − 1|, en meses con al menos `min_dias` días."""
    d = pred.assign(mes=pred.Fecha.dt.to_period("M"))
    m = d.groupby(["gauge_id", "mes"]).agg(obs=("q", "sum"), est=(col, "sum"), n=("q", "size"))
    m = m[(m.n >= min_dias) & (m.obs > 0)]
    return (m.est / m.obs - 1).abs().groupby("gauge_id").median().rename("error_volumen")


def main():
    SALIDA.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    est, X, col_num, col_cat = preparar()
    print(f"{len(X):,} filas; búsqueda con {int(X.busqueda.sum()):,} filas de entrenamiento", flush=True)
    folds, fold_item9 = folds_p6(est)

    print("Paso 1: hiperparámetros (entrenamiento -> validación)", flush=True)
    mej = mejores(paso1(X, col_num, col_cat))
    print("Paso 2: algoritmo (validación cruzada espacial en validación)", flush=True)
    elegido, nse = elegir(paso2(X, col_num, col_cat, fold_item9, mej))
    with open(SALIDA / "eleccion.json", "w") as fh:
        json.dump({"elegido": elegido, "nse_espacial_validacion": nse.round(4).to_dict(), "margen": MARGEN,
                   "parametros": mej, "semillas_p6": list(folds)}, fh, ensure_ascii=False, indent=2)
    print(f"  Escogido: {elegido}  ({nse.round(3).to_dict()})", flush=True)
    print("Paso 3: prueba (entrenamiento + validación -> prueba)", flush=True)
    paso3_temporal(X, col_num, col_cat, mej, elegido)
    paso3_espacial(X, col_num, col_cat, mej, elegido, folds)
    print(f"Listo en {(time.time() - t0) / 60:.0f} min. Resultados en {SALIDA.relative_to(BASE)}", flush=True)


if __name__ == "__main__":
    main()
