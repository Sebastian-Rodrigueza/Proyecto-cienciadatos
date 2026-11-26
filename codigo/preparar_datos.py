"""Preparación de datos para el modelo de caudal diario (CAMELS-COL).

Reúne lo que usan `notebooks/preparacion_datos.ipynb` y `notebooks/modelado_ia.ipynb`:
  - atributos estáticos de las cuencas (sin duplicados ni constantes, aridez con la ETP de Hargreaves, log a las muy asimétricas)
  - variables de la serie diaria (lluvia acumulada, temperatura, ETP, estacionalidad) y el caudal específico (mm/día)
  - partición 60/20/20 dentro de cada cuenca, en orden cronológico
  - preprocesamiento tipo clase (Práctica 6): ColumnTransformer con imputación, estandarización y one-hot
  - métricas por cuenca (NSE, KGE, RMSE, MAE)

Entradas: salidas/series_limpias/ (limpiar_series.py) y salidas/cuencas_modelado.csv (aplicar_banderas.py).
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

BASE = Path(__file__).resolve().parent.parent
CAMELS = BASE / "Camels col"
SERIES = BASE / "salidas/series_limpias"
REGIONES = {"1": "Caribe", "2": "Magdalena-Cauca", "3": "Orinoco", "4": "Amazonas", "5": "Pacífico"}
VENTANAS_P = (3, 7, 30, 90)          # días de lluvia acumulada
P_TRAIN, P_VAL = 0.60, 0.20          # el resto (20 %) es prueba, como en la Práctica 6

DINAMICAS = ["P", *[f"P{w}" for w in VENTANAS_P], "Tmean", "Trango", "ETP", "ETP30", "sin_doy", "cos_doy", "sin2_doy", "cos2_doy"]

# atributos que NO entran: firmas hidrológicas (se calculan del caudal: fuga del objetivo), duplicados y los que usan la ETP original
FUERA = {"volcaniclastic_rock_perc", "perimeter", "tc_chow", "tc_Johnstone", "tc_Engi_Corps", "aridity",
         "q_mean", "gauge_id", "gauge_lat", "gauge_lon"}


def cuencas_modelado():
    """Las cuencas que pasaron las banderas (salidas/cuencas_modelado.csv)."""
    return pd.read_csv(BASE / "salidas/cuencas_modelado.csv", dtype={"gauge_id": str}).set_index("gauge_id")


def _leer(csv, sep=";"):
    d = pd.read_csv(CAMELS / csv, sep=sep)
    d.columns = d.columns.str.strip()
    return d


def atributos_estaticos():
    """Tabla de atributos de las cuencas de modelado: una fila por cuenca, columnas numéricas + `region`."""
    cuencas = cuencas_modelado()
    partes = [_leer("05_CAMELS_COL_Geologic_characteristics.csv"), _leer("06_CAMELS_COL_Land_cover_characteristics.csv"),
              _leer("07_CAMELS_COL_Soil_characteristics.csv").rename(columns={"water_bodies_perc": "water_bodies_perc_suelo"}),
              _leer("08_CAMELS_COL_Climatic_indices.csv"), _leer("10_CAMELS_COL_Physiograpic_characteristics.csv"),
              _leer("11_CAMELS_COL_Land_use_capability.csv").dropna(how="all")]
    partes[-1]["gauge_id"] = partes[-1]["gauge_id"].astype(int)
    tabla = None
    for p in partes:
        p = p.rename(columns={"water_bodies_perc": "water_bodies_perc_cobertura"}) if "forest_perc" in p else p
        tabla = p if tabla is None else tabla.merge(p, on="gauge_id", how="outer")
    tabla["gauge_id"] = tabla["gauge_id"].astype(int).astype(str)
    tabla = tabla.set_index("gauge_id").loc[cuencas.index]
    tabla = tabla.drop(columns=[c for c in FUERA if c in tabla.columns])
    tabla = tabla.drop(columns=[c for c in tabla.columns if tabla[c].std() == 0])     # atributos constantes: sin información

    # aridez con la ETP de Hargreaves: ETP media / lluvia media de la serie limpia
    ari = {}
    for g in tabla.index:
        s = pd.read_csv(SERIES / f"Hydromet_data_{g}.txt", sep="\t", usecols=["Precipitacion", "ETP_"])
        ari[g] = s.ETP_.mean() / s.Precipitacion.mean()
    tabla["aridez"] = pd.Series(ari)

    # logaritmo a las variables continuas positivas muy asimétricas (los % con muchos ceros se dejan como están):
    #   log(x) si todos los valores son > 0; log(1 + x / mediana de los positivos) si hay ceros.
    #   La escala evita que log1p(x) no haga nada cuando los valores son muy pequeños (p. ej. equi_slope ~ 1e-5).
    for c in list(tabla.columns):
        x = tabla[c]
        if x.skew() > 2 and (x >= 0).all() and "_perc" not in c:
            if (x > 0).all():
                tabla[f"log_{c}"] = np.log(x)
            else:
                tabla[f"log1p_{c}"] = np.log1p(x / x[x > 0].median())
            tabla = tabla.drop(columns=c)
    tabla["region"] = [REGIONES[g[0]] for g in tabla.index]
    return tabla


def variables_serie(gid, area):
    """Variables de entrada y caudal específico de una cuenca, sobre el calendario diario continuo."""
    s = pd.read_csv(SERIES / f"Hydromet_data_{gid}.txt", sep="\t")
    s["Fecha"] = pd.to_datetime(s["Fecha"], format="%d/%m/%Y")
    out = pd.DataFrame({"Fecha": s.Fecha})
    out["q"] = s.Caudal * 86.4 / area                      # mm/día; NaN en los días no ok
    out["P"] = s.Precipitacion
    for w in VENTANAS_P:                                    # exige la ventana completa: sin NaN dentro
        out[f"P{w}"] = s.Precipitacion.rolling(w, min_periods=w).sum()
    out["Tmean"] = (s.Temperatura_minima + s.Temperatura_maxima) / 2
    out["Trango"] = s.Temperatura_maxima - s.Temperatura_minima
    out["ETP"] = s.ETP_
    out["ETP30"] = s.ETP_.rolling(30, min_periods=30).mean()
    doy = s.Fecha.dt.dayofyear
    out["sin_doy"] = np.sin(2 * np.pi * doy / 365.25)
    out["cos_doy"] = np.cos(2 * np.pi * doy / 365.25)
    out["sin2_doy"] = np.sin(4 * np.pi * doy / 365.25)      # segundo armónico: regímenes con dos temporadas de lluvia
    out["cos2_doy"] = np.cos(4 * np.pi * doy / 365.25)
    out["ok"] = (s.Caudal_estado == "ok").values
    return out


def filas_cuenca(gid, area):
    """Filas utilizables de una cuenca (caudal ok y variables completas) con su partición 60/20/20 cronológica."""
    v = variables_serie(gid, area)
    dias_ok = v.loc[v.ok, "Fecha"].sort_values().reset_index(drop=True)
    f1, f2 = dias_ok.iloc[int(P_TRAIN * len(dias_ok))], dias_ok.iloc[int((P_TRAIN + P_VAL) * len(dias_ok))]
    n_ok = len(dias_ok)
    v = v[v.ok & v[DINAMICAS].notna().all(axis=1)].drop(columns="ok")
    v["particion"] = np.where(v.Fecha < f1, "entrenamiento", np.where(v.Fecha < f2, "validacion", "prueba"))
    v.insert(0, "gauge_id", gid)
    v.attrs["dias_ok"] = n_ok
    return v


def construir_filas(estaticos, areas, n_por_cuenca_train=None, semilla=42):
    """Filas de todas las cuencas. Si `n_por_cuenca_train` se da, el conjunto de entrenamiento se submuestrea por cuenca."""
    rng = np.random.default_rng(semilla)
    trozos, resumen = [], []
    for gid in estaticos.index:
        f = filas_cuenca(gid, areas[gid])
        resumen.append(dict(gauge_id=gid, dias_ok=f.attrs["dias_ok"], filas_utiles=len(f)))
        if n_por_cuenca_train is not None:
            tr = f.index[f.particion == "entrenamiento"]
            if len(tr) > n_por_cuenca_train:
                quitar = rng.choice(tr, size=len(tr) - n_por_cuenca_train, replace=False)
                f = f.drop(index=quitar)
        trozos.append(f)
    return pd.concat(trozos, ignore_index=True), pd.DataFrame(resumen).set_index("gauge_id")


def areas_cuencas(ids):
    fis = _leer("10_CAMELS_COL_Physiograpic_characteristics.csv")
    fis["gauge_id"] = fis["gauge_id"].astype(int).astype(str)
    return fis.set_index("gauge_id").area.loc[list(ids)]


def columnas(estaticos):
    num_est = [c for c in estaticos.columns if c != "region"]
    return DINAMICAS + num_est, ["region"]


def agregar_estaticos(filas, estaticos):
    """Une los atributos estáticos a cada fila (float32 para ahorrar memoria)."""
    est = estaticos.copy()
    num = [c for c in est.columns if c != "region"]
    est[num] = est[num].astype("float32")
    return filas.join(est, on="gauge_id")


def crear_preprocesamiento(col_num, col_cat):
    """Mismo esquema de la Práctica 6: imputación + escalado para numéricas, imputación + one-hot para la región."""
    return ColumnTransformer([
        ("numericas", Pipeline([("imputacion", SimpleImputer(strategy="median")), ("escalado", StandardScaler())]), col_num),
        ("categoricas", Pipeline([("imputacion", SimpleImputer(strategy="most_frequent")),
                                  ("onehot", OneHotEncoder(handle_unknown="ignore"))]), col_cat),
    ])


# ---------------------------------------------------------------- métricas por cuenca
def nse(obs, sim):
    obs, sim = np.asarray(obs, float), np.asarray(sim, float)
    d = ((obs - obs.mean()) ** 2).sum()
    return np.nan if d == 0 else 1 - ((obs - sim) ** 2).sum() / d


def kge(obs, sim):
    obs, sim = np.asarray(obs, float), np.asarray(sim, float)
    if obs.std() == 0 or sim.std() == 0 or obs.mean() == 0:
        return np.nan
    r = np.corrcoef(obs, sim)[0, 1]
    return 1 - np.sqrt((r - 1) ** 2 + (sim.std() / obs.std() - 1) ** 2 + (sim.mean() / obs.mean() - 1) ** 2)


def metricas_por_cuenca(gauge_id, obs, sim):
    """DataFrame con NSE, KGE, RMSE y MAE (mm/día) por cuenca."""
    d = pd.DataFrame({"g": np.asarray(gauge_id), "o": np.asarray(obs, float), "s": np.asarray(sim, float)})
    filas = []
    for g, x in d.groupby("g", sort=False):
        e = x.s - x.o
        filas.append(dict(gauge_id=g, NSE=nse(x.o, x.s), KGE=kge(x.o, x.s), RMSE=float(np.sqrt((e ** 2).mean())), MAE=float(e.abs().mean())))
    return pd.DataFrame(filas).set_index("gauge_id")
