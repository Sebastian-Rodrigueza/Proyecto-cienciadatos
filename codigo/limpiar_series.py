"""Limpieza de las series de CAMELS-COL desde los datos crudos, para las 346 cuencas.

Pasos (los mismos que se dedujeron de `clean_timeseries/` y se validaron contra sus 248 series):
  1. Calendario diario continuo: los días sin fila se agregan con NaN en todas las variables.
  2. ETP de Hargreaves (FAO-56) con Tmin, Tmax, día del año y la latitud de la estación (reemplaza a la ETP del dataset).
  3. Caudal_estado por día:
       no_data  calendario agregado (sin caudal)
       frozen   racha de FROZEN_MIN o más días consecutivos con exactamente el mismo caudal
       spike    caudal > SPIKE_RATIO veces el mayor de los dos días vecinos, que debe ser > 0 (día aislado)
       ok       el resto
     Los días frozen y spike quedan con Caudal = NaN.
  Justificación, fuentes, sensibilidad y limitaciones de cada regla: notebooks/limpieza_series.ipynb.
  Decisión del equipo: spike con la regla A (10x sola), aunque puede borrar crecientes reales (ver notebook, sección 3b).
  Aquí NO se descartan cuencas: las banderas de descarte se aplican en un paso posterior.

Uso (desde la raíz del repositorio, con el entorno `proyecto`):
    python codigo/limpiar_series.py
Salida: salidas/series_limpias/Hydromet_data_<gauge_id>.txt (346) y salidas/series_limpias_reporte.csv.
Si existe `clean_timeseries/` (la limpieza previa de 248 cuencas) compara contra ella.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import Transformer

FROZEN_MIN = 30
SPIKE_RATIO = 10

BASE = Path(__file__).resolve().parent.parent
RAW = BASE / "Camels col/04_CAMELS_COL_Hydrometeorological_data/3_Hydrometeorological_data"
OUT = BASE / "salidas/series_limpias"
PREVIA = next((p for p in [BASE / "salidas/clean_timeseries/clean_timeseries",
                           BASE / "ACTIVIDADES/Proyecto final/Proyecto_CAMELS-COL/clean_timeseries"]
               if (p / "cleaning_report.csv").exists()), None)


def radiacion_extraterrestre(lat, doy):
    """Ra en MJ/m2/día (FAO-56, ecuación 21)."""
    phi = np.radians(lat)
    dr = 1 + 0.033 * np.cos(2 * np.pi * doy / 365)
    d = 0.409 * np.sin(2 * np.pi * doy / 365 - 1.39)
    ws = np.arccos(np.clip(-np.tan(phi) * np.tan(d), -1, 1))
    return 24 * 60 / np.pi * 0.0820 * dr * (ws * np.sin(phi) * np.sin(d) + np.cos(phi) * np.cos(d) * np.sin(ws))


def hargreaves(df, lat):
    tmean = (df.Temperatura_minima + df.Temperatura_maxima) / 2
    rango = (df.Temperatura_maxima - df.Temperatura_minima).clip(lower=0)
    ra = radiacion_extraterrestre(lat, df.Fecha.dt.dayofyear.values)
    return (0.0023 * (tmean + 17.8) * np.sqrt(rango) * 0.408 * ra).clip(lower=0)


def marcar_caudal(q):
    """Devuelve (estado, caudal_limpio) para una serie de caudal ya con calendario continuo."""
    estado = pd.Series("ok", index=q.index)
    estado[q.isna()] = "no_data"
    # rachas de valores iguales (NaN corta la racha)
    racha = ((q != q.shift()) | q.isna()).cumsum()
    largo = q.groupby(racha).transform("size")
    frozen = q.notna() & (largo >= FROZEN_MIN)
    # picos: día aislado muy por encima de sus dos vecinos
    vecino_max = np.fmax(q.shift(), q.shift(-1))
    spike = q.notna() & q.shift().notna() & q.shift(-1).notna() & (vecino_max > 0) & (q > SPIKE_RATIO * vecino_max) & ~frozen
    estado[frozen] = "frozen"
    estado[spike] = "spike"
    return estado, q.where(estado == "ok")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    info = pd.read_csv(BASE / "Camels col/02_CAMELS_COL_Catchment_information.csv", sep=";", encoding="latin-1")
    lon, lat = Transformer.from_crs("ESRI:54004", "EPSG:4326", always_xy=True).transform(
        info.gauge_lon.values, info.gauge_lat.values)
    lat_de = dict(zip(info.gauge_id, lat))

    filas = []
    for ruta in sorted(RAW.glob("Hydromet_data_*.txt")):
        gid = int(ruta.stem.split("_")[-1])
        a = pd.read_csv(ruta, sep="\t")
        a["Fecha"] = pd.to_datetime(a["Fecha"], format="%d/%m/%Y")
        cal = (a.set_index("Fecha").reindex(pd.date_range(a.Fecha.min(), a.Fecha.max()))
                 .rename_axis("Fecha").reset_index())
        cal["ETP_"] = hargreaves(cal, lat_de[gid]).round(3)
        estado, q = marcar_caudal(cal["Caudal"])
        cal["Caudal"], cal["Caudal_estado"] = q, estado
        sal = cal.copy()
        sal["Fecha"] = sal["Fecha"].dt.strftime("%d/%m/%Y")
        sal.to_csv(OUT / f"Hydromet_data_{gid}.txt", sep="\t", index=False)
        filas.append(dict(gauge_id=str(gid), dias=len(cal), dias_agregados=len(cal) - len(a),
                          spike_dias=int((estado == "spike").sum()), frozen_dias=int((estado == "frozen").sum()),
                          q_valido_pct=round(100 * (estado == "ok").mean(), 2), etp_media=round(cal.ETP_.mean(), 3)))
    rep = pd.DataFrame(filas)
    rep.to_csv(BASE / "salidas/series_limpias_reporte.csv", index=False)
    print(f"{len(rep)} series escritas en {OUT}")
    print(f"picos: {rep.spike_dias.sum()}   congelados: {rep.frozen_dias.sum()}   filas agregadas: {rep.dias_agregados.sum():,}")

    if PREVIA is not None:
        print(f"\nComparación con la limpieza previa ({PREVIA}):")
        prev = pd.read_csv(PREVIA / "cleaning_report.csv", dtype={"gauge_id": str})
        prev = prev[prev.kept].merge(rep, on="gauge_id")
        print(f"  cuencas comparadas: {len(prev)}")
        print(f"  picos:      suma previa {int(prev.spike_days.sum())}  vs  nueva {int(prev.spike_dias.sum())}")
        print(f"  congelados: suma previa {int(prev.frozen_days.sum())}  vs  nueva {int(prev.frozen_dias.sum())}")
        dif_estado, dif_q, dif_etp = 0, 0, 0.0
        for g in prev.gauge_id:
            n = pd.read_csv(OUT / f"Hydromet_data_{g}.txt", sep="\t")
            p = pd.read_csv(PREVIA / f"Hydromet_data_{g}.txt", sep="\t")
            assert (n.Fecha == p.Fecha).all(), g
            dif_estado += int((n.Caudal_estado != p.Caudal_estado).sum())
            dif_q += int((~np.isclose(n.Caudal, p.Caudal, equal_nan=True)).sum())
            dif_etp = max(dif_etp, float((n.ETP_ - p.ETP_).abs().max()))
        print(f"  días con Caudal_estado distinto: {dif_estado}   días con caudal distinto: {dif_q}   dif. máx. ETP: {dif_etp:.4f} mm/día")


if __name__ == "__main__":
    main()
