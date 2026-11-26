"""Paso 3: banderas de descarte de cuencas y lista final para modelar.

Seis banderas (una cuenca con cualquiera de ellas se descarta):
  f_q_gt_p              caudal mayor que la precipitación: coeficiente de escorrentía > 1
                        (calculado aquí con las series crudas y comparado con `runoff_ratio` del archivo 09)
  f_flatline            >= FLAT_PCT % de los días con caudal están en rachas de >= FLAT_RUN días iguales
                        (umbral inferido de la limpieza previa: separa sus 19 cuencas marcadas de las demás sin errores)
  f_regime_mismatch     el régimen anual del caudal no coincide con el de la lluvia        (criterio externo)
  f_nested_inconsistent cuenca anidada con caudales que no cuadran con la de aguas arriba (criterio externo)
  f_delineation         contorno de la cuenca (área) mal trazado                           (criterio externo)
  f_twin_mismatch       dos estaciones casi en el mismo punto con datos que no coinciden   (criterio externo)
Las cuatro últimas no se pueden calcular con los datos del repositorio: se toman de `cleaning_report.csv`
(columna `drop_flags` de la limpieza previa), que es el criterio del equipo.

Uso (desde la raíz del repositorio): python codigo/aplicar_banderas.py
Salida: salidas/banderas_descarte.csv (346 filas) y salidas/cuencas_modelado.csv (cuencas que se conservan).
"""
from pathlib import Path

import numpy as np
import pandas as pd

FLAT_RUN = 7
FLAT_PCT = 10.0

BASE = Path(__file__).resolve().parent.parent
RAW = BASE / "Camels col/04_CAMELS_COL_Hydrometeorological_data/3_Hydrometeorological_data"
PREVIA = next((p for p in [BASE / "salidas/clean_timeseries/clean_timeseries",
                           BASE / "ACTIVIDADES/Proyecto final/Proyecto_CAMELS-COL/clean_timeseries"]
               if (p / "cleaning_report.csv").exists()), None)
EXTERNAS = ["f_regime_mismatch", "f_nested_inconsistent", "f_delineation", "f_twin_mismatch"]


def pct_en_rachas(q, largo_min=FLAT_RUN):
    """% de los días con caudal que están en rachas de `largo_min` o más días consecutivos iguales."""
    q = q.dropna()
    racha = (q != q.shift()).cumsum()
    largo = q.groupby(racha).transform("size")
    return 100 * (largo >= largo_min).mean()


def calcular_bandas_propias():
    """f_q_gt_p y f_flatline para las 346 cuencas, desde las series crudas."""
    info = pd.read_csv(BASE / "Camels col/02_CAMELS_COL_Catchment_information.csv", sep=";", encoding="latin-1")
    fis = pd.read_csv(BASE / "Camels col/10_CAMELS_COL_Physiograpic_characteristics.csv", sep=";")
    fis.columns = fis.columns.str.strip()
    area = fis.set_index("gauge_id")["area"]
    firmas = pd.read_csv(BASE / "Camels col/09_CAMELS_COL_Hydrological_signatures.csv", sep=",").set_index("gauge_id")
    filas = []
    for gid in info.gauge_id:
        s = pd.read_csv(RAW / f"Hydromet_data_{gid}.txt", sep="\t")
        s["Fecha"] = pd.to_datetime(s["Fecha"], format="%d/%m/%Y")
        s = s.set_index("Fecha").reindex(pd.date_range(s.Fecha.min(), s.Fecha.max()))
        q_mm = s.Caudal * 86.4 / area[gid]
        # coeficiente de escorrentía con los días que tienen caudal y lluvia
        ok = s.Caudal.notna() & s.Precipitacion.notna()
        coef = q_mm[ok].mean() / s.Precipitacion[ok].mean()
        filas.append(dict(gauge_id=str(gid), area=area[gid], coef_serie=coef, runoff_ratio=firmas.runoff_ratio[gid],
                          pct_rachas7=pct_en_rachas(s.Caudal)))
    d = pd.DataFrame(filas)
    d["f_q_gt_p"] = d.runoff_ratio > 1
    d["f_flatline"] = d.pct_rachas7 >= FLAT_PCT
    return d


def main():
    prop = calcular_bandas_propias()
    prev = pd.read_csv(PREVIA / "cleaning_report.csv", dtype={"gauge_id": str})
    flags_prev = prev.set_index("gauge_id").drop_flags.fillna("")
    d = prop.set_index("gauge_id")
    for f in EXTERNAS:
        d[f] = flags_prev.reindex(d.index).str.contains(f)
    cols = ["f_q_gt_p", "f_flatline"] + EXTERNAS
    d["n_banderas"] = d[cols].sum(axis=1)
    d["conservada"] = d.n_banderas == 0
    d["region"] = d.index.str[0].map({"1": "Caribe", "2": "Magdalena-Cauca", "3": "Orinoco", "4": "Amazonas", "5": "Pacífico"})
    d.reset_index().to_csv(BASE / "salidas/banderas_descarte.csv", index=False)
    d[d.conservada].reset_index()[["gauge_id", "region"]].to_csv(BASE / "salidas/cuencas_modelado.csv", index=False)

    print(f"Cuencas: {len(d)}   conservadas: {int(d.conservada.sum())}   descartadas: {int((~d.conservada).sum())}")
    print(d[cols].sum().to_string())
    # comparación con la limpieza previa
    kept_prev = prev.set_index("gauge_id").kept.reindex(d.index)
    print(f"\nCoinciden con `kept` de la limpieza previa: {int((d.conservada == kept_prev).sum())} de {len(d)}")
    for f, clave in [("f_q_gt_p", "f_q_gt_p"), ("f_flatline", "f_flatline")]:
        previa = flags_prev.reindex(d.index).str.contains(clave)
        print(f"  {f}: nuestra {int(d[f].sum())}  previa {int(previa.sum())}  distintas {int((d[f] != previa).sum())}")


if __name__ == "__main__":
    main()
