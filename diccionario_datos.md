# Diccionario de datos — CAMELS-COL

Explicación en español sencillo de cada variable del dataset. La fuente oficial (en inglés) es
`Camels col/01_CAMELS_COL_Attributes/01_CAMELS_COL_Attributes/01_CAMELS_COL_Attributes.xlsx`.

Cada fila de las tablas de atributos es **una cuenca**, identificada por `gauge_id`.

---

## 1. Conceptos básicos (leer primero)

| Término | Qué es |
|---|---|
| **Cuenca** (*catchment*, *basin*) | Todo el terreno cuya agua de lluvia escurre hacia un mismo punto de un río. Imaginá un embudo gigante: todo lo que cae adentro termina saliendo por el pico. |
| **Estación de aforo** (*gauge*) | El punto del río donde se mide el caudal. Es la "salida" (el pico del embudo) de la cuenca. Cada cuenca del dataset está definida por una estación del IDEAM. |
| **Caudal** (*streamflow*, *discharge*, Q) | Cuánta agua pasa por el río en un momento. Se mide en m³/s (metros cúbicos por segundo). |
| **Caudal en mm/día** | El mismo caudal, pero dividido por el área de la cuenca. Así se pueden comparar cuencas grandes y chicas: es como si toda el agua que sale se repartiera como una lámina sobre la cuenca. Se puede comparar directamente con la lluvia (también en mm). |
| **Precipitación** (P, *pr*) | Lluvia que cae sobre la cuenca, en mm. 1 mm = 1 litro por m². |
| **Evapotranspiración potencial** (ETP, PET, ETo) | Cuánta agua *podría* evaporarse del suelo y transpirar las plantas si hubiera agua de sobra. Mide la "sed" de la atmósfera, y depende de la temperatura, el sol y el viento. |
| **Escorrentía** (*runoff*) | La parte de la lluvia que termina saliendo por el río (no se evapora ni se queda en el suelo). |
| **Flujo base** (*baseflow*) | El agua que llega al río lentamente desde el subsuelo. Es lo que mantiene al río con agua aunque no llueva por semanas. |
| **Firma hidrológica** (*hydrological signature*) | Un número que resume cómo se comporta el río (si es estable, si tiene crecidas fuertes, etc.), calculado a partir de la serie de caudal. |
| **Percentil / cuantil** | El valor por debajo del cual queda cierto % de los datos. P.ej. el percentil 95 del caudal es un valor que solo se supera el 5 % de los días. |

### Fuentes de datos que aparecen
- **IDEAM**: Instituto de Hidrología, Meteorología y Estudios Ambientales. Mide los caudales en las estaciones.
- **IGAC**: Instituto Geográfico Agustín Codazzi. Aporta el modelo de elevación (DEM), los suelos y la capacidad de uso.
- **DEM**: *Digital Elevation Model*, un mapa de alturas del terreno.
- **CHIRPS v2.0**: lluvia estimada por satélite combinada con estaciones. Ojo: es una estimación, no una medición directa.
- **MSWX**: datos meteorológicos (temperatura, ETP) reconstruidos con modelos.
- **MapBiomas**: mapas de cobertura del suelo hechos con imágenes satelitales.

---

## 2. Información de la estación — `02_CAMELS_COL_Catchment_information.csv`

| Variable | Qué significa | Unidad |
|---|---|---|
| `gauge_id` | Código IDEAM de la estación. Es la **llave** para unir todas las tablas. | — |
| `gauge_department` | Departamento donde está la estación. | — |
| `gauge_lat` | Coordenada norte de la estación. **No está en grados**: está en metros (proyección World Mercator, EPSG:3395). | m |
| `gauge_lon` | Coordenada este, también en metros (EPSG:3395). | m |
| `gauge_elev` | Altura de la estación sobre el nivel del mar. | m s. n. m. |
| `gauge_star` | Fecha del primer dato (formato dd/mm/aaaa). | fecha |
| `gauge_end` | Fecha del último dato. | fecha |
| `gauge_n` | Cuántos días con dato tiene la serie. | días |
| `gauge_missing` | % de días **sin** dato entre la fecha inicial y la final. | % |

---

## 3. Series de tiempo diarias — `04_.../Hydromet_data_<gauge_id>.txt`

Un archivo por cuenca, una fila por día. **Los nombres reales de las columnas son distintos a los del diccionario oficial**:

| Columna en el archivo | Nombre en el diccionario oficial | Qué significa | Unidad | Fuente |
|---|---|---|---|---|
| `Fecha` | — | Día de la medición (dd/mm/aaaa). | fecha | — |
| `Precipitacion` | `pr` | Lluvia promedio sobre toda la cuenca ese día. | mm/día | CHIRPS |
| `ETP_` | `poten_evapo` | Evapotranspiración potencial promedio de la cuenca. | mm/día | MSWX |
| `Temperatura_minima` | `t_min` | Temperatura mínima del día, promedio de la cuenca. | °C | MSWX |
| `Temperatura_maxima` | `t_max` | Temperatura máxima del día, promedio de la cuenca. | °C | MSWX |
| `Caudal` | `streamflow` | Caudal medio del día en la estación. | m³/s | IDEAM |

> ⚠️ `t_mean` (temperatura media) aparece en el diccionario oficial pero **no está en los archivos**. Se puede aproximar como (mín + máx) / 2.
> ⚠️ Los días faltantes **no aparecen como celdas vacías**: la fila del día directamente no existe.

---

## 4. Características fisiográficas (forma y relieve) — `10_..._Physiograpic_characteristics.csv`

| Variable | Qué significa | Cómo leerla | Unidad |
|---|---|---|---|
| `area` | Área de la cuenca. | — | km² |
| `perimeter` | Largo del borde de la cuenca. | — | km |
| `gravelius_index` | Índice de compacidad: compara el perímetro de la cuenca con el de un círculo de la misma área. | 1 = circular. Mientras más alto, más alargada o irregular. Las cuencas circulares concentran el agua más rápido (crecidas más bruscas). | — |
| `factor_form` | Factor de forma: área / (longitud de la cuenca)². | Bajo = cuenca alargada; alto = ancha o compacta. | — |
| `maximum_ele` | Altura máxima de la cuenca. | — | m |
| `mean_ele` | Altura media de la cuenca. | — | m |
| `minimum_ele` | Altura mínima, normalmente en la estación. | — | m |
| `streng_chanel` | Longitud del cauce principal (el río más largo de la cuenca). *"Streng"* es un error de tipeo de *length*. | — | km |
| `equi_slope` | Pendiente equivalente del cauce principal. | Valores muy chicos (~0,00001–0,0016). Más alta = río más empinado, agua más rápida. | m/m |
| `cn_catchment` | **Número de curva** (método SCS): qué tan fácil se convierte la lluvia en escorrentía, según el suelo y la cobertura. | Va de ~30 a 100. Alto = suelo poco permeable, mucha lluvia escurre de inmediato. Bajo = el suelo absorbe más. | — |
| `tc_kirpich` | **Tiempo de concentración** según la fórmula de Kirpich: cuánto tarda el agua en viajar desde el punto más lejano de la cuenca hasta la estación. | Más largo = la crecida llega más lenta y suavizada. | horas |
| `tc_chow` | Tiempo de concentración, fórmula de Ven Te Chow. | Igual que arriba, otra fórmula empírica. | horas |
| `tc_Johnstone` | Tiempo de concentración, fórmula de Johnstone. | Igual. | horas |
| `tc_Engi_Corps` | Tiempo de concentración, fórmula del Cuerpo de Ingenieros de EE. UU. | Igual. | horas |
| `tc_representative` | El tiempo de concentración **adoptado** para la cuenca (elegido entre los anteriores con el criterio del artículo). | Usar este si se necesita uno solo. | horas |

**Capa auxiliar `13_.../TWI_<gauge_id>.tif`**: el **Índice Topográfico de Humedad** (TWI) es un mapa (raster) que muestra dónde tiende a acumularse el agua en el terreno, según la pendiente y cuánta área drena hacia cada punto. Valores altos = zonas planas y bajas que se mantienen húmedas (fondos de valle). Solo existe para 210 de las 346 cuencas.

---

## 5. Índices climáticos — `08_CAMELS_COL_Climatic_indices.csv`

| Variable | Qué significa | Cómo leerla | Unidad |
|---|---|---|---|
| `aridity` | **Índice de aridez** (Budyko) = ETP media / lluvia media. | < 1: llueve más de lo que la atmósfera puede evaporar (húmedo). > 1: la atmósfera "pide" más agua de la que llueve (seco). | — |
| `high_prec_freq` | Cuántos días al año llueve muy fuerte (≥ 5 veces la lluvia diaria promedio). | Más alto = más días de aguaceros extremos. | días/año |
| `high_prec_dur` | Cuántos días seguidos duran en promedio esos eventos de lluvia fuerte. | — | días |
| `low_prec_freq` | Cuántos días al año son secos (< 1 mm de lluvia). | — | días/año |
| `low_prec_dur` | Cuántos días seguidos duran en promedio las rachas secas. | Más alto = sequías más largas. | días |

---

## 6. Firmas hidrológicas (cómo se comporta el río) — `09_CAMELS_COL_Hydrological_signatures.csv`

| Variable | Qué significa | Cómo leerla | Unidad |
|---|---|---|---|
| `q_mean` | Caudal medio diario, normalizado por el área. | Comparable entre cuencas y con la lluvia. | mm/día |
| `runoff_ratio` | **Coeficiente de escorrentía** = caudal medio / lluvia media. Qué fracción de la lluvia sale por el río. | Entre 0 y 1 en teoría. 0,6 = el 60 % de la lluvia sale por el río y el resto se evapora. **> 1 es físicamente raro**: puede indicar que CHIRPS subestima la lluvia, aportes subterráneos o error de datos. | — |
| `stream_elas` | **Elasticidad** caudal–lluvia: si la lluvia anual sube 1 %, ¿cuánto % sube el caudal? | 2 = si llueve 1 % más, el caudal sube 2 %. Más alto = río más sensible a cambios del clima. | — |
| `slope_fdc` | Pendiente de la **curva de duración de caudales** (entre los percentiles 33 y 66, en escala log). | Alta = río "nervioso", cambia mucho entre días (respuesta rápida). Baja = río estable, regulado por el subsuelo. | — |
| `baseflow_index` | **Índice de flujo base**: qué fracción del caudal viene del agua subterránea / lenta. | Cerca de 1 = río alimentado sobre todo por el subsuelo (estable). Cerca de 0 = depende de la lluvia directa. | — |
| `Q5` | Caudal del **percentil 5**: caudal bajo, que solo el 5 % de los días es menor. | Representa los estiajes (época seca). | mm/día |
| `Q95` | Caudal del **percentil 95**: caudal alto, que solo el 5 % de los días se supera. | Representa las crecidas. | mm/día |
| `high_q_freq` | Cuántos días al año hay crecidas (caudal > 9 veces la mediana). | — | días/año |
| `high_q_dur` | Cuántos días seguidos duran en promedio esas crecidas. | — | días |
| `low_q_freq` | Cuántos días al año hay caudal bajo (< 0,2 veces el caudal medio). | — | días/año |
| `low_q_dur` | Cuántos días seguidos duran en promedio los periodos de caudal bajo. | — | días |

> ⚠️ Ojo con la convención: en CAMELS `Q5` es el caudal **bajo** y `Q95` el **alto** (al revés de la convención hidrológica clásica, donde Q95 es el caudal excedido el 95 % del tiempo).

---

## 7. Cobertura del suelo — `06_CAMELS_COL_Land_cover_characteristics.csv`

Qué hay *encima* del suelo, como % del área de la cuenca. Fuente: MapBiomas 2022. Las columnas suman 100 %.

| Variable | Qué significa |
|---|---|
| `forest_perc` | % cubierto por bosque. |
| `nat_non_forest_form_perc` | % de vegetación natural que no es bosque (páramos, sabanas, pastizales naturales, humedales). |
| `agricul_livestock_perc` | % de cultivos y ganadería. Indicador de intervención humana. |
| `non_vegeted_perc` | % sin vegetación (zonas urbanas, minería, roca, arena). |
| `water_bodies_perc` | % de cuerpos de agua (ríos anchos, lagos, embalses). |
| `non_identi_land_perc` | % que no se pudo clasificar (normalmente por nubes en las imágenes). |

---

## 8. Geología — `05_CAMELS_COL_Geologic_characteristics.csv`

Qué tipo de roca hay *debajo*, como % del área. Influye en cuánta agua se infiltra y se almacena. Fuente: Servicio Geológico Colombiano (Gómez et al., 2023).

| Variable | Qué significa |
|---|---|
| `unconso_depo_perc` | % de **depósitos no consolidados**: material suelto (arena, grava, arcilla) dejado por ríos o glaciares. Suele ser muy permeable. |
| `hypabyssal_rock_perc` | % de **roca hipoabisal**: magma que se enfrió cerca de la superficie, sin llegar a salir. |
| `metamor_rock_perc` | % de **roca metamórfica**: roca transformada por presión y calor (p.ej. esquistos, gneis). Poco permeable salvo si está fracturada. |
| `plutonic_rock_perc` | % de **roca plutónica**: magma enfriado lentamente en profundidad (p.ej. granito). Poco permeable. |
| `sedimen_rock_perc` | % de **roca sedimentaria**: capas de sedimento compactado (areniscas, calizas, lutitas). Permeabilidad variable; las calizas y areniscas pueden almacenar mucha agua. |
| `volcanic_rock_perc` | % de **roca volcánica**: lava enfriada en superficie. |
| `volcaniclastic_rock_perc` | % de **roca volcanoclástica**: fragmentos y cenizas expulsados por volcanes. |

> ⚠️ **Problemas de calidad detectados:**
> - `volcaniclastic_rock_perc` es prácticamente **idéntica** a `volcanic_rock_perc` (correlación = 1,0; diferencias solo de redondeo). Parece una columna duplicada por error: usar solo una de las dos.
> - Las columnas de geología **no suman 100 %**: la suma va de 8,7 % a 200 % según la cuenca (mediana 100 %). El duplicado explica parte del exceso, pero no los casos muy por debajo.

---

## 9. Suelos (taxonomía) — `07_CAMELS_COL_Soil_characteristics.csv`

Tipo de suelo como % del área. Fuente: IGAC. Los nombres son **órdenes de suelo** de la clasificación de EE. UU. (USDA):

| Variable | Qué tipo de suelo es |
|---|---|
| `inceptisols_perc` | **Inceptisoles**: suelos jóvenes, poco desarrollados. Muy comunes en las laderas andinas. |
| `entisols_perc` | **Entisoles**: suelos muy jóvenes, casi sin capas. Típicos de depósitos de río recientes o laderas muy empinadas. |
| `andisols_perc` | **Andisoles**: formados sobre ceniza volcánica (Eje Cafetero, sur andino). Retienen muchísima agua. |
| `alfisols_perc` | **Alfisoles**: suelos fértiles moderadamente lavados por la lluvia. |
| `vertisols_perc` | **Vertisoles**: suelos arcillosos que se agrietan cuando se secan. Poco permeables cuando están húmedos. |
| `aridisols_perc` | **Aridisoles**: suelos de zonas secas (p.ej. La Guajira). |
| `ultisols_perc` | **Ultisoles**: suelos viejos, muy lavados y ácidos. |
| `spodosols_perc` | **Espodosoles**: suelos arenosos y ácidos. |
| `histosols_perc` | **Histosoles**: suelos orgánicos (turba), típicos de páramos y humedales. Funcionan como esponjas. |
| `oxisols_perc` | **Oxisoles**: suelos muy viejos y meteorizados, típicos de la Amazonía y la Orinoquía. |
| `mollisols_perc` | **Molisoles**: suelos oscuros y muy fértiles. |
| `urban_perc` | % de zona urbana (sin suelo natural). |
| `water_bodies_perc` | % de cuerpos de agua. ⚠️ Mismo nombre que en la tabla de cobertura: al unir tablas, pandas le agrega un sufijo. |
| `coal_mine_pit_perc` | % de minas de carbón a cielo abierto. |
| `eroded_misce_perc` | % de zonas erosionadas (misceláneo erosionado). |
| `rocky_misce_perc` | % de afloramientos rocosos (misceláneo rocoso). |
| `perpetual_snow_perc` | % de nieves perpetuas (glaciares). |

---

## 10. Capacidad de uso del suelo — `11_CAMELS_COL_Land_use_capability.csv`

Clasificación del IGAC (2014) que dice **para qué sirve la tierra**, como % del área. Las clases van de 1 (la mejor tierra) a 8 (solo apta para conservación):

| Variable | Qué significa |
|---|---|
| `class_1` | Tierras excelentes para agricultura intensiva, sin limitaciones (planas, fértiles). |
| `class_2` | Muy buenas para agricultura, con limitaciones leves. |
| `class_3` | Aptas para agricultura con limitaciones moderadas (requieren prácticas de manejo). |
| `class_4` | Agricultura limitada; mejores para cultivos permanentes o pastos. |
| `class_5` | Limitaciones fuertes por exceso de agua (inundables); aptas para pastos o usos especiales. |
| `class_6` | Aptas para ganadería, cultivos permanentes o agroforestería, no para cultivos limpios. |
| `class_7` | Aptas sobre todo para bosque productor y protector (pendientes fuertes, suelos frágiles). |
| `class_8` | Solo para conservación y protección (páramos, zonas muy empinadas, nacimientos de agua). |
| `Urban_area` | % de zona urbana. |
| `No_classified` | % sin clasificar. |

---

## 11. Capas espaciales (mapas)

| Archivo | Qué contiene |
|---|---|
| `03_.../CAMELS_COL_catchments_boundaries.shp` | Polígonos con el límite de cada cuenca. |
| `03_.../CAMELS_COL_GAUGING.shp` | Puntos con la ubicación de las estaciones. |
| `12_.../Drainage_network.shp` | Red de ríos (líneas) de las cuencas. |
| `13_.../TWI_<gauge_id>.tif` | Índice topográfico de humedad (raster), ver sección 4. |

Todas están en la proyección World Mercator (en metros). Se abren con `geopandas` (shapefiles) y `rasterio` (TIF).
