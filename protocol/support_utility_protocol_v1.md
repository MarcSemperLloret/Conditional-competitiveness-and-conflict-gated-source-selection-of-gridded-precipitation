# Support Utility Protocol v1

## Status

- Fecha: 2026-04-08
- Paper: `paper_11_density_thresholds`
- Revista objetivo: `Remote Sensing Applications: Society and Environment`
- Rol de este documento: protocolo principal del manuscrito

## Regla principal

Este protocolo busca responder una pregunta de utilidad de productos EO:

**en que condiciones los productos gridded de precipitacion mantienen valor operativo frente a estimadores locales gauge-assisted cuando el soporte in situ es heterogeneo**

## Contribution claim

El paper no presenta un nuevo producto EO ni un nuevo interpolador. Presenta:

- una evaluacion support-aware de utilidad relativa
- una lectura de limites y regimenes de fallo
- una comparacion contra una referencia local leave-cell sin leakage

## Activos congelados

- `paper_10_fusion_fail/data/processed_v3/canonical_cell_hour_imerg_v1.parquet`
- `paper_10_fusion_fail/data/processed_v3/imerg_cell_static_v1.parquet`
- `paper_10_fusion_fail/data/avamet/intermediate/avamet_station_inventory_cv_imerg.csv`
- `paper_10_fusion_fail/data/avamet/processed/avamet_cv_hourly_2019_2023.parquet`
- `paper_10_fusion_fail/results/baselines_v3/baseline_report_v3.md`

## Comparaciones principales

### Productos EO

- IMERG
- ERA5 o ERA5-Land segun benchmark congelado
- EURADCLIM agregado a rejilla IMERG

### Referencias locales

- `domain_idw_leavecell_knn08`
- `domain_idw_leavecell_knn64`
- `domain_idw_leavecell_knn64_r15km`

Regla:

- `k=8` y `k=64_r15km` son candidatos principales para la comparacion con EO
- `k=64` se mantiene como referencia de geometria amplia y cobertura plena

## Escenarios de evaluacion

### Escenario principal

`robust_ge_2`

- `avamet_n_active >= 2`
- AVAMET target finito
- IMERG finito
- ERA5 finito
- EURADCLIM finito

### Sensibilidad

`wide_ge_1`

- `avamet_n_active >= 1`

## Variables interpretativas obligatorias

- `avamet_n_active`
- `avamet_support_score_v0`
- `support_regime_label`
- `observable_conflict_bin`
- `target_intensity_bin_v2`
- `elevation_bin`
- `coast_bin`

## Preguntas que debe cerrar el protocolo

1. Donde cada producto EO sigue siendo competitivo frente a leave-cell.
2. Donde cada producto deja de ser competitivo.
3. Cuanto cambia esa lectura cuando el soporte local es fuerte o debil.
4. Si el conflicto entre fuentes anticipa perdida de utilidad operativa.
5. Si la lluvia intensa rompe antes la competitividad EO que la lluvia moderada.

## Estimandos principales

### Estimando 1

`relative utility regime map`

Para cada producto EO y para cada estrato:

- diferencia de `MAE` frente al mejor estimador local permitido
- signo de la ventaja relativa

### Estimando 2

`support-conditioned competitiveness`

Cambio de competitividad EO vs leave-cell segun:

- `avamet_n_active`
- `support_regime_label`
- `avamet_support_score_v0`

### Estimando 3

`conflict-conditioned degradation`

Cambio de rendimiento EO vs local segun:

- `observable_conflict_bin`

### Estimando 4

`heavy-rain failure regime`

Comparacion en:

- `(1,5]`
- `(5,10]`
- `(10,20]`
- `>20`

## Metricas primarias

- `MAE`
- `RMSE`
- `correlation`
- `bias`
- `coverage_share`

## Metricas derivadas

- delta `MAE` producto EO vs mejor baseline local
- tasa de victoria por producto y estrato
- ranking por regimen
- sensibilidad de ranking al soporte local

## Analisis principal

### Analisis 1 - Ranking global honesto

- ranking global en `robust_ge_2`
- ranking global en `wide_ge_1`
- dejar claro que el ranking global no agota la lectura del paper

### Analisis 2 - Regimenes de competitividad

- comparar productos EO vs referencias locales por:
  - intensidad
  - conflicto
  - elevacion
  - costa
  - soporte

### Analisis 3 - Lectura de soporte

- analizar como cambia la conclusion al pasar de `wide_ge_1` a `robust_ge_2`
- usar `avamet_support_score_v0` para ordenar los estratos de fiabilidad local

### Analisis 4 - Error interpretation

- relacionar error EO con:
  - conflicto
  - soporte
  - intensidad
  - relieve

### Analisis 5 - Sensibilidad a thinning

- usar `thinning_protocol_v1.md` solo para comprobar si la lectura de utilidad EO cambia cuando se degrada artificialmente el soporte local

## Figuras obligatorias

1. Esquema conceptual de productos EO, soporte local y referencia leave-cell.
2. Tabla o heatmap de ranking por regimen.
3. Figura de delta `MAE` EO vs leave-cell por soporte y conflicto.
4. Figura de lluvia intensa.
5. Figura de sensibilidad a degradacion del soporte local.

## Reglas editoriales

- El opening del paper debe empezar con utilidad de productos EO, no con AVAMET.
- No usar un titulo centrado en IMERG.
- No vender leave-cell como nuevo metodo; es la referencia local de comparacion.
- No vender thinning como contribucion principal.
- La discusion debe cerrar en terminos de product utility, support mismatch y limites operativos.

## Criterio de exito

El protocolo se considera exitoso si permite afirmar:

- que la utilidad de productos EO depende del soporte local y del regimen
- que el ranking entre productos no es estable en todos los contextos
- que las conclusiones cambian de forma interpretable al cambiar el soporte observacional

## Criterio de fracaso

El protocolo fracasa si el paper acaba siendo:

- una validacion mas de IMERG
- un benchmark regional sin mensaje general
- una comparacion de errores sin interpretacion de soporte

## Siguiente implementacion

El siguiente paso practico no es reescribir el pipeline entero.

Es producir una nota de resultados congelados para esta historia:

**`support_utility_results_notes_v1.md`**

Esa nota debe extraer del material ya existente:

- que producto gana donde
- donde el baseline local cambia
- y que slices cuentan realmente la historia EO del paper
