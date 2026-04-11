# Planning v1 - paper_11_density_thresholds (active RSASE framing)

## Snapshot

- Estado: activo
- Fecha de pivot: 2026-04-08
- Zona horaria: Europe/Madrid
- Directorio de trabajo: `paper_11_density_thresholds`
- Revista objetivo: `Remote Sensing Applications: Society and Environment`
- Plan editorial: paper de utilidad operativa y limites de productos EO de precipitacion bajo soporte in situ heterogeneo
- Restriccion clave: no recaer en narrativa de correccion de IMERG ni en paper puramente hidrologico de densidad de red

## Titulo de trabajo

### Opcion preferida
**Operational utility and limits of EO precipitation products under heterogeneous in situ support**

### Opcion mas aplicada
**When are EO precipitation products competitive with local gauge-assisted estimates? Evidence under heterogeneous support and complex terrain**

### Opcion mas observacional
**Support-aware evaluation of EO precipitation products against dense in situ observations and leave-cell local estimates**

## Decision editorial

Este articulo no debe venderse como:

- un paper de fusion
- un paper de correccion de IMERG
- un paper de interpolacion de gauges
- un paper sobre thinning como contribucion principal

Este articulo debe venderse como:

- un paper de remote sensing aplicado con significacion general
- una evaluacion support-aware de utilidad y limites de productos EO
- una comparacion entre productos gridded y estimadores locales bajo heterogeneidad observacional
- una lectura de en que regimenes los productos EO siguen siendo utiles y en cuales no

## Regla de framing general

El paper debe girar en torno a esta pregunta:

**cuando el soporte in situ es heterogeneo y la precipitacion tiene estructura espacio-temporal compleja, en que condiciones los productos EO mantienen valor operativo frente a estimadores locales gauge-assisted**

AVAMET y el dominio mediterraneo deben aparecer como:

- banco de pruebas observacional exigente
- plataforma de validacion densa
- caso empirico para revelar mecanismos generales de utilidad y fallo

No deben aparecer como:

- protagonista de dataset
- contribucion principal por si mismos
- excusa para un caso regional descriptivo

## Tesis central

La utilidad de productos EO de precipitacion no es fija ni uniforme: depende del soporte in situ disponible para validarlos, de la heterogeneidad espacial del campo de precipitacion, del conflicto entre fuentes, de la intensidad del evento y de la escala efectiva que compara producto y observacion.

## Tesis en una frase

Los productos EO no son simplemente "buenos" o "malos": su valor operativo cambia con el soporte observacional y con el regimen fisico que se esta intentando resolver.

## Preguntas fuertes del paper

1. En que regimenes IMERG, ERA5 y EURADCLIM siguen siendo competitivos frente a estimadores locales leave-cell.
2. Cuanto del error aparente puede atribuirse a soporte in situ limitado, desalineacion espacial o heterogeneidad subgrid.
3. Como cambia la utilidad relativa de cada producto con intensidad, conflicto entre fuentes, relieve y distancia a costa.
4. Cuando un estimador local gauge-assisted mejora claramente a los productos EO y cuando esa ventaja desaparece.
5. Si un analisis de degradacion del soporte in situ cambia la lectura sobre el valor de los productos EO.

## Hipotesis de trabajo

- H1. La competitividad de productos EO depende fuertemente del soporte local disponible para contrastarlos.
- H2. Los productos EO mantienen valor operativo relativamente mayor en situaciones de soporte pobre o incertidumbre observacional alta.
- H3. En alto conflicto e intensidades elevadas, la ventaja de productos EO frente a estimadores locales se degrada.
- H4. Parte del error atribuido a un producto EO refleja mismatch de soporte y representatividad, no solo fallo del sensor o algoritmo.
- H5. El thinning controlado puede usarse como analisis auxiliar para separar limitaciones del producto de limitaciones del sistema de referencia local.

## Lo que ya esta soportado por el material existente

- Benchmark canonico 2019-2023 ya construido en rejilla IMERG con AVAMET, IMERG, ERA5, EURADCLIM y contexto fisico.
- Comparativa ya existente entre familia `gauge-assisted` y productos gridded.
- Senal fuerte ya visible:
  - `k=8` mejora `MAE` en varios regimenes.
  - `k=64` mantiene ventaja en `RMSE`, correlacion y cobertura.
  - `r=15 km` sigue siendo competitivo.
  - `r=10 km` degrada cobertura.
- Regimenes ya definidos:
  - intensidad
  - conflicto entre fuentes
  - elevacion
  - costa
  - soporte gauge

## Hueco que aun no esta cerrado

- No existe un framing cerrado de "product utility regimes" para EO.
- No existe una tabla final de competitividad por producto y por regimen.
- Falta un bloque interpretativo claro sobre soporte, representatividad y heterogeneidad.
- El thinning existe ya como pipeline, pero aun no esta colocado como analisis secundario de interpretacion.

## Contribucion esperada

La contribucion no es un nuevo producto ni un nuevo interpolador. La contribucion es:

- una lectura support-aware de utilidad operativa de productos EO de precipitacion
- una caracterizacion de limites y regimenes de fallo bajo heterogeneidad observacional
- una comparacion honesta frente a estimadores locales leave-cell
- una interpretacion de cuanto cambia la conclusion cuando cambia el soporte in situ disponible

## Regla de redaccion

Cada resultado principal debe poder escribirse en dos niveles:

- nivel EO general: que dice sobre utilidad, limites o interpretacion de productos remotos
- nivel empirico local: como aparece ese patron en AVAMET y en el dominio de prueba

Si una figura solo puede leerse como "caso Valenciano", no debe ser figura principal.

## Estructura minima del paper

1. Introduccion
2. Study area and observational platform
3. Data and support-aware benchmark design
4. Gauge-assisted local reference and leave-cell protocol
5. Product utility across support and regime conditions
6. Error interpretation: conflict, support, and heterogeneity
7. Support degradation sensitivity
8. Discussion and operational implications
9. Conclusions

## Experimentos obligatorios

### Bloque A - Congelar comparacion principal

- Releer y congelar resultados de:
  - IMERG
  - ERA5 o ERA5-Land
  - EURADCLIM
  - `domain_idw_leavecell_knn08`
  - `domain_idw_leavecell_knn64`
  - `domain_idw_leavecell_knn64_r15km`

### Bloque B - Regimenes de utilidad

- Comparar por:
  - soporte gauge
  - intensidad de lluvia
  - conflicto entre fuentes
  - elevacion
  - distancia a costa

### Bloque C - Interpretacion del error

- Relacionar error con:
  - `avamet_n_active`
  - `avamet_support_score_v0`
  - conflicto entre fuentes
  - distancia media a gauges usados en leave-cell
  - soporte efectivo del estimador local

### Bloque D - Sensibilidad secundaria

- Mantener `thinning` como analisis de apoyo:
  - no para vender un paper de densidad
  - si para ver como cambia la competitividad de productos EO cuando se degrada el soporte local

## Figuras principales

1. Esquema conceptual de escalas y soporte: EO product vs local gauge-assisted vs AVAMET target.
2. Ranking por regimen: donde cada producto EO es o no competitivo frente a leave-cell.
3. Figura de error vs soporte/conflicto.
4. Figura de limites en lluvia intensa y alto conflicto.
5. Figura de sensibilidad a degradacion de soporte in situ.

## Tablas principales

- Tabla de datasets y escalas
- Tabla de comparacion principal por producto y baseline local
- Tabla de regimenes de competitividad
- Tabla de lectura interpretativa de soporte y conflicto

## Rol del thinning en este paper

`Thinning` no es la contribucion principal.

Su rol es:

- probar la estabilidad de las conclusiones cuando el soporte in situ se degrada
- distinguir limites del producto EO frente a limites del benchmark local
- reforzar la interpretacion de representatividad

## Riesgos cientificos

- Que el paper derive otra vez en "IMERG validation".
- Que la comparacion se lea como benchmarking regional sin mensaje general.
- Que el bloque de thinning distraiga del mensaje EO principal.
- Que la interpretacion de soporte parezca demasiado indirecta si no se explica bien.

## Respuesta a esos riesgos

- Mantener la comparacion multifuente y no centrar la narrativa en IMERG.
- Formular cada resultado en lenguaje de utilidad operativa y mismatch de soporte.
- Dejar el thinning explicitamente como sensibilidad.
- Abrir el paper con la pregunta de utilidad de productos EO, no con la red AVAMET.

## Paquete minimo que debe quedar listo

- planning activo RSASE
- protocolo de analisis principal support-aware
- protocolo de thinning como sensibilidad
- tabla congelada de comparacion principal
- lista de figuras del manuscrito

## Siguiente decision operativa

La siguiente pieza que hay que producir no es otro protocolo de thinning.

La siguiente pieza es:

**`support_utility_protocol_v1.md`**

Ese documento debe fijar:

- comparaciones principales
- slices obligatorios
- como medir competitividad EO vs leave-cell
- y como se interpreta el papel del soporte observacional
