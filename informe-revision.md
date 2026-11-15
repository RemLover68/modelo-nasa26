# Informe de revisión del modelo de ocurrencia de focos (RM, 0,02°)

Código: `src/modelo_rm_v2.py` (pipeline, iteraciones), `src/diag_v2.py` (diagnósticos), `src/extra_v2.py` (experimentos solo en validación), `src/resumen_v2.py` (tabla).
Resultados: `datos/proc/v2_res_<iteración>.json`, `v2_extra.json`, `v2_pruebas_miradas.txt`. `src/modelo_rm.py` y `datos/` no se tocaron (solo se escribió en `datos/proc/`, 19 MB).

Método: validación por rolling origin (entrena años previos, evalúa 2021, 2022 y 2023) y prueba 2024-25 mirada una vez por iteración. IC 95 % por bootstrap de semanas (100 réplicas, negativos submuestreados 10 % con peso). Dos poblaciones: "todo" (todas las celdas-día, como v1) y "nuevo" (solo celdas sin foco en la vecindad 3x3 en D-2..D, es decir incendio nuevo). Con 2 focos de cada 3.000 celdas-día por año, todo número es ruidoso.

## 1. Falencias

| # | Sospecha | Veredicto | Evidencia |
|---|---|---|---|
| 1 | Fuga de `freq_celda_mes` | CONFIRMADA | AUC de la feature en las filas de entrenamiento: 0,920 (con su propia etiqueta) contra 0,782 leave-one-year-out y 0,798 en 2023 (honesto). Pero corregirla casi no cambia la métrica final: it1 contra it0 C en prueba, PR-AUC 0,0128 vs 0,0088 y ll_skill 0,123 vs 0,111, ambos dentro del IC (validación: ll_skill 0,126 vs 0,092, IC casi disjuntos, ligera mejora). |
| 2a | Quemas/eventos múltiples | CONFIRMADA | 3.074 celda-días VIIRS = 2.052 eventos (componentes conexas 8-vecinos, hueco 2 días). 77 % de los eventos es de 1 sola celda-día (mediana 1, p90 2, máx. 106). Sugiere detecciones aisladas (quemas, fuentes puntuales), no incendios grandes. |
| 2b | Persistencia contamina | CONFIRMADA | De 5.133 positivos (celda-día, temporada) 594 (11,6 %) son incendio en curso. Pero el predictor más fuerte para los nuevos es el conteo de focos en la propia celda en 30 días (n30, nb30): it0 C (sin persistencia) → it0 D: prueba "nuevo" PR-AUC 0,010 → 0,024, ll_skill 0,122 → 0,158. Es reincidencia espacial (misma celda quema de nuevo), no el incendio en curso. |
| 2c | UTC | DESCARTADA | Pasadas VIIRS a 04-06 y 17-19 UTC = 00:30-02:30 y 13:00-16:00 locales: la fecha UTC coincide con la local. |
| 2d | MODIS / confianza `l` | NO RESUELTA como mejora | Existen: VIIRS con `l` suma 3.492 celda-días (+14 %), MODIS (conf. >=30) 881. No entrené con esas etiquetas (decisión: no aportan sin causa/CONAF, ver más abajo). `type` 2 y 3 (otras fuentes estáticas/mar) ya estaban excluidos en v1 con `type==0`. |
| 3 | Cambio de distribución | CONFIRMADA | Focos por año 389, 352, 283, 280, 426, 402, 476, 466 (2018-25). Los IC por semanas son anchos (p. ej. PR-AUC de it0 D en prueba 0,0145 a 0,039). Diferencias entre variantes de PR-AUC menores a ~0,01 no se pueden distinguir. |
| 4 | Clima por comuna | CONFIRMADA y sin remedio con estos datos | Anomalías vs climatología del día del año (2012-17, sin fuga), VPD y KBDI simplificado no mejoran (it4 vs it3: prueba "nuevo" PR-AUC 0,0240 vs 0,0241, ll_skill 0,191 vs 0,192; en validación peor). |
| 5 | Métricas | CORREGIDA | Ahora se reportan PR-AUC, ROC-AUC, log-loss skill, Brier skill, captura en top 1 % y 5 %, y operativa (celdas altas por día) con IC. |
| 6 | Modelo | PARCIAL | LGBM regularizado + monotonía (it5): sin mejora fiable. Logística con interacciones: definida (`it5b_logistica`) pero no corrida (el criterio de parada se cumplió antes). Poisson por comuna-día: el clima EMPEORA (ver abajo). Horizontes D+1/D+2 separados: sin diferencia (IC cruzan 0). Grilla 0,06°: PR-AUC 0,048 (prevalencia 0,41 %), ROC 0,885, solo informativo (otra población, no comparable). |
| 7 | lat/lon memorizan celdas | CONFIRMADA | Quitar lat/lon bajó el rendimiento (it1 → it2: prueba "nuevo" PR-AUC 0,0158 → 0,0085, top1 0,182 → 0,127, operativo N=10 0,346 → 0,190 aciertos/día). lat/lon aportaba memoria celda a celda que el 5x5 suavizado borraba. La corrección fue dar la tasa de la propia celda (`h1`) y 3x3 (`h3`) calculada fuera de muestra: eso reemplaza a lat/lon (it3b) y mejora (ver tabla). Esto es historia por celda, no señal meteorológica. |

Hallazgo central: clima solo-historia vs historia+clima. it2b (solo historia) vs it2 (+ clima): prueba "nuevo" PR-AUC 0,0083 vs 0,0085, ll_skill 0,130 vs 0,127 (iguales). it3b vs it3 con recencia, igual. A nivel comuna-día, Poisson con clima vs solo tasa comuna-mes en validación: skill de deviance 0,234 con clima vs 0,247 sin clima; la mejora relativa de deviance es -1,8 % a -0,6 % (IC 95 %), o sea el clima sobreajusta. **El modelo no tiene señal meteorológica demostrable más allá de la historia por celda/mes y la recencia de focos cercanos.** Es un resultado válido (con la salvedad de que el clima es por comuna, de baja resolución).

## 2. Antes / después (prueba 2024-25, población "nuevo", IC 95 % entre corchetes; v1 = it0 C)

| Modelo | PR-AUC | ROC-AUC | ll_skill | top 1 % | top 5 % | aciertos/día en top 10 (azar 0,009) |
|---|---|---|---|---|---|---|
| it0 C: réplica v1 (fuga, lat/lon, clima) | 0,0104 [0,0068, 0,0222] | 0,867 | 0,122 [0,104, 0,133] | 0,162 | 0,386 | 0,269 |
| it0 D: v1 + persistencia | 0,0239 [0,0145, 0,0390] | 0,883 | 0,158 [0,136, 0,172] | 0,299 | 0,474 | 0,406 |
| it1: sin fuga (LOYO) | 0,0158 [0,0102, 0,0242] | 0,871 | 0,135 [0,120, 0,149] | 0,182 | 0,379 | 0,346 |
| it2: it1 sin lat/lon | 0,0085 [0,0059, 0,0126] | 0,874 | 0,127 [0,114, 0,137] | 0,127 | 0,370 | 0,190 |
| it2b: solo historia (sin lat/lon) | 0,0083 [0,0056, 0,0113] | 0,875 | 0,130 [0,117, 0,138] | 0,147 | 0,355 | 0,203 |
| it3b: nuevo, solo historia + h1/h3 + recencia | 0,0242 [0,0151, 0,0391] | 0,908 | 0,193 [0,171, 0,210] | 0,302 | 0,582 | 0,404 |
| it3: it3b + clima raw | 0,0241 [0,0166, 0,0316] | 0,908 | 0,192 [0,171, 0,210] | 0,309 | 0,579 | 0,395 |
| it4: + anomalía, VPD, KBDI | 0,0240 [0,0163, 0,0319] | 0,906 | 0,191 [0,170, 0,209] | 0,308 | 0,578 | 0,429 |
| it5: regularizado + monótono | 0,0279 [0,0178, 0,0386] | 0,910 | 0,202 [0,178, 0,220] | 0,326 | 0,571 | 0,451 |

Prevalencia en prueba 0,09 %. Mejora real, creíble: pasar de v1 (0,010) a it3b/it3 (0,024), con ROC 0,867 → 0,908 y ll_skill 0,122 → 0,193 (IC casi no solapan). Pero it0 D (v1 con n30) ya estaba en 0,024: la ganancia viene de recencia (n30/nb30) y tasa por celda, no de las correcciones de fuga ni de las features de clima. it3b, it3, it4 e it5 son indistinguibles entre sí (IC solapan por completo; en validación it3b ≥ it5). **No declaro it5 mejor.** El ll_skill de ~0,19 se explica casi por completo por qué celdas son de riesgo (la climatología espacial); es una mejora modesta sobre el azar espacial y no hay evidencia de predicción por tiempo meteorológico.

Notas: la calibración isotónica con 2023 empeora la prueba en it3b (Brier skill -0,028) porque 2024-25 tiene más fuego que 2023; el modelo sin calibrar es mejor allí. La captura "top 1 %" 0,30 significa que el 1 % de celdas-día más altas contiene 30 % de los eventos nuevos.

La Florida en prueba (2024-25): 9.660 filas, 2 positivos y ~4,6 esperados por el modelo (it3b). Con n=2 no se puede evaluar; la predicción de volumen total es compatible pero no verificable.

## 3. Cambios de código
`src/modelo_rm_v2.py` (nuevo, no reemplaza a v1): caché de arrays, historia leave-one-year-out (`hist_feats`), población "nuevo", features de recencia, climatología sin fuga, VPD y KBDI, rolling origin, bootstrap por bloques semanales pareable, métricas (log-loss, Brier skill, top-k, operativa), conteo de miradas a la prueba. `src/extra_v2.py` (horizontes, Poisson comuna-día, grilla 0,06). `src/diag_v2.py` (eventos, fuga, UTC).

## 4. Miradas a la partición de prueba
**9** (una por iteración corrida: it0 C, it0 D, it1, it2, it2b, it3b, it3, it4, it5; ver `datos/proc/v2_pruebas_miradas.txt`). Excede las 8 previstas porque las iteraciones 0 y 2 llevaron una variante de control. Las decisiones se tomaron con validación; la prueba no se usó para elegir parámetros ni features. `extra_v2.py` no mira la prueba. Early stopping en el rolling usa el año de validación mismo (optimismo leve, igual para todas las variantes).

## 5. Sin resolver o sin poder verificar
- Parada: it4 e it5 no mejoran más allá del IC sobre it3 (criterio de 2 iteraciones seguidas), por eso no se corrieron etiquetas MODIS/`l`, la logística con interacciones ni un GAM.
- No verificable con estos datos: separar quemas agrícolas de incendios (falta cobertura/uso de suelo y causa CONAF/SIDCO); evaluar a nivel de evento real con superficie.
- Faltan datos para mejorar el clima: meteorología por celda (ERA5-Land/ pronóstico D+1), humedad del combustible real, vegetación (NDVI), cobertura y distancias (OSM, caminos, línea eléctrica).
- La Florida: n=2 en prueba y ~5 eventos en 8 años; no evaluable. Falta etiqueta de bomberos o CONAF por incendio.
- El KBDI es una aproximación mía a partir de tmax/precipitación diaria por comuna, no validada contra un índice oficial.
- Con ~1.800 eventos en 8 años y casi la mitad de ruido de detección aislada, el límite probable es la etiqueta, no el modelo.
