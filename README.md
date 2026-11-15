# aivital-modelo

## Objetivo
Predecir incendios por celda de grilla dentro de una comuna (partiendo por La Florida, Santiago): probabilidad de al menos un incendio en D+1 o D+2, y después tamaño y tipo/causa. Hallazgos de papers y fuentes: `hallazgos.md`. Catálogo de bases: `bases/`.

## Hecho
- Papers leídos y verificados; causas de incendio en Chile (`hallazgos.md`).
- Datos locales en `datos/`: focos FIRMS (VIIRS y MODIS 2018-2025), clima diario por comuna 2012-2025 (de aivital-box), límites comunales, puntos CONAF >50 ha, dataset Geprif (solo Maule, 951 incendios con causa y tamaño) y resumen CONAF por comuna 1985-2024.
- Etapa 1, `src/modelo_rm.py` (Región Metropolitana, celdas de 0,02° ≈ 2 km, solo temporada sep-abr, etiqueta = foco VIIRS en D+1 o D+2): entrenado 2018-22, validado 2023, probado 2024-25. Resultados en `datos/proc/resultados_etapa1.csv`.
  - PR-AUC 0,008 contra prevalencia 0,0009 (≈9 veces la base); ROC-AUC 0,84; baseline de historia celda-mes 0,004 y 0,81. Con persistencia (foco en vecindad en 3 días): PR-AUC 0,019, ROC-AUC 0,88, pero mezcla incendios nuevos con en curso.
  - Solo clima (sin historia de la celda): no supera el azar de forma útil (se detiene a las 3 iteraciones). El clima viene por comuna, así que no distingue celdas.
  - Calibración razonable en los deciles altos (pred 0,0033, observado 0,0046). La prueba 2024-25 tuvo más incendios que el entrenamiento (0,095 % contra 0,064 %).

- Revisión estadística (`informe-revision.md`, código `src/modelo_rm_v2.py`): se confirmó fuga en la historia por celda y que lat/lon memorizan celdas. Mejor modelo sin clima (historia fuera de muestra + recencia de focos): PR-AUC 0,024 [0,015-0,039] vs 0,010 de v1, top 5 % de celdas captura 58 % de eventos nuevos. El clima por comuna no aporta señal demostrable. Calibración isotónica empeora (2024-25 tiene más fuego).

## Falta
- **La Florida sola no se puede modelar con estas etiquetas:** 9 detecciones VIIRS en 8 años (≈5 eventos) y CONAF registra 0-14 incendios por temporada (media ≈5, ~5 ha). Hay que decidir: modelar la RM y evaluar en La Florida, o conseguir otra etiqueta (bomberos).
- Variables que distinguen celdas: elevación y pendiente (Copernicus GLO-30), cobertura (ESA WorldCover), distancia a caminos y casas (OSM), población.
- Tamaño y tipo: FIRMS no trae causa ni superficie. Causa por incendio solo hay en Geprif (Maule, una temporada). Falta la fuente CONAF/SIDCO o el dataset MDPI por incendio en la RM.
- Meteorología por celda (ERA5-Land 9 km necesita cuenta CDS) y pronóstico de mañana.
- Separar incendios nuevos de los que siguen activos.
