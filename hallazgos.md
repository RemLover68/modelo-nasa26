# Hallazgos (2026-11-14): causas, tamaños y predictores

Citas verificadas por código con `verificar-informe-lector.py` salvo donde dice lo contrario.
Papers completos en aivital-box: `/aivital/GoogleDrive/space-apps-26-aivital/Papers/` (McNorton 2024, Kondylatos 2022, Dacre 2018, Di Giuseppe 2025, Illarionova 2025, Said 2026, Chen 2026).

## Causas en Chile
- Crespo-Pérez (2025) agrupa 4 causas: no intencional (faenas agrícolas/forestales, recreación, tránsito), intencional, natural (rayos, volcanes) y origen no identificado. Datos de causa hasta 2019.
- Frecuencia: accidentales > intencionales > naturales (muy pocos). Superficie afectada: no intencionales dominan, luego intencionales. Las naturales se concentran en los Andes (64,65 % del área natural 2009-2019; 45 % de las ocurrencias naturales).
- Tamaño medio por zona (unidad no dicha en el pasaje): centro 35,2, Andes 47,5, norte 4,52, sur 5,85.
- CONAF vía La Tercera: temporada 2022-23 25 % intencionales y 70 % negligencia; 2023-24 (927 siniestros investigados) 28 % intencional, 71-72 % accidental/negligencia. Rango histórico intencional 25-30 %. Solo ~40 % de los incendios se investiga. Los negligentes afectan más hectáreas (sin cifra).
- Cifras de producto.html P7 (99 % humano; 2.194 siniestros: 66 % accidental, 31 % intencional; 15 % quema de desechos, 10 % faenas agrícolas, 8 % fallas eléctricas) vienen de Meteored citando CONAF. No las abrí.
- Superficie quemada por causa en %: no apareció en ninguna fuente abierta. Falta la fuente primaria CONAF/SIDCO.

## Diferenciar por causa (Crespo-Pérez 2025, único paper que lo hace)
- Modelos separados intencional vs no intencional por zona. AUC norte 0,934 vs 0,911; centro 0,679 vs 0,670; Andes: no intencional mayor salvo el máximo (0,919 intencional). Centro ≈ 0,68, bajo 0,70.
- Distancia a líneas eléctricas: la más importante en casi todos los modelos (excepto sur); #1 para intencionales en norte y Andes y para no intencionales en el centro.
- Humedad del combustible muerto (DFMC): importancia moderada y muy variable; casi nula en el norte, de las más altas en el sur (sobre todo intencionales).
- Advertencia de los autores: en el centro la autocorrelación espacial puede inflar el rendimiento.

## Predictores (resto de papers)
- McNorton 2024 (leído por búsqueda de texto): punto de rocío a 2 m = variable individual más importante global. AUC: XGBoost 0,818, climatología mensual 0,810, RF 0,746. Menor en bosque de aguja (0,719).
- Di Giuseppe 2025: combustible = factor más importante; solo clima o solo ignición degrada ~30 %; combinar todo da hasta 30 % de mejora. Etiqueta = foco activo MODIS por celda y día (no separa causa). Contradicción interna: «150 %» en un pasaje vs «15 %» en otro para solo combustible. No citar esa cifra.
- Kondylatos 2022 (Grecia, solo abstract verificado + SHAP leído): humedad del suelo, humedad relativa, NDVI, punto de rocío y temperatura entre lo más importante; temperatura y lluvia acumuladas 10 días; HR de los 2 días previos; viento del último día.
- Gajardo 2025 (Chile, cicatrices de área quemada, sin causas): temperatura y precipitación de verano influyentes; probabilidad más alta en cultivos (0,70) y plantaciones (0,58).
- Illarionova 2025: es Rusia (4 óblast), no Chile; usa horizonte de 5 días y 7 días de meteorología. No hay ventanas de 12 y 45 días ni anomalía de NDVI. La fila de las capturas que la cita para eso está mal atribuida.
- Kettle 2023: no se pudo identificar (3 búsquedas).
- Dacre 2018: solo el abstract (accidental en la mayoría de los incendios). PDF local sin leer aún.

## Sin resolver
- El «40 %» de mejora (texto verde en tabla): no está en `producto.html`, `brechas-tecnicas-ml.html`, `variables-motor.html` ni en los 2 docs de `datos para modelamiento`. Lo más cercano es el 30 % de Di Giuseppe.
- Base con tamaño y causa por incendio: «Wildfire Occurrence and Damage Dataset for Chile (1985-2024)», Data 2025, 10(7), 93; GitHub `cvidalmsu/FONDEF-IDEA-Dataset`. Sin verificar por código: no se confirmó si es por incendio o por comuna, ni la licencia.
- Dato propio (variables-motor.html): FIRMS detectó 13,6 % de los incendios CONAF 2023-24, que sumaban 87,4 % de la superficie quemada.
