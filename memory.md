# Memoria del proyecto aivital-modelo

## Preferencias y correcciones
- El usuario quiere modelo pequeño primero (1 año, 1 comuna: La Florida, grilla < comuna), probabilidad suma de D+1 y D+2, tamaño, tipo y ubicación en grilla; precisión por causa; guiarse por la literatura y evitar errores comunes.
- Bases candidatas en `bases/` (catálogo del usuario). Papers en aivital-box `/aivital/GoogleDrive/space-apps-26-aivital/Papers/`.
- El "+40 %" que recuerda el usuario (texto verde en tabla de producto.html) no se encontró; lo más cercano es "hasta 30 %" de Di Giuseppe 2025.

## Pendientes por confirmar
- ¿Ampliar el área (RM) o conseguir etiquetas de bomberos para La Florida?

## Decisiones
- 20261114-2200 [rempc] datos copiados a `datos/` (80 MB) y venv `~/venv-incendios` | modelar offline con lightgbm | borrar carpeta y venv | backup: ninguno
- 20261114-2210 [rempc] Etapa 1 sobre RM y no solo La Florida | La Florida tiene ~5 eventos VIIRS en 8 años | cambiar CELL/área en src/modelo_rm.py | backup: ninguno
- 20261114-2330 [rempc] Adoptar src/modelo_rm_v2.py como modelo base | corrige fuga y lat/lon; clima por comuna sin señal | volver a src/modelo_rm.py | backup: ninguno
