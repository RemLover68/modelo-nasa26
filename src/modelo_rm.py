"""Etapa 1: ocurrencia de focos en grilla de 0,02 grados (~2 km) en la Región Metropolitana.

Objetivo: P(al menos un foco VIIRS en la celda en D+1 o D+2), con la información disponible al día D.
Por qué la RM y no solo La Florida: La Florida tiene 9 detecciones VIIRS en 8 años (ver README).
Validación temporal: entrena 2018-2022, valida 2023, prueba 2024-2025 (con 2 días de separación).
Uso: ~/venv-incendios/bin/python src/modelo_rm.py
"""
import json, glob, sys
import numpy as np, pandas as pd
from shapely.geometry import shape, Point
from shapely.prepared import prep
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb

R = 'datos/'
CELL = 0.02
LAT0, LAT1, LON0, LON1 = -34.35, -32.85, -71.75, -69.70
H, W = int(round((LAT1 - LAT0) / CELL)), int(round((LON1 - LON0) / CELL))
DAYS = pd.date_range('2018-01-01', '2025-12-31')
ND = len(DAYS)

# --- comunas de la RM y máscara de celdas
g = json.load(open(R + 'conaf/Comunas__0_comunas_web.geojson'))
rm = [f for f in g['features'] if f['properties']['reg_sidco'].startswith('13')]
codes = sorted({f['properties']['dpachile_i'] for f in rm})
cell_com = -np.ones((H, W), dtype=int)
prep_geoms = [(prep(shape(f['geometry']).buffer(0)), codes.index(f['properties']['dpachile_i'])) for f in rm]
for iy in range(H):
    for ix in range(W):
        p = Point(LON0 + (ix + .5) * CELL, LAT0 + (iy + .5) * CELL)
        for pg, ci in prep_geoms:
            if pg.contains(p):
                cell_com[iy, ix] = ci; break
mask = cell_com >= 0
print('celdas en RM:', mask.sum(), 'de', H * W)
lf_code = codes.index('13110')
print('celdas con centro en La Florida:', (cell_com == lf_code).sum())

# --- focos VIIRS (vegetación, confianza nominal/alta) -> array días x celdas
det = np.zeros((ND, H, W), dtype=np.uint8)
n_raw = 0
for f in sorted(glob.glob(R + 'firms/viirs-snpp_*.csv')):
    d = pd.read_csv(f, dtype={'acq_time': str, 'confidence': str}, usecols=['latitude', 'longitude', 'acq_date', 'confidence', 'type'])
    d = d[(d.type == 0) & (d.confidence.isin(['n', 'h'])) & d.latitude.between(LAT0, LAT1) & d.longitude.between(LON0, LON1)]
    iy = ((d.latitude - LAT0) / CELL).astype(int).clip(0, H - 1).values
    ix = ((d.longitude - LON0) / CELL).astype(int).clip(0, W - 1).values
    di = (pd.to_datetime(d.acq_date) - DAYS[0]).dt.days.values
    ok = (di >= 0) & (di < ND) & mask[iy, ix]
    det[di[ok], iy[ok], ix[ok]] = 1; n_raw += ok.sum()
print('detecciones dentro de la RM (con confianza n/h):', n_raw, '| celda-días con foco:', int(det.sum()))

# --- clima por comuna -> [día, comuna]
c = pd.read_parquet(R + 'clima/clima_diario.parquet')
c = c[c.region_id.isin(codes)]
cols = ['tmax', 'hrmin', 'hr_media', 'viento_max', 'precip']
cl = {}
for v in cols:
    p = c.pivot(index='fecha_local', columns='region_id', values=v).reindex(index=DAYS, columns=codes)
    cl[v] = p.values.astype(np.float32)  # [ND, ncom]
ncom = len(codes)

def roll_sum(a, k):
    cs = np.cumsum(np.vstack([np.zeros((1, a.shape[1])), a]), axis=0)
    out = np.full_like(a, np.nan, dtype=np.float32)
    out[k - 1:] = (cs[k:] - cs[:-k])
    return out
def roll_mean(a, k): return roll_sum(a, k) / k
def lag(a, k):
    out = np.full_like(a, np.nan); out[k:] = a[:-k]; return out
def dias_sin_lluvia(p):
    out = np.zeros_like(p);
    for t in range(1, len(p)): out[t] = np.where(p[t] > 1.0, 0, np.minimum(out[t - 1] + 1, 60))
    return out

feat_com = {
    'tmax': cl['tmax'], 'hrmin': cl['hrmin'], 'hr_media': cl['hr_media'], 'viento_max': cl['viento_max'], 'precip': cl['precip'],
    'tmax_l1': lag(cl['tmax'], 1), 'hrmin_l1': lag(cl['hrmin'], 1),
    'tmax_m7': roll_mean(cl['tmax'], 7), 'hrmin_m7': roll_mean(cl['hrmin'], 7),
    'precip_s7': roll_sum(cl['precip'], 7), 'precip_s30': roll_sum(cl['precip'], 30), 'precip_s90': roll_sum(cl['precip'], 90),
    'dias_sin_lluvia': dias_sin_lluvia(cl['precip']),
}

# --- persistencia: foco en vecindad 3x3 en los últimos 3 días (hasta D)
def dilate3(a):
    o = a.copy()
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            s = np.zeros_like(a)
            ys, yd = (slice(max(dy, 0), H + min(dy, 0)), slice(max(-dy, 0), H + min(-dy, 0)))
            xs, xd = (slice(max(dx, 0), W + min(dx, 0)), slice(max(-dx, 0), W + min(-dx, 0)))
            s[:, yd, xd] = a[:, ys, xs]; o = np.maximum(o, s)
    return o
d3 = dilate3(det)
pers3 = np.zeros_like(d3)
for k in range(3): pers3[k:] = np.maximum(pers3[k:], d3[:ND - k])
d30 = np.zeros((ND, H, W), dtype=np.float32)
cs = np.cumsum(det, axis=0, dtype=np.int32)
d30[30:] = cs[30:] - cs[:-30]

# --- objetivo: foco en D+1 o D+2
y_all = np.zeros_like(det); y_all[:-2] = np.maximum(det[1:-1], det[2:])

# --- historia por celda y mes (solo años de entrenamiento: 2018-2022), suavizada 5x5
train_mask_days = (DAYS.year <= 2022)
month = DAYS.month.values
freq = np.zeros((13, H, W), dtype=np.float32)
for m in range(1, 13):
    sel = train_mask_days & (month == m)
    if sel.sum(): freq[m] = y_all[sel].mean(axis=0)
def smooth5(a):
    o = np.zeros_like(a, dtype=np.float32); n = 0
    for dy in range(-2, 3):
        for dx in range(-2, 3):
            s = np.zeros_like(a, dtype=np.float32)
            ys, yd = (slice(max(dy, 0), H + min(dy, 0)), slice(max(-dy, 0), H + min(-dy, 0)))
            xs, xd = (slice(max(dx, 0), W + min(dx, 0)), slice(max(-dx, 0), W + min(-dx, 0)))
            s[..., yd, xd] = a[..., ys, xs]; o += s; n += 1
    return o / n
freq_s = smooth5(freq)

# --- tabla larga: días de temporada (sep-abr) x celdas de la RM
season = np.isin(month, [9, 10, 11, 12, 1, 2, 3, 4])
valid_d = np.where(season & (np.arange(ND) >= 90) & (np.arange(ND) < ND - 2))[0]
iy, ix = np.where(mask)
ci = cell_com[iy, ix]
rows = {}
n_cells = len(iy)
dd = np.repeat(valid_d, n_cells); cc = np.tile(np.arange(n_cells), len(valid_d))
for k, a in feat_com.items(): rows[k] = a[dd, ci[cc]]
rows['mes'] = month[dd]; rows['doy'] = DAYS.dayofyear.values[dd]; rows['dow'] = DAYS.dayofweek.values[dd]
rows['lat'] = (LAT0 + (iy[cc] + .5) * CELL).astype(np.float32); rows['lon'] = (LON0 + (ix[cc] + .5) * CELL).astype(np.float32)
rows['freq_celda_mes'] = freq_s[month[dd], iy[cc], ix[cc]]
rows['pers3'] = pers3[dd, iy[cc], ix[cc]]; rows['n30'] = d30[dd, iy[cc], ix[cc]]
X = pd.DataFrame(rows)
y = y_all[dd, iy[cc], ix[cc]]
fecha = DAYS[dd]; es_lf = (ci[cc] == lf_code)
print('filas:', len(X), '| positivos:', int(y.sum()), f'({100*y.mean():.3f} %)')

# --- particiones temporales con 2 días de separación
def part(a, b):  # fechas [a, b)
    return (fecha >= a) & (fecha < b)
tr = part('2018-04-01', '2022-12-29'); va = part('2023-01-01', '2023-12-29'); te = part('2024-01-01', '2026-01-01')
print('train/val/test filas:', tr.sum(), va.sum(), te.sum(), '| positivos:', int(y[tr].sum()), int(y[va].sum()), int(y[te].sum()))

def metrics(name, p, yy):
    base = yy.mean()
    return dict(modelo=name, pr_auc=average_precision_score(yy, p), roc_auc=roc_auc_score(yy, p), brier=brier_score_loss(yy, p),
                brier_skill_vs_prevalencia=1 - brier_score_loss(yy, p) / brier_score_loss(yy, np.full_like(p, base, dtype=float)),
                top1pct_captura=float(yy[np.argsort(-p)[:max(1, int(.01 * len(p)))]].sum() / max(1, yy.sum())),
                prevalencia=base)

res = []
sets = {
    'A_historia_celda_mes': ['freq_celda_mes'],
    'B_clima': ['tmax', 'hrmin', 'hr_media', 'viento_max', 'precip', 'tmax_l1', 'hrmin_l1', 'tmax_m7', 'hrmin_m7', 'precip_s7', 'precip_s30', 'precip_s90', 'dias_sin_lluvia', 'mes', 'doy', 'dow'],
}
sets['C_clima+historia+ubicacion'] = sets['B_clima'] + ['freq_celda_mes', 'lat', 'lon']
sets['D_C+persistencia'] = sets['C_clima+historia+ubicacion'] + ['pers3', 'n30']
res.append(metrics('baseline: climatologia celda-mes (sin entrenar)', X.loc[te, 'freq_celda_mes'].values.astype(float) + 1e-9, y[te]))
preds = {}
for name, cols_ in sets.items():
    if name.startswith('A'): continue
    m = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.03, num_leaves=15, min_child_samples=200, subsample=0.8, subsample_freq=1,
                           colsample_bytree=0.8, reg_lambda=5.0, verbose=-1, random_state=0)
    m.fit(X.loc[tr, cols_], y[tr], eval_set=[(X.loc[va, cols_], y[va])], eval_metric='average_precision',
          callbacks=[lgb.early_stopping(40, verbose=False)])
    pv = m.predict_proba(X.loc[va, cols_])[:, 1]
    iso = IsotonicRegression(out_of_bounds='clip', y_min=1e-7).fit(pv, y[va])   # calibración con validación
    pt = m.predict_proba(X.loc[te, cols_])[:, 1]; ptc = iso.predict(pt)
    r = metrics(f'LGBM {name} (sin calibrar)', pt, y[te]); r['iter'] = m.best_iteration_; res.append(r)
    res.append(metrics(f'LGBM {name} (calibrado isotonica)', ptc + 1e-9, y[te]))
    preds[name] = (ptc, m, cols_)
# regresión logística solo clima como referencia lineal
sc = StandardScaler().fit(X.loc[tr, sets['B_clima']].fillna(0))
lr = LogisticRegression(C=0.1, max_iter=300).fit(sc.transform(X.loc[tr, sets['B_clima']].fillna(0)), y[tr])
res.append(metrics('logistica clima', lr.predict_proba(sc.transform(X.loc[te, sets['B_clima']].fillna(0)))[:, 1], y[te]))
R_ = pd.DataFrame(res); pd.set_option('display.width', 250); pd.set_option('display.max_colwidth', 60)
print(R_.round(4).to_string())
R_.to_csv('datos/proc/resultados_etapa1.csv', index=False)

# calibración (modelo C calibrado) en prueba
ptc = preds['C_clima+historia+ubicacion'][0]
q = pd.qcut(pd.Series(ptc).rank(method='first'), 10, labels=False)
cal = pd.DataFrame({'p_pred': ptc, 'y': y[te], 'dec': q}).groupby('dec').agg(p_pred=('p_pred', 'mean'), freq_obs=('y', 'mean'), n=('y', 'size'))
print('\nCalibración por decil (modelo C):'); print(cal.to_string())
# importancia
m, cols_ = preds['C_clima+historia+ubicacion'][1], preds['C_clima+historia+ubicacion'][2]
imp = pd.Series(m.booster_.feature_importance('gain'), index=cols_).sort_values(ascending=False); print('\nImportancia (ganancia):'); print((imp / imp.sum()).round(3).head(10).to_string())
# La Florida en prueba
lf_te = te & es_lf
print('\nLa Florida en prueba: filas', int(lf_te.sum()), '| positivos', int(y[lf_te].sum()), '| p media predicha', float(preds['C_clima+historia+ubicacion'][0][es_lf[te]].mean()))
