"""Experimentos extra solo en VALIDACION (rolling origin 2021-23); no tocan la prueba."""
import sys; sys.path.insert(0, 'src')
import numpy as np, pandas as pd, lightgbm as lgb, json
from modelo_rm_v2 import *
D = Data(); cfg = dict(CFG['it3b_nuevo_solo_hist']); out = {}
yflat12 = D.cells(D.y('v', '12')).ravel(); dayflat = np.repeat(np.arange(ND), D.nc)
npop = (D.cells(D.pers3) == 0).ravel()
inv = np.isin(YEAR[dayflat], [2021, 2022, 2023]) & D.season[dayflat] & (dayflat >= 90) & (dayflat < ND - 2)
# (a) horizontes separados vs suma (misma poblacion 'nuevo', etiqueta de cada horizonte)
base, _ = rolling(D, cfg); base = np.nan_to_num(base, nan=1e-6)
for hz in ('1', '2'):
    p, _ = rolling(D, dict(cfg, hz=hz)); p = np.nan_to_num(p, nan=1e-6)
    yh = D.cells(D.y('v', hz)).ravel()
    e = Eval(yh, dayflat, inv & npop, seed=0)
    m_sep = e.point(p); m_sum = e.point(base)   # 'suma' = modelo D+1|D+2 evaluado contra el horizonte hz (se espera mayor prob.; el ranking es lo comparable)
    out[f'hz{hz}'] = dict(separado={k: round(float(v), 4) for k, v in m_sep.items()}, modelo_suma={k: round(float(v), 4) for k, v in m_sum.items()}, ci_dif=e.boot(p, 60, ref=base, keys=('pr_auc', 'roc_auc', 'top1pct', 'top5pct'))['dif'])
    print('horizonte', hz, out[f'hz{hz}']['separado']['pr_auc'], out[f'hz{hz}']['modelo_suma']['pr_auc'], out[f'hz{hz}']['ci_dif'], flush=True)
# (b) comuna-dia: conteo de celdas con evento nuevo; Poisson con y sin clima (rolling por anio)
ynew = (D.cells(D.y('v', '12')) * (D.cells(D.pers3) == 0)).astype(np.float32)      # [ND,nc]
ncom = len(D.z['codes']); cnt = np.zeros((ND, ncom), np.float32); np.add.at(cnt.T, D.ci, ynew.T) if False else None
for c in range(ncom): cnt[:, c] = ynew[:, D.ci == c].sum(1)
popc = np.array([(D.ci == c).sum() for c in range(ncom)])
rows = D.valid_d
def feats(days, tr_years, with_clim):
    X = {}
    # tasa comuna-mes LOYO (promedio de conteo por celda)
    rate = np.zeros((len(days), ncom), np.float32); yd = YEAR[days]
    for y_ in sorted(set(yd)):
        yrs = [t for t in tr_years if t != y_] if y_ in tr_years else tr_years
        for m in range(1, 13):
            sel = np.isin(YEAR, yrs) & (MONTH == m) & D.season
            r = cnt[sel].mean(0) if sel.sum() else np.zeros(ncom)
            rate[(yd == y_) & (MONTH[days] == m)] = r
    X['crate'] = rate.ravel(); X['npop'] = np.tile(popc, len(days)).astype(np.float32)
    X['mes'] = np.repeat(MONTH[days], ncom).astype(np.float32)
    if with_clim:
        for k in CLIM_RAW + CLIM_NEW: X[k] = D.clim[k][days].ravel()
    return pd.DataFrame(X)
def dev(y, mu):
    mu = np.clip(mu, 1e-9, None); return 2 * np.sum(np.where(y > 0, y * np.log(y / mu), 0) - (y - mu))
res = {}
for with_clim in (False, True):
    ps = []; ys = []; dd = []
    for vy, tr in ((2021, [2018, 2019, 2020]), (2022, [2018, 2019, 2020, 2021]), (2023, [2018, 2019, 2020, 2021, 2022])):
        dtr = rows[np.isin(YEAR[rows], tr)]; dva = rows[YEAR[rows] == vy]
        Xtr = feats(dtr, tr, with_clim); ytr = cnt[dtr].ravel(); Xva = feats(dva, tr, with_clim); yva = cnt[dva].ravel()
        m = lgb.LGBMRegressor(objective='poisson', n_estimators=500, learning_rate=0.03, num_leaves=7, min_child_samples=300, reg_lambda=20, colsample_bytree=0.7, subsample=0.8, subsample_freq=1, verbose=-1, n_jobs=10)
        m.fit(Xtr, ytr, eval_set=[(Xva, yva)], callbacks=[lgb.early_stopping(50, verbose=False)])
        ps.append(m.predict(Xva)); ys.append(yva); dd.append(np.repeat(dva, ncom))
    p = np.concatenate(ps); y_ = np.concatenate(ys); d_ = np.concatenate(dd)
    res[with_clim] = (p, y_, d_)
p0, y0, d0 = res[False]; p1, _, _ = res[True]
wk = d0 // 7; rng = np.random.default_rng(0); weeks = np.unique(wk)
idx = {w: np.where(wk == w)[0] for w in weeks}
def skill(p, sel): return 1 - dev(y0[sel], p[sel]) / dev(y0[sel], np.full(sel.sum(), y0[sel].mean()))
diffs = []
for _ in range(200):
    sel = np.concatenate([idx[w] for w in rng.choice(weeks, len(weeks))])
    diffs.append((dev(y0[sel], p0[sel]) - dev(y0[sel], p1[sel])) / dev(y0[sel], np.full(len(sel), y0[sel].mean())))
sel = np.ones(len(y0), bool)
out['poisson_comuna_dia'] = dict(skill_solo_historia=round(float(skill(p0, sel)), 4), skill_con_clima=round(float(skill(p1, sel)), 4),
                                  mejora_deviance_rel_ic95=[round(float(np.percentile(diffs, 2.5)), 4), round(float(np.percentile(diffs, 97.5)), 4)], filas=int(len(y0)), eventos=int(y0.sum()))
print(out['poisson_comuna_dia'], flush=True)
# (c) grilla gruesa 0.06 (bloques 3x3) con el modelo de celda (solo historia+recencia): p_bloque = 1 - prod(1-p_celda)
by = np.arange(H) // 3; bx = np.arange(W) // 3; nbx = bx.max() + 1
blk = (by[D.iy] * nbx + bx[D.ix]); ub, binv = np.unique(blk, return_inverse=True)
sel_days = D.valid_d[np.isin(YEAR[D.valid_d], [2021, 2022, 2023])]
P_ = base.reshape(ND, D.nc)[sel_days]; Y_ = yflat12.reshape(ND, D.nc)[sel_days]; N_ = npop.reshape(ND, D.nc)[sel_days]
lp = np.zeros((len(sel_days), len(ub))); yb = np.zeros_like(lp); nb = np.zeros_like(lp)
logq = np.log1p(-np.clip(P_ * N_, 0, 0.999))
for j in range(len(ub)):
    c = binv == j; lp[:, j] = logq[:, c].sum(1); yb[:, j] = (Y_[:, c] * N_[:, c]).max(1); nb[:, j] = N_[:, c].any(1)
pb = 1 - np.exp(lp); m = nb.astype(bool)
out['grilla_0.06'] = dict(prevalencia=float(yb[m].mean()), pr_auc=float(average_precision_score(yb[m], pb[m])), roc_auc=float(roc_auc_score(yb[m], pb[m])), n_pos=int(yb[m].sum()))
print(out['grilla_0.06'], flush=True)
json.dump(out, open('datos/proc/v2_extra.json', 'w'), indent=1, default=float)
