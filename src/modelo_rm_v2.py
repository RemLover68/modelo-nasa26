"""v2: revision critica del modelo de ocurrencia de focos (RM, celdas 0,02 grados).

Fases (python src/modelo_rm_v2.py <fase>):
  prep   : construye arrays (focos VIIRS/MODIS por dia-celda, clima por comuna) y los deja en datos/proc/v2_cache.npz
  diag   : diagnosticos de etiqueta/fuga/eventos (sin tocar la prueba)
  iter   : corre las iteraciones 0..N (rolling origin en validacion + UNA mirada a la prueba 2024-25 por iteracion)
Este archivo no modifica modelo_rm.py ni datos/ (solo escribe en datos/proc/).
"""
import json, glob, sys, os, time
import numpy as np, pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, log_loss

R = 'datos/'
P = 'datos/proc/'
CELL = 0.02
LAT0, LAT1, LON0, LON1 = -34.35, -32.85, -71.75, -69.70
H, W = int(round((LAT1 - LAT0) / CELL)), int(round((LON1 - LON0) / CELL))
DAYS = pd.date_range('2018-01-01', '2025-12-31')
ND = len(DAYS)
YEAR = DAYS.year.values
MONTH = DAYS.month.values


# ---------------------------------------------------------------- prep
def prep():
    from shapely.geometry import shape, Point
    from shapely.prepared import prep as sprep
    g = json.load(open(R + 'conaf/Comunas__0_comunas_web.geojson'))
    rm = [f for f in g['features'] if f['properties']['reg_sidco'].startswith('13')]
    codes = sorted({f['properties']['dpachile_i'] for f in rm})
    cell_com = -np.ones((H, W), dtype=np.int16)
    pg = [(sprep(shape(f['geometry']).buffer(0)), codes.index(f['properties']['dpachile_i'])) for f in rm]
    for iy in range(H):
        for ix in range(W):
            p = Point(LON0 + (ix + .5) * CELL, LAT0 + (iy + .5) * CELL)
            for g_, ci in pg:
                if g_.contains(p):
                    cell_com[iy, ix] = ci; break
    mask = cell_com >= 0
    out = {'cell_com': cell_com, 'codes': np.array(codes)}

    def grid(files, keep, conf_ok):
        a = np.zeros((ND, H, W), dtype=np.uint8)
        for f in sorted(files):
            d = pd.read_csv(f, dtype={'confidence': str}, usecols=['latitude', 'longitude', 'acq_date', 'confidence', 'type'])
            d = d[(d.type == 0) & d.confidence.isin(conf_ok) & d.latitude.between(LAT0, LAT1) & d.longitude.between(LON0, LON1)]
            iy = ((d.latitude - LAT0) / CELL).astype(int).clip(0, H - 1).values
            ix = ((d.longitude - LON0) / CELL).astype(int).clip(0, W - 1).values
            di = (pd.to_datetime(d.acq_date) - DAYS[0]).dt.days.values
            ok = (di >= 0) & (di < ND) & mask[iy, ix]
            a[di[ok], iy[ok], ix[ok]] = 1
        return a
    out['det_v'] = grid(glob.glob(R + 'firms/viirs-snpp_*.csv'), None, ['n', 'h'])          # igual que v1
    out['det_vl'] = grid(glob.glob(R + 'firms/viirs-snpp_*.csv'), None, ['l', 'n', 'h'])    # + confianza baja
    # MODIS: confianza numerica 0-100; se usa >=30 (nominal+)
    a = np.zeros((ND, H, W), dtype=np.uint8)
    for f in sorted(glob.glob(R + 'firms/modis_*.csv')):
        d = pd.read_csv(f, usecols=['latitude', 'longitude', 'acq_date', 'confidence', 'type'])
        d = d[(d.type == 0) & (d.confidence >= 30) & d.latitude.between(LAT0, LAT1) & d.longitude.between(LON0, LON1)]
        iy = ((d.latitude - LAT0) / CELL).astype(int).clip(0, H - 1).values
        ix = ((d.longitude - LON0) / CELL).astype(int).clip(0, W - 1).values
        di = (pd.to_datetime(d.acq_date) - DAYS[0]).dt.days.values
        ok = (di >= 0) & (di < ND) & mask[iy, ix]
        a[di[ok], iy[ok], ix[ok]] = 1
    out['det_m'] = a
    c = pd.read_parquet(R + 'clima/clima_diario.parquet')
    c = c[c.region_id.isin(codes)]
    days_full = pd.date_range('2012-01-01', '2025-12-31')
    for v in ['tmax', 'hrmin', 'hr_media', 'viento_max', 'precip']:
        out['cl_' + v] = c.pivot(index='fecha_local', columns='region_id', values=v).reindex(index=days_full, columns=codes).values.astype(np.float32)
    np.savez_compressed(P + 'v2_cache.npz', **out)
    print('cache ok', {k: v.shape for k, v in out.items()})


def load():
    z = np.load(P + 'v2_cache.npz')
    return {k: z[k] for k in z.files}


# ---------------------------------------------------------------- utilidades de grilla
def shift_max(a, dy, dx):
    """a[..., H, W] desplazado (dy,dx), relleno 0 (para dilatacion)."""
    s = np.zeros_like(a)
    Hh, Ww = a.shape[-2:]
    ys, yd = slice(max(dy, 0), Hh + min(dy, 0)), slice(max(-dy, 0), Hh + min(-dy, 0))
    xs, xd = slice(max(dx, 0), Ww + min(dx, 0)), slice(max(-dx, 0), Ww + min(-dx, 0))
    s[..., yd, xd] = a[..., ys, xs]
    return s


def dilate(a, r=1):
    o = a.copy()
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            o = np.maximum(o, shift_max(a, dy, dx))
    return o


def box_mean(a, r):
    o = np.zeros(a.shape, dtype=np.float32)
    n = 0
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            o += shift_max(a.astype(np.float32), dy, dx); n += 1
    return o / n


def recent_any(a, k):
    """a[t] = 1 si hubo foco en t-k+1..t (inclusive)."""
    o = np.zeros_like(a)
    for j in range(k):
        o[j:] = np.maximum(o[j:], a[:ND - j])
    return o


# ---------------------------------------------------------------- metricas
def wmetrics(p, y, w=None, topfr=(0.01, 0.05)):
    """Metricas con pesos (los negativos submuestreados llevan peso 1/frac)."""
    if w is None: w = np.ones(len(y))
    y = y.astype(float)
    base = (w * y).sum() / w.sum()
    pc = np.clip(p, 1e-7, 1 - 1e-7)
    ll = -(w * (y * np.log(pc) + (1 - y) * np.log(1 - pc))).sum() / w.sum()
    ll0 = -(base * np.log(base) + (1 - base) * np.log(1 - base))
    br = (w * (pc - y) ** 2).sum() / w.sum()
    br0 = base * (1 - base)
    o = np.argsort(-p, kind='stable'); wc = np.cumsum(w[o]); tot = wc[-1]
    wy = np.cumsum((w * y)[o])
    r = dict(pr_auc=average_precision_score(y, p, sample_weight=w), roc_auc=roc_auc_score(y, p, sample_weight=w),
             logloss=ll, ll_skill=1 - ll / ll0, brier_skill=1 - br / br0, prev=base)
    for f in topfr:
        i = np.searchsorted(wc, f * tot)
        r[f'top{int(f*100)}pct'] = wy[min(i, len(wy) - 1)] / wy[-1]
    return r


class Eval:
    """Evalua en un conjunto: poblacion (mask), submuestrea negativos para el bootstrap por bloques semanales."""
    def __init__(self, y, day, pop, neg_frac=0.1, seed=0):
        rng = np.random.default_rng(seed)
        idx = np.where(pop)[0]
        keep = (y[idx] == 1) | (rng.random(len(idx)) < neg_frac)
        self.idx = idx[keep]
        self.w = np.where(y[self.idx] == 1, 1.0, 1.0 / neg_frac)
        self.y = y[self.idx]; self.day = day[self.idx]
        wk = (self.day // 7)
        self.weeks = np.unique(wk)
        order = np.argsort(wk, kind='stable'); self.ord = order
        ws = wk[order]
        self.bounds = {w_: (np.searchsorted(ws, w_, 'left'), np.searchsorted(ws, w_, 'right')) for w_ in self.weeks}

    def point(self, p_full):
        return wmetrics(p_full[self.idx], self.y, self.w)

    def boot(self, p_full, n=100, seed=1, keys=('pr_auc', 'roc_auc', 'll_skill', 'brier_skill', 'top1pct', 'top5pct'), ref=None):
        """IC95 por bootstrap de semanas. Si ref (otro p_full) se da, tambien IC de la diferencia pareada."""
        rng = np.random.default_rng(seed)
        p = p_full[self.idx]; pr = ref[self.idx] if ref is not None else None
        res, dif = [], []
        for _ in range(n):
            ws = rng.choice(self.weeks, len(self.weeks))
            sel = np.concatenate([self.ord[self.bounds[w_][0]:self.bounds[w_][1]] for w_ in ws])
            if self.y[sel].sum() < 5: continue
            m = wmetrics(p[sel], self.y[sel], self.w[sel]); res.append([m[k] for k in keys])
            if pr is not None:
                m2 = wmetrics(pr[sel], self.y[sel], self.w[sel]); dif.append([m[k] - m2[k] for k in keys])
        res = np.array(res)
        out = {k: (np.percentile(res[:, i], 2.5), np.percentile(res[:, i], 97.5)) for i, k in enumerate(keys)}
        if pr is not None:
            dif = np.array(dif)
            out['dif'] = {k: (np.percentile(dif[:, i], 2.5), np.percentile(dif[:, i], 97.5)) for i, k in enumerate(keys)}
        return out


def op_metric(p_full, y, day, pop, N=10):
    """Operativa: cada dia las N celdas mas altas (dentro de la poblacion en riesgo). Devuelve aciertos/dia, precision@N,
    fraccion de eventos capturados y el esperado al azar."""
    d = np.unique(day[pop]); hits = tot = 0; rand = 0.0; nd = 0
    for t in d:
        s = np.where(pop & (day == t))[0]
        if len(s) < N: continue
        yy = y[s]; pp = p_full[s]
        top = np.argpartition(-pp, N - 1)[:N]
        hits += yy[top].sum(); tot += yy.sum(); rand += N * yy.mean(); nd += 1
    return dict(aciertos_dia=hits / nd, prec_at_N=hits / (nd * N), recall=hits / max(tot, 1), azar_aciertos_dia=rand / nd, N=N)

if __name__ == '__main__' and len(sys.argv) > 1 and sys.argv[1] == 'prep':
    prep()


# ---------------------------------------------------------------- features
OFF = 2192  # dias entre 2012-01-01 y 2018-01-01


def roll(a, k, fn='sum'):
    cs = np.cumsum(np.vstack([np.zeros((1, a.shape[1])), np.nan_to_num(a)]), axis=0)
    out = np.full(a.shape, np.nan, dtype=np.float32); out[k - 1:] = cs[k:] - cs[:-k]
    return out / k if fn == 'mean' else out


def lagf(a, k):
    o = np.full(a.shape, np.nan, dtype=np.float32); o[k:] = a[:-k]; return o


def climate_feats(z):
    """Features de clima por comuna, [ND, ncom]. Anomalias contra climatologia del dia del ano 2012-2017 (sin fuga)."""
    cl = {v: z['cl_' + v] for v in ['tmax', 'hrmin', 'hr_media', 'viento_max', 'precip']}
    full = pd.date_range('2012-01-01', '2025-12-31'); doy = full.dayofyear.values
    f = {}
    for v in ['tmax', 'hrmin', 'viento_max']:
        base = cl[v][:OFF]; d0 = doy[:OFF]
        clim = np.zeros((367, base.shape[1]), np.float32)
        for d in range(1, 367):
            s = np.abs(((d0 - d + 183) % 366) - 183) <= 7
            clim[d] = np.nanmean(base[s], axis=0)
        f[v + '_anom'] = cl[v] - clim[doy]
    tmax, hrmin, p = cl['tmax'], cl['hrmin'], np.nan_to_num(cl['precip'])
    # VPD proxy (kPa) con tmax y hrmin
    es = 0.6108 * np.exp(17.27 * tmax / (tmax + 237.3)); f['vpd'] = es * (1 - hrmin / 100)
    # KBDI simplificado (Keetch-Byram, 0-800), lluvia neta tras 5 mm
    rain_year = p.sum(0) / (p.shape[0] / 365.25)
    corr = 1 + 10.88 * np.exp(-0.001736 * rain_year)   # Rain anual en mm (formula original usa pulgadas*)
    Q = np.zeros(p.shape[1], np.float32); kb = np.zeros_like(p, dtype=np.float32)
    for t in range(p.shape[0]):
        Tf = np.nan_to_num(tmax[t]) * 9 / 5 + 32
        net = np.maximum(p[t] - 5.0, 0) / 25.4 * 100
        Q = np.maximum(Q - net, 0)
        inc = (800 - Q) * (0.968 * np.exp(0.0486 * Tf) - 8.3) * 0.001 / corr * 1.0
        Q = Q + np.maximum(inc, 0) * (p[t] <= 5.0); kb[t] = Q
    f['kbdi'] = kb
    f['kbdi_d7'] = kb - lagf(kb, 7)
    f['tmax_m7'] = roll(cl['tmax'], 7, 'mean'); f['hrmin_m7'] = roll(cl['hrmin'], 7, 'mean')
    f['precip_s7'] = roll(cl['precip'], 7); f['precip_s30'] = roll(cl['precip'], 30); f['precip_s90'] = roll(cl['precip'], 90)
    ds = np.zeros_like(p); 
    for t in range(1, len(p)): ds[t] = np.where(p[t] > 1.0, 0, np.minimum(ds[t - 1] + 1, 60))
    f['dias_sin_lluvia'] = ds
    f['tmax_l1'] = lagf(cl['tmax'], 1); f['hrmin_l1'] = lagf(cl['hrmin'], 1)
    for v in cl: f[v] = cl[v]
    return {k: v[OFF:].astype(np.float32) for k, v in f.items()}


CLIM_RAW = ['tmax', 'hrmin', 'hr_media', 'viento_max', 'precip', 'tmax_l1', 'hrmin_l1', 'tmax_m7', 'hrmin_m7', 'precip_s7', 'precip_s30', 'precip_s90', 'dias_sin_lluvia']
CLIM_NEW = ['tmax_anom', 'hrmin_anom', 'viento_max_anom', 'vpd', 'kbdi', 'kbdi_d7']


class Data:
    def __init__(self, label='v'):
        self.z = load(); z = self.z
        self.mask = z['cell_com'] >= 0
        self.iy, self.ix = np.where(self.mask); self.nc = len(self.iy)
        self.ci = z['cell_com'][self.iy, self.ix].astype(int)
        self.lf = (z['codes'][self.ci] == '13110')
        self.clim = climate_feats(z)
        self.det = {k: z['det_' + k] for k in ['v', 'vl', 'm']}
        d = self.det['v']
        self.d3 = dilate(d, 1); self.pers3 = recent_any(self.d3, 3)       # actividad 3x3 en D-2..D
        self.lab_cache = {}
        self.season = np.isin(MONTH, [9, 10, 11, 12, 1, 2, 3, 4])
        self.valid_d = np.where(self.season & (np.arange(ND) >= 90) & (np.arange(ND) < ND - 2))[0]

    def y(self, label='v', hz='12'):
        """etiqueta [ND,H,W]: foco en D+1 o D+2 ('12'), solo D+1 ('1'), solo D+2 ('2')."""
        key = (label, hz)
        if key not in self.lab_cache:
            if label == 'vm': a = np.maximum(self.det['v'], self.det['m'])
            else: a = self.det[label]
            y = np.zeros_like(a)
            if hz == '12': y[:-2] = np.maximum(a[1:-1], a[2:])
            elif hz == '1': y[:-1] = a[1:]
            else: y[:-2] = a[2:]
            self.lab_cache[key] = y
        return self.lab_cache[key]

    def cells(self, vol):  # [ND,H,W] -> [ND,nc]
        return vol[:, self.iy, self.ix]


def rate_map(yvol, years, new_pop=None):
    """tasa mensual por celda a partir de 'years' ([13,H,W]); si new_pop, solo cuenta filas en la poblacion."""
    f = np.zeros((13, H, W), np.float32)
    for m in range(1, 13):
        sel = np.isin(YEAR, years) & (MONTH == m)
        if sel.sum(): f[m] = yvol[sel].mean(0)
    return f


def hist_feats(D, yvol, days, train_years, mode='loyo'):
    """Features de historia para las filas (days x celdas). Dias de train_years: leave-one-year-out; otros: todos train_years.
    mode='insample': como v1 (usa todos train_years tambien en sus propias filas)."""
    out = {k: np.zeros((len(days), D.nc), np.float32) for k in ['h5', 'h5a', 'h11', 'hn', 'h1', 'h3']}
    ydays = YEAR[days]
    groups = [(y_, [t for t in train_years if t != y_]) for y_ in sorted(set(ydays) & set(train_years))] if mode == 'loyo' else []
    done = np.zeros(len(days), bool)
    def fill(sel, yrs):
        f = rate_map(yvol, yrs); anyf = (np.stack([f[m] for m in range(1, 13)]).mean(0))
        f5 = box_mean(f, 2); f11 = box_mean(f, 5); a5 = box_mean(anyf[None], 2)[0]
        # numero de anios (de yrs) con algun foco en 5x5 (mismo mes del anio no, anual total)
        cnt = np.zeros((H, W), np.float32)
        for y_ in yrs: cnt += dilate(yvol[YEAR == y_].max(0), 2)
        cnt /= len(yrs)
        idx = np.where(sel)[0]; mm = MONTH[days[idx]]
        out['h5'][idx] = f5[mm][:, D.iy, D.ix]; out['h11'][idx] = f11[mm][:, D.iy, D.ix]
        out['h5a'][idx] = a5[D.iy, D.ix][None]; out['hn'][idx] = cnt[D.iy, D.ix][None]
        out['h1'][idx] = anyf[D.iy, D.ix][None]; out['h3'][idx] = box_mean(anyf[None], 1)[0][D.iy, D.ix][None]
        done[idx] = True
    for y_, yrs in groups: fill(ydays == y_, yrs)
    fill(~done, list(train_years))
    return out


def recency_feats(D, days):
    d = D.det['v']
    n30 = np.zeros((ND, H, W), np.float32); cs = np.cumsum(d, 0, dtype=np.int32); n30[30:] = cs[30:] - cs[:-30]
    nb30 = box_mean(n30, 2) * 25                  # focos en 5x5 en 30 dias
    pers3 = D.pers3
    out = {'pers3': D.cells(pers3)[days].astype(np.float32), 'n30': D.cells(n30)[days], 'nb30': D.cells(nb30)[days]}
    return out


# ---------------------------------------------------------------- ensamblado, ajuste y evaluacion
import lightgbm as lgb
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

PARAMS = dict(n_estimators=600, learning_rate=0.03, num_leaves=15, min_child_samples=200, subsample=0.8, subsample_freq=1,
              colsample_bytree=0.8, reg_lambda=5.0, verbose=-1, random_state=0, n_jobs=10)
PARAMS_REG = dict(PARAMS, num_leaves=7, min_child_samples=3000, reg_lambda=50.0, colsample_bytree=0.6, learning_rate=0.02)


def assemble(D, days, cfg, train_years):
    """DataFrame de features para las filas days x celdas (orden dia-mayor) y vectores flat idx/day."""
    nd = len(days)
    cols = {}
    for k in cfg.get('clim', []):
        cols[k] = D.clim[k][days][:, D.ci].ravel()
    yvol_h = D.y(cfg.get('hist_label', 'v'), '12') if cfg.get('hist_target', 'all') == 'all' else None
    if cfg.get('hist_target') == 'new':
        yvol_h = D.y('v', '12') * (1 - D.pers3)
    if cfg.get('hist'):
        h = hist_feats(D, yvol_h, days, train_years, cfg['hist_mode'])
        for k in cfg['hist']: cols[k] = h[k].ravel()
    if cfg.get('rec'):
        r = recency_feats(D, days)
        for k in cfg['rec']: cols[k] = r[k].ravel()
    if 'mes' in cfg.get('cal', []): cols['mes'] = np.repeat(MONTH[days], D.nc).astype(np.float32)
    if 'doy' in cfg.get('cal', []): cols['doy'] = np.repeat(DAYS.dayofyear.values[days], D.nc).astype(np.float32)
    if 'dow' in cfg.get('cal', []): cols['dow'] = np.repeat(DAYS.dayofweek.values[days], D.nc).astype(np.float32)
    if cfg.get('latlon'):
        cols['lat'] = np.tile(LAT0 + (D.iy + .5) * CELL, nd).astype(np.float32); cols['lon'] = np.tile(LON0 + (D.ix + .5) * CELL, nd).astype(np.float32)
    X = pd.DataFrame(cols)
    day_flat = np.repeat(days, D.nc); cell_flat = np.tile(np.arange(D.nc), nd)
    return X, day_flat, cell_flat


def days_of(D, years, gap_lo=0):
    return D.valid_d[np.isin(YEAR[D.valid_d], years)]


def fit_predict(D, cfg, tr_years, va_years, te_years, train_hist_years=None, verbose=False):
    """Entrena en tr_years (early stopping en va_years si hay), predice va y te. Devuelve p (flat len ND*nc, NaN fuera)."""
    ytr = D.cells(D.y(cfg.get('train_label', 'v'), cfg.get('hz', '12')))
    npop = D.cells(D.pers3) == 0
    def rows(years, hist_years, hist_mode):
        days = days_of(D, years)
        X, df, cf = assemble(D, days, dict(cfg, hist_mode=hist_mode), hist_years)
        return X, df, cf
    # train
    X_tr, d_tr, c_tr = rows(tr_years, tr_years, cfg['hist_mode'])
    y_tr = ytr[d_tr, c_tr]
    keep = np.ones(len(y_tr), bool)
    if cfg.get('pop') == 'new': keep = npop[d_tr, c_tr]
    X_tr, y_tr = X_tr[keep], y_tr[keep]
    # valid (usa historia de tr_years)
    pred = np.full(ND * D.nc, np.nan, np.float32)
    model_kind = cfg.get('model', 'lgbm')
    X_va = None
    if va_years:
        X_va, d_va, c_va = rows(va_years, tr_years, 'loyo')
        y_va = ytr[d_va, c_va]
        kv = npop[d_va, c_va] if cfg.get('pop') == 'new' else np.ones(len(d_va), bool)
    if model_kind == 'lgbm':
        prm = dict(cfg.get('params', PARAMS))
        m = lgb.LGBMClassifier(**prm)
        if cfg.get('mono'):
            m.set_params(monotone_constraints=[cfg['mono'].get(c, 0) for c in X_tr.columns])
        if va_years:
            m.fit(X_tr, y_tr, eval_set=[(X_va[kv], y_va[kv])], callbacks=[lgb.early_stopping(60, verbose=False)])
        else:
            m.fit(X_tr, y_tr)
        predict = lambda X: m.predict_proba(X)[:, 1]
        info = dict(iters=int(m.best_iteration_ or prm['n_estimators']), imp=dict(zip(X_tr.columns, (m.booster_.feature_importance('gain') / max(1, m.booster_.feature_importance('gain').sum())).round(3))))
    else:  # logistica penalizada
        mu = X_tr.mean(); sd = X_tr.std().replace(0, 1)
        def tf(X):
            Z = ((X - mu) / sd).fillna(0).clip(-5, 5)
            if cfg.get('inter'):
                for a, b in cfg['inter']: Z[a + '*' + b] = Z[a] * Z[b]
            return Z
        m = LogisticRegression(C=cfg.get('C', 0.05), max_iter=400).fit(tf(X_tr), y_tr)
        predict = lambda X: m.predict_proba(tf(X))[:, 1]
        info = dict(coef=dict(zip(tf(X_tr.head(2)).columns, m.coef_[0].round(3))))
    out = {}
    for nm, yrs, hy in (('va', va_years, tr_years), ('te', te_years, train_hist_years or tr_years)):
        if not yrs: continue
        X, d_, c_ = (X_va, d_va, c_va) if nm == 'va' else rows(yrs, hy, 'loyo')
        p = predict(X)
        pf = np.full(ND * D.nc, np.nan, np.float32); pf[d_ * D.nc + c_] = p
        out[nm] = pf
    return out, info


def rolling(D, cfg, pops=('todo', 'nuevo'), folds=((2021, [2018, 2019, 2020]), (2022, [2018, 2019, 2020, 2021]), (2023, [2018, 2019, 2020, 2021, 2022])), log=print):
    """Rolling origin: ajusta con anios previos, evalua el anio siguiente; early stopping SOLO con el mismo anio de validacion (ligero optimismo
    comun a todas las variantes, no usa la prueba). Devuelve p pooled flat de 2021-2023."""
    pooled = np.full(ND * D.nc, np.nan, np.float32); infos = []
    for vy, tr in folds:
        o, info = fit_predict(D, cfg, tr, [vy], [], None)
        v = o['va']; ok = ~np.isnan(v); pooled[ok] = v[ok]; infos.append(info['iters'] if 'iters' in info else 0)
    return pooled, infos


def final(D, cfg):
    tr = [2018, 2019, 2020, 2021, 2022]
    o, info = fit_predict(D, cfg, tr, [2023], [2024, 2025], tr + [2023])
    return o, info


# ---------------------------------------------------------------- iteraciones
BASE = dict(clim=CLIM_RAW, cal=['mes', 'doy', 'dow'], hist=['h5'], hist_mode='insample', latlon=True, pop='todo', model='lgbm')
MONO_H = {'h1': 1, 'h3': 1, 'h5': 1, 'h5a': 1, 'h11': 1, 'hn': 1, 'nb30': 1, 'n30': 1, 'kbdi': 1, 'vpd': 1, 'tmax_anom': 1}
CFG = {
    'it0_v1_C': BASE,
    'it0_v1_D': dict(BASE, rec=['pers3', 'n30']),
    'it1_loyo': dict(BASE, hist_mode='loyo'),
    'it2_sin_latlon': dict(BASE, hist_mode='loyo', latlon=False, hist=['h5', 'h5a', 'h11', 'hn']),
    'it2b_solo_historia': dict(BASE, hist_mode='loyo', latlon=False, clim=[], cal=['mes'], hist=['h5', 'h5a', 'h11', 'hn']),
    'it3_nuevo': dict(BASE, hist_mode='loyo', latlon=False, hist=['h5', 'h5a', 'h11', 'hn', 'h1', 'h3'], pop='new', hist_target='new', rec=['n30', 'nb30']),
    'it3b_nuevo_solo_hist': dict(BASE, hist_mode='loyo', latlon=False, clim=[], cal=['mes'], hist=['h5', 'h5a', 'h11', 'hn', 'h1', 'h3'], pop='new', hist_target='new', rec=['n30', 'nb30']),
    'it4_clima_anom_kbdi': dict(BASE, hist_mode='loyo', latlon=False, hist=['h5', 'h5a', 'h11', 'hn', 'h1', 'h3'], pop='new', hist_target='new', rec=['n30', 'nb30'], clim=CLIM_RAW + CLIM_NEW),
    'it5_reg_mono': dict(BASE, hist_mode='loyo', latlon=False, hist=['h5', 'h5a', 'h11', 'hn', 'h1', 'h3'], pop='new', hist_target='new', rec=['n30', 'nb30'], clim=CLIM_NEW + ['precip_s30', 'dias_sin_lluvia'], cal=['doy'], params=PARAMS_REG, mono=MONO_H),
}
CFG['it5b_logistica'] = dict(CFG['it5_reg_mono'], model='logit', C=0.05, inter=[('h5', 'kbdi'), ('h5', 'vpd'), ('h5', 'tmax_anom')])


def build_evals(D):
    yflat = D.cells(D.y('v', '12')).ravel()
    dayflat = np.repeat(np.arange(ND), D.nc)
    npop = (D.cells(D.pers3) == 0).ravel()
    ev = {}
    for nm, yrs in (('val', [2021, 2022, 2023]), ('test', [2024, 2025])):
        inyr = np.isin(YEAR[dayflat], yrs) & D.season[dayflat] & (dayflat >= 90) & (dayflat < ND - 2)
        ev[(nm, 'todo')] = Eval(yflat, dayflat, inyr, seed=0)
        ev[(nm, 'nuevo')] = Eval(yflat, dayflat, inyr & npop, seed=0)
    return ev, yflat, dayflat, npop


def run_iter(name, extra_cfg=None):
    t0 = time.time(); D = Data(); ev, yflat, dayflat, npop = build_evals(D)
    cfg = dict(CFG[name], **(extra_cfg or {}))
    pooled, its = rolling(D, cfg)
    resj = {'cfg': {k: v for k, v in cfg.items() if k not in ('params', 'mono')}, 'rolling_iters': its}
    store = {}
    for pop in ('todo', 'nuevo'):
        e = ev[('val', pop)]; p = np.nan_to_num(pooled, nan=1e-6)
        resj['val_' + pop] = e.point(p); resj['val_' + pop + '_ci'] = {k: v for k, v in e.boot(p, 100).items()}
        store['val_' + pop] = p[e.idx]
    # --- prueba: UNA mirada por iteracion
    o, info = final(D, cfg)
    pt, pv = o['te'], o['va']
    resj['final_info'] = info
    cnt_f = P + 'v2_pruebas_miradas.txt'
    with open(cnt_f, 'a') as f: f.write(f'{time.strftime("%Y%m%d-%H%M")} {name}\n')
    # calibracion isotonica con 2023 (solo en filas validas de val)
    vi = ~np.isnan(pv); iso = IsotonicRegression(out_of_bounds='clip', y_min=1e-7).fit(pv[vi], yflat[vi])
    for pop in ('todo', 'nuevo'):
        e = ev[('test', pop)]; p = np.nan_to_num(pt, nan=1e-6)
        resj['test_' + pop] = e.point(p); resj['test_' + pop + '_ci'] = e.boot(p, 100)
        pc = np.where(np.isnan(pt), 1e-6, iso.predict(np.nan_to_num(pt))).astype(np.float32)
        resj['test_' + pop + '_cal'] = e.point(pc)
        store['test_' + pop] = p[e.idx]; store['test_' + pop + '_cal'] = pc[e.idx]
    popt = np.isin(YEAR[dayflat], [2024, 2025]) & D.season[dayflat] & (dayflat >= 90) & (dayflat < ND - 2)
    for N in (10, 30):
        resj[f'op_nuevo_N{N}'] = op_metric(np.nan_to_num(pt, nan=-1), yflat, dayflat, popt & npop, N)
    lf_te = popt & np.tile(D.lf, ND)
    resj['laflorida_test'] = dict(filas=int(lf_te.sum()), positivos=int(yflat[lf_te].sum()), p_media=float(np.nanmean(pt[lf_te])), p_esperado_celdas=float(np.nansum(pt[lf_te])))
    np.savez_compressed(P + f'v2_preds_{name}.npz', **store)
    json.dump(resj, open(P + f'v2_res_{name}.json', 'w'), indent=1, default=float)
    print(name, 'listo en %.0f s' % (time.time() - t0))
    for k in ('val_todo', 'val_nuevo', 'test_todo', 'test_nuevo', 'test_nuevo_cal'): print(k, {a: round(b, 4) for a, b in resj[k].items()})
    print('iters', its, info.get('iters'), 'imp', dict(sorted(info.get('imp', {}).items(), key=lambda x: -x[1])[:6]))


if __name__ == '__main__' and len(sys.argv) > 2 and sys.argv[1] == 'iter':
    run_iter(sys.argv[2])
