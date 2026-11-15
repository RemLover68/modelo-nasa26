import sys; sys.path.insert(0,'src')
import numpy as np, pandas as pd
from scipy import ndimage
from sklearn.metrics import roc_auc_score, average_precision_score
from modelo_rm_v2 import *
z=load(); mask=z['cell_com']>=0
season=np.isin(MONTH,[9,10,11,12,1,2,3,4])
for k in ['det_v','det_vl','det_m']:
    a=z[k]; print(k,'celda-dias',int(a.sum()),'por anio',[int(a[YEAR==y].sum()) for y in range(2018,2026)])
a=z['det_v']
# eventos: componentes conexas 3x3 espacial, hueco temporal <=2 dias (dilatar 1 dia en t)
b=a.copy(); b[1:]=np.maximum(b[1:],a[:-1]); b[:-1]=np.maximum(b[:-1],a[1:])
lab,n=ndimage.label(b,structure=np.ones((3,3,3)))
lab=lab*a  # solo celdas con deteccion
ids=np.unique(lab[lab>0]); print('eventos (cc, hueco 2d, 8-vecinos):',len(ids),'vs celda-dias',int(a.sum()))
sizes=np.bincount(lab[lab>0].ravel()); sizes=sizes[sizes>0]
print('tamano evento (celda-dias): mediana',np.median(sizes),'p90',np.percentile(sizes,90),'max',sizes.max(),'| % eventos de 1 celda-dia',100*(sizes==1).mean())
ev_ids=np.zeros(0); 
# eventos por anio (por dia de inicio)
first={}
tt,yy,xx=np.where(lab>0)
df=pd.DataFrame({'t':tt,'e':lab[tt,yy,xx]}); st=df.groupby('e').t.min()
print('eventos por anio',[int((YEAR[st.values]==y).sum()) for y in range(2018,2026)])
print('eventos en temporada sep-abr',int(season[st.values].sum()),'de',len(st))
# y_all, persistencia, nuevo
y_all=np.zeros_like(a); y_all[:-2]=np.maximum(a[1:-1],a[2:])
d3=dilate(a,1); pers3=recent_any(d3,3)
dayok=season&(np.arange(ND)>=90)&(np.arange(ND)<ND-2)
pop=season[:,None,None]&mask[None]&(np.arange(ND)>=90)[:,None,None]&(np.arange(ND)<ND-2)[:,None,None]
print('positivos (celda-dia) en temporada:',int(y_all[pop].sum()),'| con actividad 3x3 en D-2..D (en curso):',int((y_all&pers3)[pop].sum()),'| nuevos:',int((y_all&(1-pers3))[pop].sum()))
print('prevalencia nuevo en poblacion sin actividad: %.4f %%'%(100*(y_all&(1-pers3))[pop&(pers3==0)].mean()))
# fuga: freq in-sample vs LOYO
yv=y_all
def rate(years):
    f=np.zeros((13,H,W),np.float32)
    for m in range(1,13):
        sel=np.isin(YEAR,years)&(MONTH==m)
        if sel.sum(): f[m]=yv[sel].mean(0)
    return f
sm=lambda f:box_mean(f,2)
tr=[2018,2019,2020,2021,2022]
fin=sm(rate(tr))
dd=np.where(dayok&np.isin(YEAR,tr))[0]
cells=np.where(mask)
def col(f,days): return f[MONTH[days]][:,cells[0],cells[1]]  # [nd,ncell]
Xin=col(fin,dd)
Yt=yv[dd][:,cells[0],cells[1]]
oof=np.zeros_like(Xin)
for y_ in tr:
    f=sm(rate([t for t in tr if t!=y_])); s=YEAR[dd]==y_; oof[s]=col(f,dd[s])
print('AUC freq in-sample train: %.3f  PR %.4f | LOYO: %.3f PR %.4f'%(roc_auc_score(Yt.ravel(),Xin.ravel()),average_precision_score(Yt.ravel(),Xin.ravel()),roc_auc_score(Yt.ravel(),oof.ravel()),average_precision_score(Yt.ravel(),oof.ravel())))
# freq train->2023 val
dv=np.where(dayok&(YEAR==2023))[0]; Xv=col(fin,dv); Yv=yv[dv][:,cells[0],cells[1]]
print('AUC freq (train->val 2023): %.3f PR %.4f'%(roc_auc_score(Yv.ravel(),Xv.ravel()),average_precision_score(Yv.ravel(),Xv.ravel())))
# concentracion: celdas con algun positivo
print('celdas con >=1 foco en 8 anios',int((a.sum(0)>0).sum()),'de',int(mask.sum()))
