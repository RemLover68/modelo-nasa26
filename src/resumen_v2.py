import json,glob,sys,numpy as np
names=sys.argv[1:] or sorted(f[len('datos/proc/v2_res_'):-5] for f in glob.glob('datos/proc/v2_res_*.json'))
def f(m,ci,k):
    v=m[k]; c=ci.get(k); return f"{v:.4f}"+(f" [{c[0]:.4f},{c[1]:.4f}]" if c else "")
for pop in ('nuevo','todo'):
  for part in ('val','test'):
    print(f'== {part} / {pop}')
    for n in names:
        r=json.load(open(f'datos/proc/v2_res_{n}.json')); m=r[f'{part}_{pop}']; ci=r[f'{part}_{pop}_ci']
        print(f"{n:24s} PR {f(m,ci,'pr_auc')} | ROC {m['roc_auc']:.3f} | LLskill {f(m,ci,'ll_skill')} | top1 {m['top1pct']:.3f} top5 {m['top5pct']:.3f}")
print('== operativa test nuevo N=10 / 30 (aciertos/dia, azar)')
for n in names:
    r=json.load(open(f'datos/proc/v2_res_{n}.json')); a,b=r['op_nuevo_N10'],r['op_nuevo_N30']
    print(f"{n:24s} N10 {a['aciertos_dia']:.3f} (azar {a['azar_aciertos_dia']:.3f}, recall {a['recall']:.3f}) | N30 {b['aciertos_dia']:.3f} (azar {b['azar_aciertos_dia']:.3f}, recall {b['recall']:.3f})")
