import sys, pathlib, json
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import numpy as np
from make_figures import load_layers, load_coords, ART, S

def sobel_mag(z):
    gx = np.zeros_like(z); gy = np.zeros_like(z)
    gx[:,1:-1] = (z[:,2:]-z[:,:-2])*0.5
    gy[1:-1,:] = (z[2:,:]-z[:-2,:])*0.5
    return np.hypot(gx, gy)

def p(a,b):
    a=a.ravel().astype(float); b=b.ravel().astype(float)
    a-=a.mean(); b-=b.mean()
    d=np.sqrt((a*a).sum()*(b*b).sum())
    return float((a*b).sum()/d) if d else np.nan

layers = load_layers(); rows = load_coords()
rng = np.random.default_rng(99)
out={}
for K in (2,4,6):
    coords=[r for r in rows if r['tier']==K]
    edges=[(r,next(q for q in coords if q['anchor_id']==r['anchor_id'] and q['level']==r['level']+1))
           for r in coords if r['level']<K-1]
    sr_s=[]; sr_l=[]; null=[]
    for r,q in edges:
        wf=q['side']; wk=r['side']
        xk,yk=r['x0'],r['y0']; xf,yf=q['x0'],q['y0']
        As=layers['shared'][q['level']][yf:yf+wf, xf:xf+wf]
        Bs=layers['shared'][r['level']][yk:yk+wk, xk:xk+wk][::2,::2]
        Al=layers['layered'][q['level']][yf:yf+wf, xf:xf+wf]
        Bl=layers['layered'][r['level']][yk:yk+wk, xk:xk+wk][::2,::2]
        sr_s.append(p(sobel_mag(As), sobel_mag(Bs)))
        sr_l.append(p(sobel_mag(Al), sobel_mag(Bl)))
        for _ in range(30):
            xA=rng.integers(0,S-wf); yA=rng.integers(0,S-wf)
            xB=rng.integers(0,S-wk); yB=rng.integers(0,S-wk)
            A=layers['layered'][q['level']][yA:yA+wf, xA:xA+wf]
            B=layers['layered'][r['level']][yB:yB+wk, xB:xB+wk][::2,::2]
            if B.shape==(wf,wf):
                null.append(p(sobel_mag(A), sobel_mag(B)))
    out[K]=dict(shared_ridge=float(np.mean(sr_s)),
                layered_ridge=float(np.mean(sr_l)), layered_ridge_sd=float(np.std(sr_l)),
                null_ridge=float(np.mean(null)), null_ridge_sd=float(np.std(null)))
    print(f"K={K}: shared ridge r={np.mean(sr_s):.4f} | layered ridge r={np.mean(sr_l):+.3f}"
          f"+-{np.std(sr_l):.3f} | null ridge r={np.mean(null):+.3f}+-{np.std(null):.3f}")
(ART/'ridge_correlation.json').write_text(json.dumps(out,indent=2))
