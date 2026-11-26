import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import numpy as np
from goproc_port import fbm_layers

S=1024; F=0.02; SEED=7; R=2
CENTERS=[256,512,768]
W0=256

def bbox(c, w):
    h=w//2
    return (c[0]-h, c[1]-h, c[0]+h, c[1]+h)  # x0,y0,x1,y1

def level_box(c0, k):
    d=R**k
    c=(c0[0]//d, c0[1]//d)
    w=W0//d
    return c, w, bbox(c,w)

def pearson(a,b):
    a=a.ravel().astype(float); b=b.ravel().astype(float)
    a=a-a.mean(); b=b-b.mean()
    den=np.sqrt((a*a).sum()*(b*b).sum())
    return float((a*b).sum()/den) if den>0 else np.nan

def sobel_mag(z):
    gx=np.zeros_like(z); gy=np.zeros_like(z)
    gx[:,1:-1]=(z[:,2:]-z[:,:-2])*0.5
    gy[1:-1,:]=(z[2:,:]-z[:-2,:])*0.5
    return np.hypot(gx,gy)

for K in (2,4,6):
    raw_s,_=fbm_layers(S,SEED,K,R,0.5,F,"shared")
    raw_l,_=fbm_layers(S,SEED,K,R,0.5,F,"layered")
    cs=[]; cl=[]; rs=[]; rl=[]
    npairs=0
    for cx in CENTERS:
        for cy in CENTERS:
            c0=(cx,cy)
            for k in range(K-1):
                ck,wk,bk=level_box(c0,k)
                k1,wf,bf=level_box(c0,k+1)
                x0k,y0k,x1k,y1k=bk
                x0f,y0f,x1f,y1f=bf
                A_s=raw_s[k+1][y0f:y1f, x0f:x1f]
                B_s=raw_s[k][y0k:y1k, x0k:x1k][::R,::R]
                A_l=raw_l[k+1][y0f:y1f, x0f:x1f]
                B_l=raw_l[k][y0k:y1k, x0k:x1k][::R,::R]
                assert A_s.shape==B_s.shape==(wf,wf), (A_s.shape,B_s.shape,wf)
                cs.append(pearson(A_s,B_s)); cl.append(pearson(A_l,B_l))
                # ridge-magnitude version
                rs.append(pearson(sobel_mag(A_s), sobel_mag(B_s)))
                rl.append(pearson(sobel_mag(A_l), sobel_mag(B_l)))
                npairs+=1
    print(f"K={K} pairs={npairs}")
    print(f"  aligned NCC  shared mean={np.mean(cs):.5f} min={np.min(cs):.5f} | "
          f"layered mean={np.mean(cl):+.4f} sd={np.std(cl):.4f} "
          f"range[{np.min(cl):+.2f},{np.max(cl):+.2f}]")
    print(f"  ridge   NCC  shared mean={np.mean(rs):.4f} min={np.min(rs):.3f} | "
          f"layered mean={np.mean(rl):+.4f} sd={np.std(rl):.4f}")
