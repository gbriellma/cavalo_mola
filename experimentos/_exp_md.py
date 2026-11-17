import os, numpy as np, itertools, time
import _caminho
from pavimento.prototipos import IDX, DESCRITORES, kmeans
d = np.load(os.path.join(_caminho.SAIDAS, "blocos_dnit.npz"))
X, y, img, teste = d["X"][:, IDX].astype(np.float64), d["y"], d["img"], d["teste"]
val = (~teste) & (img % 5 == 0); tr = (~teste) & ~val
Z = (X - X[tr].mean(0)) / (X[tr].std(0) + 1e-9)
def f1(p, t):
    tp=(p&t).sum(); fp=(p&~t).sum(); fn=(~p&t).sum(); pr=tp/max(tp+fp,1); rc=tp/max(tp+fn,1); return 2*pr*rc/max(pr+rc,1e-9)
def protos(cols, k):
    P=[]
    for c,kc in zip(range(3),k):
        Zc=Z[tr&(y==c)][:,cols]; Zc=Zc[::max(1,len(Zc)//20000)]
        P.append(kmeans(Zc,kc,iters=25))
    return P
def dist(P, cols, Zs):
    return np.stack([((Zs[:,None,cols]-p[None])**2).sum(2).min(1) for p in P],1)
def melhor(D, yv):
    best=(0,1,1)
    for wt,wb in itertools.product(np.arange(0.5,2.01,0.1), np.arange(0.5,2.51,0.1)):
        p=(D*np.array([1,wt,wb])).argmin(1); n=f1(p==1,yv==1)+f1(p==2,yv==2)
        if n>best[0]: best=(n,wt,wb)
    return best
t0=time.time()
k=(32,24,12)
vsub = np.flatnonzero(val)[::2]
allc=list(range(len(IDX)))
# seleção gulosa para trás (remove o descritor cuja retirada mais melhora)
sel=allc[:]
P=protos(sel,k); base=melhor(dist(P,sel,Z[vsub]),y[vsub])[0]; print('todos', round(base,4), flush=True)
while len(sel)>4 and time.time()-t0<120:
    cand=[]
    for c in sel:
        s=[x for x in sel if x!=c]; P=protos(s,(16,12,6)); cand.append((melhor(dist(P,s,Z[vsub]),y[vsub])[0],c))
    n,c=max(cand)
    if n<=base+0.002: break
    base=n; sel.remove(c); print(' -', DESCRITORES[c], round(n,4), flush=True)
P=protos(sel,k); b=melhor(dist(P,sel,Z[val]),y[val]); Dt=dist(P,sel,Z[teste]); p=(Dt*np.array([1,b[1],b[2]])).argmin(1)
print('DistMin', k, [DESCRITORES[c] for c in sel], 'teste trinca', round(f1(p==1,y[teste]==1),4), 'buraco', round(f1(p==2,y[teste]==2),4), 'pesos', round(b[1],2), round(b[2],2))
