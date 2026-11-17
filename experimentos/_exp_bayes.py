import os, numpy as np, itertools
import _caminho
from pavimento.prototipos import IDX, DESCRITORES
d = np.load(os.path.join(_caminho.SAIDAS, "blocos_dnit.npz"))
X, y, img, teste = d["X"][:, IDX].astype(np.float64), d["y"], d["img"], d["teste"]
val = (~teste) & (img % 5 == 0); tr = (~teste) & ~val
mu0, sd0 = X[tr].mean(0), X[tr].std(0) + 1e-9
Z = (X - mu0) / sd0
def f1(p, t):
    tp=(p&t).sum(); fp=(p&~t).sum(); fn=(~p&t).sum(); pr=tp/max(tp+fp,1); rc=tp/max(tp+fn,1); return 2*pr*rc/max(pr+rc,1e-9)
def bayes_scores(cols, Zs):
    S=[]
    for c in range(3):
        Zc=Z[tr & (y==c)][:, cols]
        m=Zc.mean(0); C=np.cov(Zc.T)+1e-3*np.eye(len(cols)); Ci=np.linalg.inv(C); _,ld=np.linalg.slogdet(C)
        D=Zs[:, cols]-m
        S.append(-0.5*np.einsum('ij,jk,ik->i',D,Ci,D)-0.5*ld)   # log-verossimilhança (sem prior)
    return np.stack(S,1)
def melhor_bias(S, yv):
    best=(0,0,0)
    for bt, bb in itertools.product(np.arange(-4,8.1,0.5), np.arange(-4,8.1,0.5)):
        p=(S+np.array([0,bt,bb])).argmax(1); n=f1(p==1,yv==1)+f1(p==2,yv==2)
        if n>best[0]: best=(n,bt,bb)
    return best
allc=list(range(len(IDX)))
Sv=bayes_scores(allc, Z[val]); b=melhor_bias(Sv, y[val])
St=bayes_scores(allc, Z[teste]); p=(St+np.array([0,b[1],b[2]])).argmax(1)
print('Bayes gaussiano (todos descritores) val', round(b[0],4), 'teste trinca', round(f1(p==1,y[teste]==1),4), 'buraco', round(f1(p==2,y[teste]==2),4), 'bias', b[1:])
# seleção gulosa para frente (Bayes)
sel=[]; resto=allc[:]; atual=0
while resto:
    cand=[]
    for c in resto:
        s=bayes_scores(sel+[c], Z[val]); cand.append((melhor_bias(s,y[val])[0], c))
    n,c=max(cand)
    if n<=atual+0.003: break
    atual=n; sel.append(c); resto.remove(c); print(' +', DESCRITORES[c], round(n,4), flush=True)
Sv=bayes_scores(sel, Z[val]); b=melhor_bias(Sv,y[val]); St=bayes_scores(sel, Z[teste]); p=(St+np.array([0,b[1],b[2]])).argmax(1)
print('Bayes + seleção', [DESCRITORES[c] for c in sel], 'teste trinca', round(f1(p==1,y[teste]==1),4), 'buraco', round(f1(p==2,y[teste]==2),4), 'bias', b[1:])
