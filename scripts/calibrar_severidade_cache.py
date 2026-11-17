"""Calibra pesos do índice e limiares de severidade só no treino (cache), contra a referência congelada.

Uso: python scripts/calibrar_severidade_cache.py <raiz-src> params.json saida.json
"""
import json, os, sys, itertools
from multiprocessing import Pool
import numpy as np
import avaliar_cache as bench

def feat(args):
    n,pd=args
    nu,pa=bench.M["nu"],bench.M["pa"]
    p=pa.Parametros(**{k:v for k,v in pd.items() if k in pa.Parametros.__dataclass_fields__})
    g,roi,gt,gb=bench.carregar(n)
    r=nu.analisar(g,roi,p)
    comp=sum(x["comprimento"] for x in r["regioes_trinca"]); jac=sum(x["comprimento"] for x in r["regioes_trinca"] if x["tipo"]=="couro_de_jacare")
    ab=sum(x["area"] for x in r["regioes_buraco"]); nr=int(roi.sum())
    cg=int(nu.afinar(gt&roi).sum()); ag=int((gb&roi).sum())
    return comp+jac, ab, nr, bench.sev(bench.indice(cg,ag,nr))

def indice(c,a,n,wt,wb):
    n=max(n,1); return min(1000,(wt*1000*c)//n+(wb*((10000*a)//n))//10)

if __name__=="__main__":
    src,pj,out=sys.argv[1:4]
    pd=json.load(open(pj))
    bench.init(src)
    ns=bench.nomes("treino")
    with Pool(os.cpu_count(),initializer=bench.init,initargs=(src,)) as pool:
        F=pool.map(feat,[(n,pd) for n in ns],chunksize=8)
    niv={"nenhum":0,"baixa":1,"media":2,"alta":3}
    y=np.array([niv[f[3]] for f in F])
    best=(-1,)
    for wt,wb in itertools.product((10,20,30,40,60,80,120),(5,10,15,25,40,60,100)):
        ind=np.array([indice(f[0],f[1],f[2],wt,wb) for f in F])
        cuts=np.unique(np.concatenate([[1,5,10,20],np.percentile(ind,np.linspace(2,98,60)).astype(int)]))
        o=np.argsort(ind,kind="stable"); si=ind[o]; sy=y[o]
        pre=np.zeros((4,len(si)+1),int)
        for c in range(4): pre[c,1:]=np.cumsum(sy==c)
        pos=np.searchsorted(si,cuts,side="left")  # nº de elementos com ind<cut
        L=len(si); bst=(-1,)
        for i,a in enumerate(pos):
            for j in range(i+1,len(pos)):
                b=pos[j]
                for k in range(j+1,len(pos)):
                    c=pos[k]
                    s=pre[0,a]+pre[1,b]-pre[1,a]+pre[2,c]-pre[2,b]+pre[3,L]-pre[3,c]
                    if s>bst[0]: bst=(s,cuts[i],cuts[j],cuts[k])
        acc=bst[0]/L
        if acc>best[0]: best=(acc,wt,wb,*bst[1:])
        print(wt,wb,round(acc,4),bst[1:],flush=True)
    print("MELHOR",best)
    json.dump({"acc":best[0],"dano_peso_trinca":best[1],"dano_peso_buraco":best[2],"sev_min_1000":int(best[3]),"sev_media_1000":int(best[4]),"sev_alta_1000":int(best[5])},open(out,"w"))
