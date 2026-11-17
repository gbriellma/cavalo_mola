import os, sys, numpy as np
from multiprocessing import Pool
import _caminho
from _comum import preparar
from pavimento.dataset import listar_amostras, particao
from pavimento.parametros import Parametros
from pavimento import nucleo as N
P=Parametros.carregar(_caminho.PARAMS)
def f(d):
    nome,g,roi,gt,gb=preparar(d)
    r=N.analisar(g,roi,P)
    lt=int(N.afinar(gt&roi).sum()); ab=int((gb&roi).sum())
    comps=[x['comprimento'] for x in r['regioes_trinca']]
    return lt, ab, r['quant']['comprimento_trincas_px'], r['quant']['area_buracos_px'], comps, int(roi.sum())
if __name__=='__main__':
    ds=[d for d in listar_amostras(_caminho.DATASET) if particao(os.path.basename(d))=='teste'][::3]
    with Pool(2) as p: R=p.map(f,ds)
    lt=np.array([r[0] for r in R]); ab=np.array([r[1] for r in R]); pt=np.array([r[2] for r in R]); pb=np.array([r[3] for r in R])
    def sp(a,b):
        ra=np.argsort(np.argsort(a)); rb=np.argsort(np.argsort(b)); return np.corrcoef(ra,rb)[0,1]
    print('n',len(R),'GT comp mediana',np.median(lt),'pred mediana',np.median(pt))
    print('trinca comprimento: pearson %.3f spearman %.3f'%(np.corrcoef(lt,pt)[0,1],sp(lt,pt)))
    print('buraco area: pearson %.3f spearman %.3f'%(np.corrcoef(ab,pb)[0,1],sp(ab,pb)))
    allc=np.concatenate([r[4] for r in R]); print('frag comp quantis',np.percentile(allc,[25,50,75,90]))
    for L in [10,20,30,40,60]:
        pl=np.array([sum(c for c in r[4] if c>=L) for r in R]); print('Lmin',L,'pearson %.3f spearman %.3f razao mediana %.2f'%(np.corrcoef(lt,pl)[0,1],sp(lt,pl),np.median(pl[lt>0]/lt[lt>0])))
