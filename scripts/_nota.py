"""Nota de busca exclusivamente no treino, sem referência móvel de severidade."""
import numpy as np
from pavimento.metricas import deteccao_imagem, resumir


def pearson(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 2 or a.std() == 0 or b.std() == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def nota(res, referencias):
    soma = lambda k: [sum(r[k][i] for r in res) for i in range(4)]
    ft = resumir(*soma("trinca_tol"))["f1"]
    buraco = resumir(*soma("buraco_px"))
    dt = deteccao_imagem([r["tem_trinca_pred"] for r in res], [r["tem_trinca_gt"] for r in res])["f1"]
    db = deteccao_imagem([r["tem_buraco_pred"] for r in res], [r["tem_buraco_gt"] for r in res])["f1"]
    lp = [r["quant"]["comprimento_trincas_px"] for r in res]
    bp = [r["quant"]["area_buracos_px"] for r in res]
    lr, br = [x[0] for x in referencias], [x[1] for x in referencias]
    ct, cb = pearson(lp, lr), pearson(bp, br)
    vies = abs(float(np.log((sum(lp) + 1) / (sum(lr) + 1))))
    # Buracos recebem mais peso que trincas. Severidade só em T3 congelada.
    valor = 2 * buraco["f1"] + 2 * buraco["iou"] + ft + 0.3 * (dt + db) + ct + cb - 0.5 * vies
    return valor, {"f1_trinca_tol": ft, "f1_buraco_px": buraco["f1"], "iou_buraco": buraco["iou"],
                   "f1_det_trinca": dt, "f1_det_buraco": db, "corr_comp_trinca": ct,
                   "corr_area_buraco": cb, "vies_log_comp": vies}
