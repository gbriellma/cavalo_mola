import json, os, html
from pathlib import Path
RAIZ = Path(__file__).resolve().parents[1]
import numpy as np, cv2
import bench
OUT = str(RAIZ / "outputs" / "visual_pc")
os.makedirs(OUT, exist_ok=True)
nomes = json.load(open("nomes_vis.json"))
A = np.load("vis_antigo.npz"); N = np.load("vis_novo.npz")
H, W = 320, 512
un = lambda z, k, sh: np.unpackbits(z[k])[:sh[0] * sh[1]].reshape(sh).astype(bool)

def pintar(g, roi, t, b, gt=None, gb=None, titulo=""):
    im = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR).astype(np.float32)
    im[~roi] *= 0.45
    def fill(m, cor, a):
        im[m] = (1 - a) * im[m] + a * np.array(cor, np.float32)
    if gb is not None: fill(gb, (0, 215, 255), 0.45)      # buraco real: amarelo
    if gt is not None: fill(gt, (60, 220, 60), 0.9)       # trinca real: verde
    if b is not None: fill(b, (255, 200, 0), 0.35)         # buraco previsto: ciano
    if t is not None: fill(t, (40, 40, 255), 0.95)         # trinca prevista: vermelho
    im = im.astype(np.uint8)
    if b is not None:
        cs, _ = cv2.findContours(b.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE); cv2.drawContours(im, cs, -1, (255, 200, 0), 1)
    if gb is not None and b is not None:
        cs, _ = cv2.findContours(gb.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE); cv2.drawContours(im, cs, -1, (0, 215, 255), 1)
    cv2.rectangle(im, (0, 0), (im.shape[1], 22), (0, 0, 0), -1)
    cv2.putText(im, titulo, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return im

secs = []
for sec, lista in nomes.items():
    itens = []
    for n in lista:
        g, roi, gt, gb = bench.carregar(n)
        sh = g.shape; ta, ba = un(A, "t_" + n, sh), un(A, "b_" + n, sh)
        tn, bn = un(N, "t_" + n, sh), un(N, "b_" + n, sh)
        iou = lambda p, q: int((p & q).sum()) / max(int((p | q).sum()), 1)
        p0 = pintar(g, roi, None, None, gt, gb, "Rotulo real (verde=trinca, amarelo=buraco)")
        p1 = pintar(g, roi, ta, ba, None, gb, f"ANTIGO  IoU buraco {iou(ba, gb):.2f}  |  {int(ba.sum())} px prev.")
        p2 = pintar(g, roi, tn, bn, None, gb, f"NOVO    IoU buraco {iou(bn, gb):.2f}  |  {int(bn.sum())} px prev.")
        cv2.imencode(".png", np.hstack([p0, p1, p2]))[1].tofile(os.path.join(OUT, n + ".png"))
        itens.append((n, int(gb.sum()), iou(ba, gb), iou(bn, gb), iou(ta & roi, gt & roi), iou(tn & roi, gt & roi)))
    secs.append((sec, itens))

R = json.load(open(RAIZ / "outputs" / "resultados_pc.json", encoding="utf-8"))
a, d = R["anterior_teste"], R["novo_teste"]
linhas = [("Trincas: F1 por imagem", a["trinca_deteccao_imagem"]["f1"], d["trinca_deteccao_imagem"]["f1"]),
          ("Trincas: F1 pixel (tol. 2 px)", a["trinca_tolerancia_2px"]["f1"], d["trinca_tolerancia_2px"]["f1"]),
          ("Trincas: F1 pixel exato", a["trinca_pixel"]["f1"], d["trinca_pixel"]["f1"]),
          ("Trincas: Spearman do comprimento", a["quantificacao"]["comprimento_trincas_px"]["spearman"], d["quantificacao"]["comprimento_trincas_px"]["spearman"]),
          ("Buracos: F1 por imagem", a["buraco_deteccao_imagem"]["f1"], d["buraco_deteccao_imagem"]["f1"]),
          ("Buracos: F1 pixel", a["buraco_pixel"]["f1"], d["buraco_pixel"]["f1"]),
          ("Buracos: IoU pixel", a["buraco_pixel"]["iou"], d["buraco_pixel"]["iou"]),
          ("Severidade: acerto exato", a["quantificacao"]["severidade"]["acuracia"], d["quantificacao"]["severidade"]["acuracia"]),
          ("Severidade: erro de no máximo 1 nível", a["quantificacao"]["severidade"]["acuracia_ate_1_nivel"], d["quantificacao"]["severidade"]["acuracia_ate_1_nivel"])]
barras = "".join(f'<tr><td>{html.escape(n)}</td><td class="b"><div class="bar ant" style="width:{x*100:.1f}%"></div><span>{x:.3f}</span></td>'
                 f'<td class="b"><div class="bar nov" style="width:{y*100:.1f}%"></div><span>{y:.3f}</span></td></tr>' for n, x, y in linhas)
corpo = ""
for sec, itens in secs:
    corpo += f"<h2>{html.escape(sec)}</h2>"
    for n, ab, ia, inn, ca, cn in itens:
        corpo += (f'<figure><img src="{n}.png" loading="lazy"><figcaption>{n} — buraco real: {ab} px · IoU buraco {ia:.2f} → <b>{inn:.2f}</b> · '
                  f'IoU trinca {ca:.2f} → <b>{cn:.2f}</b></figcaption></figure>')
pagina = f"""<!doctype html><html lang="pt-br"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Resultados PC - Danos em Pavimento</title><style>
:root{{--bg:#fff;--fg:#1d1d1f;--mu:#6b7280;--ln:#e5e7eb;--a:#9ca3af;--n:#2563eb}}
@media(prefers-color-scheme:dark){{:root{{--bg:#111318;--fg:#eceef2;--mu:#9aa3b2;--ln:#2a2f3a;--a:#6b7280;--n:#60a5fa}}}}
body{{margin:0;background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,Segoe UI,sans-serif}}main{{max-width:1560px;margin:auto;padding:16px}}
h1{{font-size:22px}}h2{{font-size:17px;margin:28px 0 8px;border-bottom:1px solid var(--ln);padding-bottom:4px}}
table{{border-collapse:collapse;width:100%;max-width:760px}}td{{padding:4px 8px;border-bottom:1px solid var(--ln)}}td.b{{width:30%;position:relative}}
.bar{{height:14px;border-radius:3px;display:inline-block;vertical-align:middle}}.ant{{background:var(--a)}}.nov{{background:var(--n)}}td.b span{{margin-left:6px;color:var(--mu);font-size:13px}}
figure{{margin:10px 0}}img{{max-width:100%;height:auto;border:1px solid var(--ln)}}figcaption{{color:var(--mu);font-size:13px}}
.leg i{{display:inline-block;width:12px;height:12px;margin:0 4px 0 12px;vertical-align:middle}}p{{color:var(--mu);max-width:900px}}
</style></head><body><main>
<h1>Resultados no PC - teste, 666 imagens</h1>
<p>Cada painel mostra: rótulo real, código <b>antigo</b> e código <b>novo</b>, sobre a mesma imagem (512×320, pista em destaque).</p>
<p class="leg">Legenda: <i style="background:#3cdc3c"></i>trinca real <i style="background:#ffd700"></i>buraco real <i style="background:#ff2828"></i>trinca prevista <i style="background:#00c8ff"></i>buraco previsto</p>
<h2>Métricas (antigo cinza, novo azul)</h2><table>{barras}</table>
<p>Os exemplos de buracos foram escolhidos pelos melhores acertos, e as falhas e os falsos positivos estão incluídos de propósito, para não mostrar só o lado bom.</p>
{corpo}</main></body></html>"""
open(os.path.join(OUT, "index.html"), "w", encoding="utf-8").write(pagina)
print("ok", OUT)
