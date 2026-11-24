"""Widgets gráficos customizados em PyQt5 (Canvas interativo, Gauge, Histograma e Mosaico)."""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
from PyQt5.QtCore import QPointF, QRectF, Qt, pyqtSignal
from PyQt5.QtGui import (
    QBrush,
    QColor,
    QCursor,
    QFont,
    QImage,
     QPainter,
    QPainterPath,
    QPen,
    QPolygonF,
)
from PyQt5.QtWidgets import QFrame, QHBoxLayout, QLabel, QSizePolicy, QVBoxLayout, QWidget

from .tema import bgr_para_qimage


class ImageCanvasWidget(QWidget):
    """Canvas interativo com cortina Antes×Depois, Lado a Lado, Zoom/Pan, Lupa 6× e edição de ROI."""

    pixelInspecionado = pyqtSignal(int, int, bool)  # x, y, fixado
    poligonoAlterado = pyqtSignal(object)           # List[Tuple[float, float]]
    zoomAlterado = pyqtSignal(float)

    def __init__(self, familia_mono: str = "Monospace", parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setMouseTracking(True)
        self.setMinimumSize(520, 340)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._familia_mono = familia_mono

        self._qimg_antes: Optional[QImage] = None
        self._qimg_depois: Optional[QImage] = None
        self._rotulo_antes: str = "Antes"
        self._rotulo_depois: str = "Depois"

        self._modo_comparacao: str = "cortina"  # "cortina" | "unico" | "lado"
        self._pos_cortina: float = 0.50
        self._arrastando_cortina: bool = False

        self._zoom: float = 1.0
        self._pan_x: float = 0.0
        self._pan_y: float = 0.0
        self._arrastando_pan: bool = False
        self._ultimo_mouse: QPointF = QPointF()

        self._lupa_ativa: bool = False
        self._mouse_pos: Optional[QPointF] = None
        self._pixel_fixado: Optional[Tuple[int, int]] = None

        self._editando_roi: bool = False
        self._poligono_roi: List[Tuple[float, float]] = []
        self._vertice_ativo: int = -1

        self._bbox_destaque: Optional[Tuple[int, int, int, int]] = None

    # ------------------------- API pública ------------------------- #
    def definir_imagens(
        self,
        bgr_antes: np.ndarray,
        bgr_depois: np.ndarray,
        rotulo_antes: str,
        rotulo_depois: str,
    ) -> None:
        mudou_tam = (
            self._qimg_depois is None
            or self._qimg_depois.width() != bgr_depois.shape[1]
            or self._qimg_depois.height() != bgr_depois.shape[0]
        )
        self._qimg_antes = bgr_para_qimage(bgr_antes)
        self._qimg_depois = bgr_para_qimage(bgr_depois)
        self._rotulo_antes = rotulo_antes
        self._rotulo_depois = rotulo_depois
        if mudou_tam:
            self._pixel_fixado = None
            self._bbox_destaque = None
            self.resetar_zoom()
        self.update()

    def definir_modo_comparacao(self, modo: str) -> None:
        if modo in ("cortina", "unico", "lado"):
            self._modo_comparacao = modo
            self.update()

    def modo_comparacao(self) -> str:
        return self._modo_comparacao

    def definir_lupa(self, ativa: bool) -> None:
        self._lupa_ativa = ativa
        self.update()

    def definir_edicao_roi(self, ativa: bool, poligono: Optional[List[Tuple[float, float]]] = None) -> None:
        self._editando_roi = ativa
        if poligono is not None:
            self._poligono_roi = [(float(x), float(y)) for x, y in poligono]
        self._vertice_ativo = -1
        self.update()

    def obter_poligono_roi(self) -> List[Tuple[float, float]]:
        return list(self._poligono_roi)

    def destacar_bbox(self, bbox: Optional[List[int]]) -> None:
        if bbox and len(bbox) == 4:
            self._bbox_destaque = (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))
        else:
            self._bbox_destaque = None
        self.update()

    def limpar_pixel_fixado(self) -> None:
        self._pixel_fixado = None
        self.update()

    def aplicar_zoom(self, fator: float) -> None:
        novo = max(1.0, min(8.0, fator))
        self._zoom = novo
        if novo <= 1.01:
            self._pan_x = 0.0
            self._pan_y = 0.0
        self.zoomAlterado.emit(self._zoom)
        self.update()

    def resetar_zoom(self) -> None:
        self._zoom = 1.0
        self._pan_x = 0.0
        self._pan_y = 0.0
        self.zoomAlterado.emit(1.0)
        self.update()

    # --------------------- Geometria interna ----------------------- #
    def _retangulo_imagem_unica(self) -> QRectF:
        if self._qimg_depois is None:
            return QRectF()
        cw, ch = max(1.0, float(self.width())), max(1.0, float(self.height()))
        iw, ih = float(self._qimg_depois.width()), float(self._qimg_depois.height())
        margem = 10.0
        escala_base = min((cw - 2 * margem) / iw, (ch - 2 * margem) / ih)
        escala = escala_base * self._zoom
        dw, dh = iw * escala, ih * escala

        # Limita pan para manter a imagem visível
        max_px = max(0.0, (dw - cw) * 0.5 + 24.0)
        max_py = max(0.0, (dh - ch) * 0.5 + 24.0)
        self._pan_x = max(-max_px, min(max_px, self._pan_x))
        self._pan_y = max(-max_py, min(max_py, self._pan_y))

        x = (cw - dw) * 0.5 + self._pan_x
        y = (ch - dh) * 0.5 + self._pan_y
        return QRectF(x, y, dw, dh)

    def _retangulos_lado_a_lado(self) -> Tuple[QRectF, QRectF]:
        if self._qimg_depois is None:
            return QRectF(), QRectF()
        cw, ch = max(1.0, float(self.width())), max(1.0, float(self.height()))
        iw, ih = float(self._qimg_depois.width()), float(self._qimg_depois.height())
        gap = 12.0
        margem = 10.0
        metade_w = (cw - 2 * margem - gap) * 0.5
        escala = min(metade_w / iw, (ch - 2 * margem - 22.0) / ih) * self._zoom
        dw, dh = iw * escala, ih * escala
        cy = (ch - dh) * 0.5 + 8.0 + self._pan_y
        cx_esq = margem + (metade_w - dw) * 0.5 + self._pan_x
        cx_dir = margem + metade_w + gap + (metade_w - dw) * 0.5 + self._pan_x
        return QRectF(cx_esq, cy, dw, dh), QRectF(cx_dir, cy, dw, dh)

    def _tela_para_pixel(self, pt: QPointF) -> Optional[Tuple[int, int]]:
        if self._qimg_depois is None:
            return None
        iw, ih = self._qimg_depois.width(), self._qimg_depois.height()
        if self._modo_comparacao == "lado" and not self._editando_roi:
            r_esq, r_dir = self._retangulos_lado_a_lado()
            alvo = r_esq if r_esq.contains(pt) else (r_dir if r_dir.contains(pt) else None)
            if alvo is None:
                return None
        else:
            alvo = self._retangulo_imagem_unica()
            if not alvo.contains(pt):
                return None
        rx = (pt.x() - alvo.left()) / max(alvo.width(), 1e-6)
        ry = (pt.y() - alvo.top()) / max(alvo.height(), 1e-6)
        px = max(0, min(iw - 1, int(rx * iw)))
        py = max(0, min(ih - 1, int(ry * ih)))
        return px, py

    # ------------------------- Pintura ----------------------------- #
    def paintEvent(self, _event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        p.fillRect(self.rect(), QColor("#0E131A"))

        if self._qimg_depois is None or self._qimg_antes is None:
            p.setPen(QColor("#94A3B8"))
            p.drawText(self.rect(), Qt.AlignCenter, "Carregando imagem…")
            return

        suavizar = self._zoom < 2.4
        p.setRenderHint(QPainter.SmoothPixmapTransform, suavizar)

        if self._modo_comparacao == "lado" and not self._editando_roi:
            r_esq, r_dir = self._retangulos_lado_a_lado()
            p.drawImage(r_esq, self._qimg_antes)
            p.drawImage(r_dir, self._qimg_depois)
            p.setPen(QPen(QColor("#334155"), 1))
            p.setBrush(Qt.NoBrush)
            p.drawRect(r_esq)
            p.drawRect(r_dir)
            self._desenhar_badge(p, r_esq.left() + 8, r_esq.top() + 8, f"Antes · {self._rotulo_antes}", False)
            self._desenhar_badge(p, r_dir.left() + 8, r_dir.top() + 8, f"Depois · {self._rotulo_depois}", True)
            if self._bbox_destaque is not None:
                self._desenhar_bbox_em_rect(p, r_esq)
                self._desenhar_bbox_em_rect(p, r_dir)
            if self._pixel_fixado is not None:
                self._desenhar_mira_pixel(p, r_esq, *self._pixel_fixado)
                self._desenhar_mira_pixel(p, r_dir, *self._pixel_fixado)
        else:
            rect = self._retangulo_imagem_unica()
            usar_cortina = (
                self._modo_comparacao == "cortina"
                and not self._editando_roi
                and self._rotulo_antes != self._rotulo_depois
            )
            p.drawImage(rect, self._qimg_depois)

            if usar_cortina:
                cx = rect.left() + rect.width() * self._pos_cortina
                p.save()
                p.setClipRect(QRectF(rect.left(), rect.top(), cx - rect.left(), rect.height()))
                p.drawImage(rect, self._qimg_antes)
                p.restore()

                # Linha divisória e botão circular âmbar
                p.setPen(QPen(QColor("#F59E0B"), 2))
                p.drawLine(QPointF(cx, rect.top()), QPointF(cx, rect.bottom()))

                cy = rect.center().y()
                cy = max(rect.top() + 24.0, min(rect.bottom() - 24.0, cy))
                raio_knob = 14.0
                p.setPen(QPen(QColor("#111827"), 2))
                p.setBrush(QBrush(QColor("#F59E0B")))
                p.drawEllipse(QPointF(cx, cy), raio_knob, raio_knob)
                p.setPen(QColor("#111827"))
                f_knob = QFont(self.font())
                f_knob.setBold(True)
                f_knob.setPointSize(9)
                p.setFont(f_knob)
                p.drawText(
                    QRectF(cx - raio_knob, cy - raio_knob, 2 * raio_knob, 2 * raio_knob),
                    Qt.AlignCenter,
                    "⇄",
                )

                self._desenhar_badge(p, rect.left() + 10, rect.top() + 10, f"Antes · {self._rotulo_antes}", False)
                self._desenhar_badge_direita(p, rect.right() - 10, rect.top() + 10, f"Depois · {self._rotulo_depois}")

            p.setPen(QPen(QColor("#283241"), 1))
            p.setBrush(Qt.NoBrush)
            p.drawRect(rect)

            if self._bbox_destaque is not None:
                self._desenhar_bbox_em_rect(p, rect)

            if self._pixel_fixado is not None:
                self._desenhar_mira_pixel(p, rect, *self._pixel_fixado)

            if self._editando_roi and len(self._poligono_roi) >= 3:
                self._desenhar_poligono_roi(p, rect)

        if self._lupa_ativa and self._mouse_pos is not None:
            coord = self._tela_para_pixel(self._mouse_pos)
            if coord is not None:
                self._desenhar_lupa(p, self._mouse_pos, coord[0], coord[1])

    def _desenhar_badge(self, p: QPainter, x: float, y: float, texto: str, destaque: bool) -> None:
        f = QFont(self.font())
        f.setPointSize(9)
        f.setBold(True)
        p.setFont(f)
        fm = p.fontMetrics()
        w = fm.horizontalAdvance(texto) + 16
        h = 22
        r = QRectF(x, y, w, h)
        bg = QColor(15, 76, 92, 225) if destaque else QColor(15, 23, 42, 215)
        borda = QColor("#38BDF8") if destaque else QColor("#475569")
        p.setPen(QPen(borda, 1))
        p.setBrush(QBrush(bg))
        p.drawRoundedRect(r, 5, 5)
        p.setPen(QColor("#F8FAFC"))
        p.drawText(r, Qt.AlignCenter, texto)

    def _desenhar_badge_direita(self, p: QPainter, x_dir: float, y: float, texto: str) -> None:
        f = QFont(self.font())
        f.setPointSize(9)
        f.setBold(True)
        p.setFont(f)
        w = p.fontMetrics().horizontalAdvance(texto) + 16
        self._desenhar_badge(p, x_dir - w, y, texto, True)

    def _desenhar_bbox_em_rect(self, p: QPainter, rect: QRectF) -> None:
        if self._qimg_depois is None or self._bbox_destaque is None:
            return
        bx, by, bw, bh = self._bbox_destaque
        iw, ih = self._qimg_depois.width(), self._qimg_depois.height()
        sx = rect.width() / max(iw, 1)
        sy = rect.height() / max(ih, 1)
        pad = 4.0
        alvo = QRectF(
            rect.left() + bx * sx - pad,
            rect.top() + by * sy - pad,
            max(8.0, bw * sx + 2 * pad),
            max(8.0, bh * sy + 2 * pad),
        )
        p.setPen(QPen(QColor("#F59E0B"), 2.2))
        p.setBrush(QBrush(QColor(245, 158, 11, 35)))
        p.drawRoundedRect(alvo, 3, 3)

    def _desenhar_mira_pixel(self, p: QPainter, rect: QRectF, px: int, py: int) -> None:
        if self._qimg_depois is None:
            return
        iw, ih = self._qimg_depois.width(), self._qimg_depois.height()
        cx = rect.left() + (px + 0.5) * (rect.width() / max(iw, 1))
        cy = rect.top() + (py + 0.5) * (rect.height() / max(ih, 1))
        p.setPen(QPen(QColor("#F59E0B"), 1.6))
        p.setBrush(Qt.NoBrush)
        p.drawEllipse(QPointF(cx, cy), 6, 6)
        p.drawLine(QPointF(cx - 10, cy), QPointF(cx + 10, cy))
        p.drawLine(QPointF(cx, cy - 10), QPointF(cx, cy + 10))

    def _desenhar_poligono_roi(self, p: QPainter, rect: QRectF) -> None:
        pts = [
            QPointF(rect.left() + px * rect.width(), rect.top() + py * rect.height())
            for px, py in self._poligono_roi
        ]
        poly = QPolygonF(pts)
        p.setPen(QPen(QColor("#F59E0B"), 2, Qt.DashLine))
        p.setBrush(QBrush(QColor(245, 158, 11, 45)))
        p.drawPolygon(poly)
        for idx, pt in enumerate(pts):
            ativo = idx == self._vertice_ativo
            p.setPen(QPen(QColor("#111827"), 1.5))
            p.setBrush(QBrush(QColor("#FDE047" if ativo else "#F59E0B")))
            p.drawEllipse(pt, 6.5 if ativo else 5.5, 6.5 if ativo else 5.5)

    def _desenhar_lupa(self, p: QPainter, cursor: QPointF, px: int, py: int) -> None:
        if self._qimg_depois is None:
            return
        raio_px = 10  # janela 21×21 pixels ampliada em 6× = 126×126
        fator = 6
        lado = (2 * raio_px + 1) * fator

        lx = cursor.x() + 20.0
        ly = cursor.y() - lado - 20.0
        if lx + lado + 8 > self.width():
            lx = cursor.x() - lado - 20.0
        if ly < 8:
            ly = cursor.y() + 20.0

        caixa = QRectF(lx, ly, lado, lado)
        p.save()
        caminho = QPainterPath()
        caminho.addRoundedRect(caixa, 8, 8)
        p.setClipPath(caminho)
        p.fillRect(caixa, QColor("#090D12"))
        p.setRenderHint(QPainter.SmoothPixmapTransform, False)

        iw, ih = self._qimg_depois.width(), self._qimg_depois.height()
        src_x0 = px - raio_px
        src_y0 = py - raio_px
        inter_x0 = max(0, src_x0)
        inter_y0 = max(0, src_y0)
        inter_x1 = min(iw, src_x0 + 2 * raio_px + 1)
        inter_y1 = min(ih, src_y0 + 2 * raio_px + 1)
        if inter_x1 > inter_x0 and inter_y1 > inter_y0:
            sub = self._qimg_depois.copy(inter_x0, inter_y0, inter_x1 - inter_x0, inter_y1 - inter_y0)
            dst = QRectF(
                lx + (inter_x0 - src_x0) * fator,
                ly + (inter_y0 - src_y0) * fator,
                (inter_x1 - inter_x0) * fator,
                (inter_y1 - inter_y0) * fator,
            )
            p.drawImage(dst, sub)

        # Destaca o pixel central exato
        centro = QRectF(lx + raio_px * fator, ly + raio_px * fator, fator, fator)
        p.setPen(QPen(QColor("#F59E0B"), 1.5))
        p.setBrush(Qt.NoBrush)
        p.drawRect(centro)
        p.restore()

        p.setPen(QPen(QColor("#F59E0B"), 2))
        p.setBrush(Qt.NoBrush)
        p.drawRoundedRect(caixa, 8, 8)

    # ------------------------- Eventos ----------------------------- #
    def mousePressEvent(self, event) -> None:
        pos = QPointF(event.pos())
        self._ultimo_mouse = pos

        if self._editando_roi and event.button() == Qt.LeftButton:
            rect = self._retangulo_imagem_unica()
            for idx, (px, py) in enumerate(self._poligono_roi):
                vx = rect.left() + px * rect.width()
                vy = rect.top() + py * rect.height()
                if (pos.x() - vx) ** 2 + (pos.y() - vy) ** 2 <= 12.0 ** 2:
                    self._vertice_ativo = idx
                    self.update()
                    return

        if event.button() == Qt.LeftButton and self._modo_comparacao == "cortina" and not self._editando_roi:
            rect = self._retangulo_imagem_unica()
            cx = rect.left() + rect.width() * self._pos_cortina
            if abs(pos.x() - cx) <= 18.0 and rect.top() <= pos.y() <= rect.bottom():
                self._arrastando_cortina = True
                self.setCursor(QCursor(Qt.SplitHCursor))
                return

        if event.button() in (Qt.MiddleButton, Qt.RightButton) or (
            event.button() == Qt.LeftButton and self._zoom > 1.02 and not self._editando_roi
        ):
            self._arrastando_pan = True
            self.setCursor(QCursor(Qt.ClosedHandCursor))
            return

        if event.button() == Qt.LeftButton and not self._editando_roi:
            coord = self._tela_para_pixel(pos)
            if coord is not None:
                if self._pixel_fixado == coord:
                    self._pixel_fixado = None
                    self.pixelInspecionado.emit(coord[0], coord[1], False)
                else:
                    self._pixel_fixado = coord
                    self.pixelInspecionado.emit(coord[0], coord[1], True)
                self.update()

    def mouseMoveEvent(self, event) -> None:
        pos = QPointF(event.pos())
        self._mouse_pos = pos

        if self._editando_roi and self._vertice_ativo >= 0:
            rect = self._retangulo_imagem_unica()
            if rect.width() > 1 and rect.height() > 1:
                nx = max(0.0, min(1.0, (pos.x() - rect.left()) / rect.width()))
                ny = max(0.0, min(1.0, (pos.y() - rect.top()) / rect.height()))
                self._poligono_roi[self._vertice_ativo] = (round(nx, 4), round(ny, 4))
                self.poligonoAlterado.emit(self.obter_poligono_roi())
                self.update()
            return

        if self._arrastando_cortina:
            rect = self._retangulo_imagem_unica()
            if rect.width() > 1:
                self._pos_cortina = max(0.02, min(0.98, (pos.x() - rect.left()) / rect.width()))
                self.update()
            return

        if self._arrastando_pan:
            dx = pos.x() - self._ultimo_mouse.x()
            dy = pos.y() - self._ultimo_mouse.y()
            self._ultimo_mouse = pos
            self._pan_x += dx
            self._pan_y += dy
            self.update()
            return

        # Atualiza cursor ao passar perto da cortina
        if self._modo_comparacao == "cortina" and not self._editando_roi:
            rect = self._retangulo_imagem_unica()
            cx = rect.left() + rect.width() * self._pos_cortina
            if abs(pos.x() - cx) <= 14.0 and rect.top() <= pos.y() <= rect.bottom():
                self.setCursor(QCursor(Qt.SplitHCursor))
            else:
                self.setCursor(QCursor(Qt.CrossCursor))
        else:
            self.setCursor(QCursor(Qt.CrossCursor))

        coord = self._tela_para_pixel(pos)
        if coord is not None and self._pixel_fixado is None:
            self.pixelInspecionado.emit(coord[0], coord[1], False)
        if self._lupa_ativa:
            self.update()

    def mouseReleaseEvent(self, _event) -> None:
        self._arrastando_cortina = False
        self._arrastando_pan = False
        self._vertice_ativo = -1
        self.setCursor(QCursor(Qt.CrossCursor))

    def leaveEvent(self, _event) -> None:
        self._mouse_pos = None
        if self._lupa_ativa:
            self.update()

    def wheelEvent(self, event) -> None:
        delta = event.angleDelta().y()
        if delta == 0:
            return
        fator = 1.18 if delta > 0 else (1.0 / 1.18)
        self.aplicar_zoom(self._zoom * fator)


class SeverityGaugeWidget(QWidget):
    """Barra visual horizontal do Índice de Dano (0..100) com faixas de severidade."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setFixedHeight(36)
        self._indice_100: float = 0.0
        self._corte_media: float = 21.7
        self._corte_alta: float = 65.6

    def atualizar(self, indice_1000: int, sev_media_1000: int, sev_alta_1000: int) -> None:
        self._indice_100 = max(0.0, min(100.0, indice_1000 / 10.0))
        self._corte_media = max(2.0, min(90.0, sev_media_1000 / 10.0))
        self._corte_alta = max(self._corte_media + 2.0, min(98.0, sev_alta_1000 / 10.0))
        self.update()

    def paintEvent(self, _event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        w = float(self.width())
        margem = 6.0
        bw = max(10.0, w - 2 * margem)
        by = 8.0
        bh = 10.0

        # Faixas: Baixa (azul), Média (âmbar), Alta (vermelho)
        x_med = margem + bw * (self._corte_media / 100.0)
        x_alt = margem + bw * (self._corte_alta / 100.0)

        caminho = QPainterPath()
        caminho.addRoundedRect(QRectF(margem, by, bw, bh), 5, 5)
        p.save()
        p.setClipPath(caminho)
        p.fillRect(QRectF(margem, by, x_med - margem, bh), QColor("#38BDF8"))
        p.fillRect(QRectF(x_med, by, x_alt - x_med, bh), QColor("#F59E0B"))
        p.fillRect(QRectF(x_alt, by, margem + bw - x_alt, bh), QColor("#EF4444"))
        p.restore()

        p.setPen(QPen(QColor("#CBD5E1"), 1))
        p.setBrush(Qt.NoBrush)
        p.drawRoundedRect(QRectF(margem, by, bw, bh), 5, 5)

        # Marcador triangular na posição atual
        xm = margem + bw * (self._indice_100 / 100.0)
        tri = QPolygonF([QPointF(xm - 5, by - 5), QPointF(xm + 5, by - 5), QPointF(xm, by + 2)])
        p.setPen(QPen(QColor("#0F172A"), 1))
        p.setBrush(QBrush(QColor("#0F172A")))
        p.drawPolygon(tri)
        p.drawLine(QPointF(xm, by), QPointF(xm, by + bh + 1))

        # Rótulos inferiores das faixas
        f = QFont(self.font())
        f.setPointSize(8)
        p.setFont(f)
        p.setPen(QColor("#64748B"))
        p.drawText(QRectF(margem, by + bh + 2, 30, 14), Qt.AlignLeft | Qt.AlignVCenter, "0")
        p.drawText(QRectF(x_med - 18, by + bh + 2, 36, 14), Qt.AlignCenter, f"{self._corte_media:.0f}")
        p.drawText(QRectF(x_alt - 18, by + bh + 2, 36, 14), Qt.AlignCenter, f"{self._corte_alta:.0f}")
        p.drawText(QRectF(margem + bw - 30, by + bh + 2, 30, 14), Qt.AlignRight | Qt.AlignVCenter, "100")


class TextureHistogramWidget(QWidget):
    """Gráfico vetorial compacto do histograma de textura s_b na ROI (Etapa 3B)."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setFixedHeight(108)
        self._bins: List[int] = []
        self._mediana: int = 0
        self._limiar: int = 0

    def atualizar(self, bins: List[int], mediana: int, limiar: int) -> None:
        self._bins = bins or []
        self._mediana = mediana
        self._limiar = limiar
        self.update()

    def paintEvent(self, _event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        r = QRectF(0, 0, self.width(), self.height())
        p.setPen(QPen(QColor("#E2E8F0"), 1))
        p.setBrush(QBrush(QColor("#F8FAFC")))
        p.drawRoundedRect(r.adjusted(0.5, 0.5, -0.5, -0.5), 6, 6)

        if not self._bins:
            return

        pad_l, pad_r, pad_t, pad_b = 10.0, 10.0, 22.0, 18.0
        gw = max(10.0, self.width() - pad_l - pad_r)
        gh = max(10.0, self.height() - pad_t - pad_b)
        n = len(self._bins)
        topo = max(max(self._bins), 1)
        bw = gw / max(n, 1)

        for i, val in enumerate(self._bins):
            if val <= 0:
                continue
            # Escala raiz suave para evidenciar a cauda de buracos (<1% da pista)
            frac = (val / topo) ** 0.55
            bh = max(1.5, frac * gh)
            x = pad_l + i * bw
            y = pad_t + gh - bh
            cor = QColor("#F59E0B") if i > self._limiar else QColor("#64748B")
            p.fillRect(QRectF(x, y, max(1.0, bw - 0.6), bh), cor)

        # Linhas de Mediana e Limiar t_tex
        xm = pad_l + min(n - 1, self._mediana) * bw + bw * 0.5
        xt = pad_l + min(n - 1, self._limiar) * bw + bw * 0.5
        p.setPen(QPen(QColor("#0F4C5C"), 1.6, Qt.DashLine))
        p.drawLine(QPointF(xm, pad_t - 2), QPointF(xm, pad_t + gh))
        p.setPen(QPen(QColor("#DC2626"), 1.8, Qt.DashLine))
        p.drawLine(QPointF(xt, pad_t - 2), QPointF(xt, pad_t + gh))

        f = QFont(self.font())
        f.setPointSize(8)
        f.setBold(True)
        p.setFont(f)
        p.setPen(QColor("#0F4C5C"))
        p.drawText(QRectF(pad_l, 4, gw * 0.55, 14), Qt.AlignLeft | Qt.AlignVCenter, f"Mediana = {self._mediana}")
        p.setPen(QColor("#DC2626"))
        p.drawText(QRectF(pad_l + gw * 0.45, 4, gw * 0.55, 14), Qt.AlignRight | Qt.AlignVCenter, f"Corte t_tex = {self._limiar}")

        f.setBold(False)
        p.setFont(f)
        p.setPen(QColor("#64748B"))
        p.drawText(QRectF(pad_l, pad_t + gh + 2, gw, 14), Qt.AlignCenter, "Histograma da rugosidade local s_b na ROI")


class _MiniaturaImagemWidget(QWidget):
    """Renderiza uma QImage mantendo a proporção exata dentro do card do Mosaico."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._qimg: Optional[QImage] = None
        self.setMinimumHeight(148)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def definir_qimage(self, qimg: QImage) -> None:
        self._qimg = qimg
        self.update()

    def paintEvent(self, _event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        p.setRenderHint(QPainter.SmoothPixmapTransform, True)
        r = QRectF(0, 0, self.width(), self.height())
        caminho = QPainterPath()
        caminho.addRoundedRect(r, 6, 6)
        p.save()
        p.setClipPath(caminho)
        p.fillRect(r, QColor("#0E131A"))
        if self._qimg is not None:
            iw, ih = float(self._qimg.width()), float(self._qimg.height())
            esc = min((r.width() - 6) / iw, (r.height() - 6) / ih)
            dw, dh = iw * esc, ih * esc
            alvo = QRectF((r.width() - dw) * 0.5, (r.height() - dh) * 0.5, dw, dh)
            p.drawImage(alvo, self._qimg)
        p.restore()


class MosaicTileWidget(QFrame):
    """Card clicável de uma etapa no Painel Panorâmico (Mosaico)."""

    clicado = pyqtSignal(str, str)  # etapa_id, chave_img

    def __init__(
        self,
        etapa_id: str,
        chave_img: str,
        codigo: str,
        titulo: str,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._etapa_id = etapa_id
        self._chave_img = chave_img
        self.setProperty("card", True)
        self.setCursor(QCursor(Qt.PointingHandCursor))

        lay = QVBoxLayout(self)
        lay.setContentsMargins(10, 8, 10, 8)
        lay.setSpacing(6)

        topo = QHBoxLayout()
        topo.setSpacing(6)
        badge = QLabel(codigo)
        eh_ambar = codigo in ("3B", "04")
        bg_b = "#FEF3C7" if eh_ambar else "#E6F3F6"
        fg_b = "#92400E" if eh_ambar else "#0F4C5C"
        badge.setStyleSheet(
            f"background:{bg_b}; color:{fg_b}; font-weight:800; font-size:10px; "
            "padding:2px 6px; border-radius:4px;"
        )
        lbl_titulo = QLabel(titulo)
        lbl_titulo.setStyleSheet("font-weight:700; font-size:12px; color:#0F172A;")
        self._lbl_metrica = QLabel("")
        self._lbl_metrica.setStyleSheet("font-size:11px; color:#64748B; font-weight:600;")

        topo.addWidget(badge)
        topo.addWidget(lbl_titulo, 1)
        topo.addWidget(self._lbl_metrica)
        lay.addLayout(topo)

        self._mini = _MiniaturaImagemWidget(self)
        lay.addWidget(self._mini, 1)

    def atualizar(self, img_bgr: np.ndarray, metrica_curta: str = "") -> None:
        self._mini.definir_qimage(bgr_para_qimage(img_bgr))
        self._lbl_metrica.setText(metrica_curta)

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.LeftButton:
            self.clicado.emit(self._etapa_id, self._chave_img)
        super().mousePressEvent(event)
