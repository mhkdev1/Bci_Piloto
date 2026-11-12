# -*- coding: utf-8 -*-
"""
gerar_apresentacao.py — monta o PDF de aula "Do pensamento ao movimento".

    python apresentacao/gerar_apresentacao.py            # gera o PDF
    python apresentacao/gerar_apresentacao.py --png DIR  # e também uma imagem por página

Tudo que aparece como medida sai dos arquivos deste projeto: os EDFs em dados/physionet,
as gravações da Cyton em dados/gravacoes e os pesos de modelos/cnn2d_transicao.pt. Os
desenhos de anatomia e os diagramas são esquemas, e estão marcados como esquema na página.
"""
from __future__ import annotations

import argparse
import os
import sys
import textwrap
import warnings

import numpy as np

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RAIZ)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Circle, Ellipse, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle

import config as cfg
from nucleo import dados, modelo, preproc

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------- estilo
FUNDO = "#fcfcfb"
TINTA = "#0b0b0b"
TINTA2 = "#52514e"
APAGADO = "#8a8984"
GRADE = "#e3e2dc"
PAINEL = "#f1f0ec"
AZUL, LARANJA, VERDE = "#2a78d6", "#eb6834", "#1baf7a"      # C3, C4, Cz
VIOLETA, AMARELO = "#6250d6", "#eda100"
COR_CANAL = {"C3": AZUL, "C4": LARANJA, "Cz": VERDE}
FAIXA = {"esq": "#e4e0f8", "dir": "#fbeecb"}                 # fundo das dicas de mão
SEQ = LinearSegmentedColormap.from_list("seq", ["#f3f6fb", "#9ec0ea", AZUL, "#123f7a"])
DIV = LinearSegmentedColormap.from_list("div", [LARANJA, "#f3c9b6", "#e9e8e4", "#b5cdee", AZUL])

plt.rcParams.update({
    "font.family": ["Segoe UI", "DejaVu Sans"],
    "font.size": 12,
    "axes.edgecolor": APAGADO, "axes.linewidth": 0.8,
    "axes.labelcolor": TINTA2, "axes.labelsize": 12,
    "xtick.color": TINTA2, "ytick.color": TINTA2,
    "xtick.labelsize": 11, "ytick.labelsize": 11,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.facecolor": FUNDO, "figure.facecolor": FUNDO,
    "pdf.fonttype": 42,
})

ETAPAS = ["Cérebro", "Eletrodos", "Placa", "Limpeza", "Rede", "Decisão", "Órtese"]
LARG, ALT = 13.333, 7.5
_paginas = []


def nova(etapa, titulo, sub=None, ativa=None):
    """Página em branco com cabeçalho. `ativa` acende um ponto da trilha no alto à direita."""
    fig = plt.figure(figsize=(LARG, ALT))
    fig.text(0.05, 0.925, etapa.upper(), size=11, color=AZUL, weight="bold")
    fig.text(0.05, 0.858, titulo, size=27, color=TINTA, weight="bold")
    if sub:
        fig.text(0.05, 0.815, textwrap.fill(sub, 118), size=14.5, color=TINTA2, va="top",
                 linespacing=1.45)
    if ativa is not None:
        x0, dx = 0.615, 0.055
        fig.add_artist(plt.Line2D([x0, x0 + dx * (len(ETAPAS) - 1)], [0.945, 0.945],
                                  color=GRADE, lw=1.5, zorder=1))
        for i, nome in enumerate(ETAPAS):
            on = i == ativa
            fig.add_artist(Circle((x0 + dx * i, 0.945), 0.0065 if on else 0.004,
                                  transform=fig.transFigure, zorder=3,
                                  facecolor=AZUL if on else "#c9c8c1", edgecolor="none"))
            fig.text(x0 + dx * i, 0.915, nome, size=8.5, ha="center",
                     color=TINTA if on else APAGADO, weight="bold" if on else "normal")
    fig.text(0.05, 0.035, "BCI de imagética motora para órtese de mão", size=9, color=APAGADO)
    _paginas.append(fig)
    fig.text(0.95, 0.035, str(len(_paginas)), size=9, color=APAGADO, ha="right")
    return fig


def paragrafo(fig, x, y, txt, larg=52, size=14, cor=TINTA2, peso="normal"):
    """Escreve um bloco de texto e devolve o y logo abaixo dele."""
    linhas = textwrap.wrap(txt, larg)
    fig.text(x, y, "\n".join(linhas), size=size, color=cor, va="top", linespacing=1.45,
             weight=peso)
    return y - len(linhas) * size * 1.45 / (ALT * 72) - 0.022


def topicos(fig, x, y, itens, larg=50, size=14):
    """Lista com marcador. Cada item é um texto ou um par (título, texto)."""
    for it in itens:
        tit, txt = it if isinstance(it, tuple) else (None, it)
        fig.add_artist(Rectangle((x, y - 0.019), 0.006, 0.0105, transform=fig.transFigure,
                                 facecolor=AZUL, edgecolor="none"))
        if tit:
            fig.text(x + 0.016, y, tit, size=size, color=TINTA, va="top", weight="bold")
            y -= size * 1.5 / (ALT * 72)
        y = paragrafo(fig, x + 0.016, y, txt, larg, size)
    return y


def quadro(fig, pos):
    """Eixo sem escala, em coordenadas 0-100 na horizontal, para desenhar esquemas."""
    ax = fig.add_axes(pos)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100 * pos[3] * ALT / (pos[2] * LARG))
    ax.set_aspect("equal")
    ax.axis("off")
    return ax


def caixa(ax, x, y, w, h, titulo, texto=None, cor=PAINEL, borda=None, size=12, cor_titulo=TINTA):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.2",
                                facecolor=cor, edgecolor=borda or "none", lw=1.4))
    if texto:
        ax.text(x + w / 2, y + h - 1.6, titulo, ha="center", va="top", size=size,
                weight="bold", color=cor_titulo)
        ax.text(x + w / 2, y + h - 1.6 - size * 0.42, texto, ha="center", va="top",
                size=size - 2, color=TINTA2, linespacing=1.4)
    else:
        ax.text(x + w / 2, y + h / 2, titulo, ha="center", va="center", size=size,
                weight="bold", color=cor_titulo, linespacing=1.35)


def seta(ax, a, b, cor=APAGADO, lw=1.8, estilo="-|>", curva=0.0):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle=estilo, mutation_scale=16, color=cor, lw=lw,
                                 connectionstyle=f"arc3,rad={curva}", shrinkA=0, shrinkB=0))


def limpar(ax, grade="y"):
    ax.grid(axis=grade, color=GRADE, lw=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)


def empilhar(ax, t, X, passo, nomes=cfg.CANAIS, lw=1.0):
    """Três canais um sobre o outro, cada um com o nome à esquerda."""
    for i, nome in enumerate(nomes):
        desl = -i * passo
        ax.plot(t, X[i] - np.median(X[i]) + desl, color=COR_CANAL[nome], lw=lw)
        ax.text(t[0] - (t[-1] - t[0]) * 0.012, desl, nome, ha="right", va="center",
                size=13, weight="bold", color=TINTA)
    ax.set_ylim(-(len(nomes) - 0.45) * passo, 0.6 * passo)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlim(t[0], t[-1])
    ax.tick_params(length=0)


def regua(ax, x, y, valor, unidade="µV"):
    """Barra vertical de escala, no lugar do eixo y."""
    ax.plot([x, x], [y, y + valor], color=TINTA, lw=2, solid_capstyle="butt", clip_on=False)
    ax.text(x, y + valor / 2, f"  {valor:g} {unidade}", va="center", ha="left", size=11,
            color=TINTA2, clip_on=False)


def faixas(ax, eventos, rotulo=True):
    """Pinta o fundo nos trechos de dica de mão. eventos = [(t0, t1, 'esq'|'dir'|None)]."""
    nome = {"esq": "mão esquerda", "dir": "mão direita"}
    lim = ax.get_xlim()
    for t0, t1, k in eventos:
        if k is None or t1 < lim[0] or t0 > lim[1]:
            continue
        a, b = max(t0, lim[0]), min(t1, lim[1])
        ax.axvspan(a, b, color=FAIXA[k], lw=0, zorder=0)
        if rotulo and b - a > 0.5 * (t1 - t0):
            ax.text((a + b) / 2, 1.01, nome[k], transform=ax.get_xaxis_transform(),
                    ha="center", va="bottom", size=11, color=TINTA2)


# ---------------------------------------------------------------- dados
def carregar_tudo():
    import mne
    import pandas as pd
    import torch
    from scipy.signal import butter, hilbert, sosfiltfilt, welch
    mne.set_log_level("ERROR")
    D = {}
    suj = dados.sujeitos_physionet()

    # --- PhysioNet: um run de imagética do S001, bruto
    raw = mne.io.read_raw_edf(suj["S001"][4], preload=True)
    mne.datasets.eegbci.standardize(raw)
    raw.pick_channels(list(cfg.CANAIS), ordered=True)
    fs = float(raw.info["sfreq"])
    ev, _ = mne.events_from_annotations(raw, event_id={"T0": 3, "T1": 1, "T2": 2})
    t_ev = ev[:, 0] / fs
    fim = np.append(t_ev[1:], raw.times[-1])
    D["edf"] = {"fs": fs, "x": raw.get_data() * 1e6,
                "ev": [(a, b, {1: "esq", 2: "dir", 3: None}[c])
                       for a, b, c in zip(t_ev, fim, ev[:, 2])]}

    # --- Cyton: sessão S004_R1, bruta
    pasta = os.path.join(cfg.DIR_GRAVACOES, "S004_R1")
    x = pd.read_csv(os.path.join(pasta, "continuo.csv"))[
        [f"{c}_uV" for c in cfg.CANAIS]].to_numpy(float).T
    e = pd.read_csv(os.path.join(pasta, "eventos.csv"))
    ini = e["amostra_inicio"].to_numpy() / cfg.TAXA_PLACA
    fim = np.append(ini[1:], x.shape[1] / cfg.TAXA_PLACA)
    D["cyton"] = {"fs": float(cfg.TAXA_PLACA), "x": x,
                  "ev": [(a, b, {"LH": "esq", "RH": "dir"}.get(c))
                         for a, b, c in zip(ini, fim, e["code"])]}

    # --- as etapas da limpeza, na sessão inteira da Cyton (o FIR de 0,5 Hz é longo)
    fsc = D["cyton"]["fs"]
    v = x * 1e-6
    notch = mne.filter.notch_filter(v, fsc, cfg.NOTCH, verbose=False)
    banda = mne.filter.filter_data(notch, fsc, *cfg.PASSA_BANDA, fir_design="firwin",
                                   verbose=False)
    reamostrado = mne.filter.resample(banda, down=fsc / cfg.TAXA_IA, npad="auto", verbose=False)
    f0, p0 = welch(x, fsc, nperseg=1024)
    f1, p1 = welch(banda * 1e6, fsc, nperseg=1024)
    D["limpeza"] = {"bruto": x, "notch": notch * 1e6, "banda": banda * 1e6,
                    "final": reamostrado * 1e6, "f0": f0, "p0": p0, "f1": f1, "p1": p1}

    # --- ritmo mu (8-13 Hz) em torno do início do movimento da mão direita, 10 sujeitos
    acc = []
    for _, runs in suj.items():
        for r in (3, 7, 11):
            raw = mne.io.read_raw_edf(runs[r], preload=True)
            mne.datasets.eegbci.standardize(raw)
            raw.pick_channels(list(cfg.CANAIS), ordered=True)
            fse = float(raw.info["sfreq"])
            sos = butter(4, [8, 13], btype="band", fs=fse, output="sos")
            pot = np.abs(hilbert(sosfiltfilt(sos, raw.get_data(), axis=1), axis=1)) ** 2
            evr, _ = mne.events_from_annotations(raw, event_id={"T0": 3, "T1": 1, "T2": 2})
            a, b = int(-3 * fse), int(5 * fse)
            for s, _, c in evr:
                if c == 2 and s + a >= 0 and s + b < pot.shape[1]:
                    seg = pot[:, s + a:s + b]
                    acc.append(seg / seg.mean(axis=1, keepdims=True))   # tira a escala do trial
    med = np.mean(acc, axis=0)
    # referência: de 1,5 a 0,25 s antes da dica, medida na média (e não trial a trial, que
    # dividiria por um número ruidoso e puxaria a média para cima)
    ref = med[:, int(1.5 * fse):int(2.75 * fse)].mean(axis=1, keepdims=True)
    med = (med / ref - 1) * 100
    nucleo = np.ones(int(0.5 * fse)) / int(0.5 * fse)
    med = np.array([np.convolve(c, nucleo, mode="same") for c in med])
    t = np.arange(med.shape[1]) / fse - 3
    j = (t > 0.5) & (t < 3.5)
    D["mu"] = {"t": t, "m": med, "n": len(acc), "sujeitos": len(suj),
               "c3": med[0, j].mean(), "c4": med[1, j].mean()}

    # --- a rede e uma janela real de mão direita atravessando cada camada
    m, amostras, _ = modelo.carregar(cfg.MODELO_PADRAO)
    X, y, tj = dados.janelas_edf(suj["S001"][4])
    Xn = preproc.normalizar_com_modelo(m, X)
    with torch.no_grad():
        prob = torch.softmax(m(torch.tensor(Xn)), 1).numpy()
        certos = np.where((y == 2) & (prob.argmax(1) == 2))[0]
        i = certos[np.argmax(prob[certos, 2])]
        h = torch.tensor(Xn[i:i + 1]).unsqueeze(1)
        etapas = {}
        for k, camada in enumerate(m.bloco):
            h = camada(h)
            etapas[k] = h[0].numpy().copy()
        etapas["pool"] = m.pool(h)[0].numpy()
    D["rede"] = {"m": m, "x": Xn[i], "bruta": X[i], "prob": prob[i], "etapas": etapas,
                 "kernels": [r.weight.detach().numpy()[:, 0, 0, :] for r in m.bloco[0].ramos],
                 "espacial": m.bloco[2].weight.detach().numpy()[:, 0, :, 0],
                 "n_param": sum(p.numel() for p in m.parameters()),
                 "X": Xn, "y": y, "t": tj}

    # --- matriz de confusão do S001 (os seis runs)
    conf = np.zeros((3, 3), int)
    for r in cfg.RUNS_EDF:
        Xr, yr, _ = dados.janelas_edf(suj["S001"][r])
        with torch.no_grad():
            pr = m(torch.tensor(preproc.normalizar_com_modelo(m, Xr))).argmax(1).numpy()
        for a, b in zip(yr, pr):
            conf[a, b] += 1
    D["conf"] = conf

    # --- sessão de tempo real com a órtese (S012_RT4)
    pasta = os.path.join(cfg.DIR_GRAVACOES, "S012_RT4")
    inf = pd.read_csv(os.path.join(pasta, "inferencias.csv"))
    e = pd.read_csv(os.path.join(pasta, "eventos.csv"))
    ini = e["t_s"].to_numpy()
    fim = np.append(ini[1:], inf["t_s"].max())
    D["rt"] = {"inf": inf, "ev": [(a, b, "dir" if c == "RH" else None)
                                  for a, b, c in zip(ini, fim, e["code"])]}
    return D


MU_TEXTO = ("Nos nossos dados o ritmo mu cai quando a mão direita se move, só que cai quase igual dos "
            "dois lados: {c3:.0f}% em C3 e {c4:.0f}% em C4. A queda avisa que houve movimento. "
            "Dizer qual das mãos, com três eletrodos, é bem mais difícil, e é por isso que o "
            "projeto entrega o sinal a uma rede neural.")

# ---------------------------------------------------------------- páginas
def p_capa(D):
    fig = plt.figure(figsize=(LARG, ALT))
    _paginas.append(fig)
    fig.text(0.07, 0.86, "MATERIAL DE AULA", size=12, color=AZUL, weight="bold")
    fig.text(0.07, 0.82, "Do pensamento\nao movimento", size=54, color=TINTA, weight="bold",
             linespacing=1.05, va="top")
    fig.text(0.07, 0.535, textwrap.fill(
        "Como três eletrodos, uma rede convolucional pequena e um servo fazem uma órtese "
        "fechar a mão de quem só pensou em fechá-la.", 62),
        size=18, color=TINTA2, va="top", linespacing=1.45)
    ax = fig.add_axes([0.07, 0.09, 0.86, 0.26])
    c = D["cyton"]
    a, b = int(104 * c["fs"]), int(116 * c["fs"])
    seg = D["limpeza"]["banda"][:, a:b]
    t = np.arange(seg.shape[1]) / c["fs"]
    empilhar(ax, t, seg, 110, lw=1.1)
    ax.set_xticks([])
    ax.spines["bottom"].set_visible(False)
    fig.text(0.93, 0.06, "12 segundos de EEG gravados com o capacete do projeto (sessão S004_R1), já filtrados",
             size=10, color=APAGADO, ha="right")


def p_visao(D):
    fig = nova("Visão geral", "O caminho inteiro em uma página",
               "Sete etapas separam a intenção do movimento. As três primeiras acontecem no corpo "
               "e no capacete, as três seguintes dentro do computador e a última na órtese.")
    ax = quadro(fig, [0.04, 0.10, 0.92, 0.60])
    blocos = [
        ("1  Cérebro", "a pessoa imagina\nfechar a mão\ndireita", "#e3edfa"),
        ("2  Eletrodos", "C3, C4 e Cz\ncaptam alguns\nmicrovolts", "#e3edfa"),
        ("3  Placa Cyton", "amplifica e mede\n250 vezes\npor segundo", "#e3edfa"),
        ("4  Limpeza", "filtra, reduz para\n128 Hz e corta\njanelas de 4,1 s", PAINEL),
        ("5  Rede neural", f"{D['rede']['n_param']:,}".replace(",", ".") + " pesos\ndevolvem três\nprobabilidades", PAINEL),
        ("6  Decisor", "fecha se a chance\nde mão direita\npassa de 70%", PAINEL),
        ("7  Órtese", "o ESP32 gira o\nservo um grau\npor comando", "#fde9df"),
    ]
    w, h, gap = 12.2, 19, 2.2
    y = 15
    for i, (tit, txt, cor) in enumerate(blocos):
        x = 0.4 + i * (w + gap)
        caixa(ax, x, y, w, h, tit, txt, cor=cor, size=14)
        if i:
            seta(ax, (x - gap + 0.2, y + h / 2), (x - 0.2, y + h / 2))
    for x0, n, nome in ((0.4, 3, "no corpo e no capacete"), (0.4 + 3 * (w + gap), 3, "no computador (Python)"),
                        (0.4 + 6 * (w + gap), 1, "na mão")):
        x1 = x0 + n * w + (n - 1) * gap
        ax.plot([x0, x0, x1, x1], [y - 3, y - 5, y - 5, y - 3], color=APAGADO, lw=1.2)
        ax.text((x0 + x1) / 2, y - 7.5, nome, ha="center", va="top", size=12.5, color=TINTA2)
    fig.text(0.05, 0.12, "Do início da intenção ao servo girando passam cerca de 2 segundos. "
             "Quase todo esse tempo é a espera para a janela de sinal encher.",
             size=14, color=TINTA)


# ---- figuras de terceiros, todas de licença livre com atribuição (ver apresentacao/img/CREDITOS.txt)
DIR_IMG = os.path.join(RAIZ, "apresentacao", "img")
# Rótulos em inglês do cérebro da Blausen: (caixa a cobrir x0,y0,x1,y1), texto, x, y, alinhamento.
# Coordenadas em pixels da imagem original (1600 x 960).
CEREBRO_ROTULOS = [
    ((140, 68, 390, 110), "Córtex motor primário", 384, 90, "right"),
    ((4, 212, 176, 252), "Sulco central", 170, 232, "right"),
    ((4, 548, 184, 590), "Sulco lateral", 178, 570, "right"),
    ((170, 818, 444, 860), "Córtex auditivo", 438, 838, "right"),
    ((1316, 66, 1600, 134), "Córtex sensorial\nprimário", 1322, 96, "left"),
    ((1382, 322, 1600, 362), "Sulco calcarino", 1389, 342, "left"),
    ((1418, 578, 1600, 642), "Córtex visual", 1424, 598, "left"),
]
CEREBRO_HEMISFERIOS = [("Hemisfério direito", 1120, 254), ("Hemisfério esquerdo", 905, 395)]


def _figura(nome, corte=None):
    """Lê uma imagem de apresentacao/img e troca o branco dela pelo fundo da página."""
    from PIL import Image
    im = Image.open(os.path.join(DIR_IMG, nome)).convert("RGB")
    if corte:
        im = im.crop(corte)
    im = np.asarray(im).copy()
    fundo = np.array([int(FUNDO[i:i + 2], 16) for i in (1, 3, 5)], dtype=np.uint8)
    im[(im > 246).all(axis=2)] = fundo
    return im, fundo


def p_comando(D):
    fig = nova("1 · O cérebro", "O comando nasce no córtex motor", ativa=0)
    im, fundo = _figura("cerebro_cortex_motor.png")
    for (x0, y0, x1, y1), *_ in CEREBRO_ROTULOS:
        im[y0:y1, x0:x1] = fundo
    ax = fig.add_axes([0.035, 0.115, 0.575, 0.70])
    ax.imshow(im, extent=[0, im.shape[1], im.shape[0], 0], interpolation="lanczos")
    ax.set_xlim(-10, im.shape[1] + 60)
    ax.set_ylim(im.shape[0], 0)
    ax.set_aspect("equal")
    ax.axis("off")
    for _, txt, x, y, lado in CEREBRO_ROTULOS:
        ax.text(x, y, txt, ha=lado, va="center", size=11.5, color=TINTA, linespacing=1.25,
                weight="bold" if "motor" in txt else "normal", clip_on=False)
    for txt, x, y in CEREBRO_HEMISFERIOS:
        ax.text(x, y, txt, ha="center", va="center", size=11, weight="bold", color=TINTA,
                bbox=dict(boxstyle="round,pad=0.25", facecolor="#f6e9e2", edgecolor="none", alpha=0.93))

    foto, _ = _figura("homunculo_motor.jpg", corte=(880, 400, 1900, 1215))
    # no canto de cima sobra um pedaço do texto da vitrine do museu: cobre com a cor do vidro
    canto = foto[:125, :300]
    canto[:] = np.median(canto.reshape(-1, 3), axis=0).astype(np.uint8)
    ax = fig.add_axes([0.645, 0.515, 0.305, 0.29])
    ax.imshow(foto, interpolation="lanczos")
    ax.set_anchor("W")
    ax.axis("off")
    y = paragrafo(fig, 0.645, 0.495,
                  "Homúnculo motor: o corpo como ele seria se cada parte tivesse o tamanho do pedaço "
                  "de córtex que a controla. As mãos ficam enormes.", larg=50, size=11.5, cor=TINTA)
    topicos(fig, 0.645, y - 0.012, [
        ("O lado é trocado",
         "Cada hemisfério comanda o lado oposto do corpo. A mão direita é assunto do córtex "
         "motor esquerdo, a faixa verde da figura."),
        ("O que sobra quando o braço não responde",
         "Em muitas lesões o caminho até o músculo se rompe, mas o córtex continua preparando o "
         "comando. O projeto lê essa preparação."),
    ], larg=44, size=12.5)
    fig.text(0.05, 0.068, "Figuras: cérebro de Blausen.com staff, Medical gallery of Blausen Medical 2014 "
             "(CC BY 3.0), com rótulos traduzidos; homúnculo, foto de Dr. Joe Kiff (CC BY-SA 3.0), "
             "recortada. Ambas via Wikimedia Commons.", size=8.5, color=APAGADO)


def p_neuronio(D):
    fig = nova("1 · O cérebro", "Como um neurônio vira sinal elétrico", ativa=0)
    im, _ = _figura("origem_do_eeg.jpg")
    ax = fig.add_axes([0.04, 0.215, 0.665, 0.60])
    ax.imshow(im, interpolation="lanczos")
    ax.set_anchor("NW")
    ax.axis("off")
    fig.text(0.05, 0.185, textwrap.fill(
        "O eletrodo só percebe quando dezenas de milhares de neurônios alinhados ficam ativos ao "
        "mesmo tempo. Depois de atravessar osso e pele, sobram de 10 a 100 microvolts.", 100),
        size=13.5, color=TINTA, va="top", linespacing=1.45)
    topicos(fig, 0.735, 0.80, [
        ("a  Um neurônio piramidal",
         "O sinal chega nos dendritos de cima (apical dendrites). A corrente circula em volta "
         "da célula (current flow) e ela vira um pequeno dipolo."),
        ("b  A sinapse",
         "Um neurônio solta o neurotransmissor (NT) e abre canais no neurônio seguinte."),
        ("c  O córtex em camadas",
         "Milhares de neurônios lado a lado, em seis camadas (I a VI). Em cima, dois eletrodos "
         "e o sinal de EEG, que é a soma de todos."),
        ("d, e  O caminho até o eletrodo",
         "Meninges, osso do crânio (skull) e couro cabeludo (scalp) espalham e enfraquecem o "
         "sinal antes de ele chegar ao eletrodo, na superfície da cabeça."),
    ], larg=34, size=11.5)
    fig.text(0.05, 0.068, "Figura: Portillo-Lara et al., \"Mind the gap: state-of-the-art technologies and "
             "applications for EEG-based brain-computer interfaces\", APL Bioengineering 5, 031507 (2021), "
             "CC BY 4.0.", size=8.5, color=APAGADO)


def p_imagetica(D):
    mu = D["mu"]
    fig = nova("1 · O cérebro", "Imaginar o movimento também mexe no sinal",
               "Em repouso, o córtex motor oscila perto de 10 Hz (o ritmo mu). Ao mover a mão, ou só "
               "ao imaginar o movimento, essa oscilação enfraquece, principalmente do lado oposto à mão.",
               ativa=0)
    # esquema do livro
    ax = fig.add_axes([0.06, 0.225, 0.40, 0.385])
    t = np.linspace(-3, 5, 2000)
    env = 1 - 0.62 / (1 + np.exp(-(t - 0.4) * 5)) + 0.62 / (1 + np.exp(-(t - 4.2) * 5))
    ax.plot(t, env * np.sin(2 * np.pi * 10 * t), color=AZUL, lw=0.9)
    ax.plot(t, env, color=TINTA, lw=1.6)
    ax.plot(t, -env, color=TINTA, lw=1.6)
    ax.axvspan(0, 4.1, color=FAIXA["dir"], lw=0, zorder=0)
    ax.text(2.05, 1.18, "imaginando a mão direita", ha="center", size=11.5, color=TINTA2)
    ax.set_xlim(-3, 5)
    ax.set_ylim(-1.3, 1.35)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("tempo (s), zero no início da tarefa")
    ax.tick_params(length=0)
    fig.text(0.06, 0.665, "O que os livros descrevem (esquema)", size=14, weight="bold", color=TINTA)
    fig.text(0.06, 0.632, "ritmo mu sobre C3, o lado oposto à mão", size=11, color=TINTA2)

    # medida nos dados do projeto
    ax = fig.add_axes([0.56, 0.225, 0.31, 0.385])
    ax.set_xlim(-2.4, 4.4)
    ax.axvspan(0, 4.1, color=FAIXA["dir"], lw=0, zorder=0)
    ax.axhline(0, color=APAGADO, lw=1)
    k = np.searchsorted(mu["t"], 4.4) - 1
    for i, nome in enumerate(("C3", "C4")):
        ax.plot(mu["t"], mu["m"][i], color=COR_CANAL[nome], lw=2.2)
    ordem = np.argsort([mu["m"][0][k], mu["m"][1][k]])
    faixa_y = np.ptp(ax.get_ylim())
    for pos, i in zip((-1, 1), ordem):
        txt = ("C3, lado\noposto à mão", "C4, mesmo\nlado da mão")[i]
        ax.text(4.5, mu["m"][i][k] + pos * faixa_y * 0.05, txt, size=11.5, color=TINTA,
                va="center", clip_on=False, linespacing=1.3)
    ax.set_ylabel("potência do ritmo mu, variação em %")
    ax.set_xlabel("tempo (s), zero no início do movimento")
    limpar(ax)
    fig.text(0.56, 0.665, "O que os nossos dados mostram (medido)", size=14, weight="bold", color=TINTA)
    fig.text(0.56, 0.632, f"mão direita, {mu['n']} movimentos de {mu['sujeitos']} voluntários do PhysioNet",
             size=11, color=TINTA2)
    print(f"   ritmo mu: C3 {mu['c3']:+.1f}%  C4 {mu['c4']:+.1f}%")
    fig.text(0.06, 0.075, textwrap.fill(MU_TEXTO.format(c3=abs(mu["c3"]), c4=abs(mu["c4"]),
                                                       dif=abs(mu["c3"] - mu["c4"])), 150),
             size=13, color=TINTA, va="bottom", linespacing=1.45)


def p_eletrodos(D):
    fig = nova("2 · Os eletrodos", "Onde o capacete encosta", ativa=1)
    ax = quadro(fig, [0.05, 0.09, 0.42, 0.72])
    H = ax.get_ylim()[1]
    cx, cy, R = 50, H / 2 - 2, 38
    ax.add_patch(Circle((cx, cy), R, facecolor="#f6f5f1", edgecolor=TINTA2, lw=1.6))
    ax.add_patch(Polygon([[cx - 5, cy + R - 0.6], [cx, cy + R + 6], [cx + 5, cy + R - 0.6]],
                         facecolor="#f6f5f1", edgecolor=TINTA2, lw=1.6))
    for s in (-1, 1):
        ax.add_patch(Ellipse((cx + s * (R + 1.6), cy), 4, 11, facecolor="#f6f5f1", edgecolor=TINTA2, lw=1.6))
    ax.plot([cx - R, cx + R], [cy, cy], color=GRADE, lw=1.2)
    ax.plot([cx, cx], [cy - R, cy + R], color=GRADE, lw=1.2)
    pos = {"Fp1": (-.25, .76), "Fp2": (.25, .76), "F3": (-.36, .40), "Fz": (0, .40), "F4": (.36, .40),
           "T7": (-.80, 0), "C3": (-.40, 0), "Cz": (0, 0), "C4": (.40, 0), "T8": (.80, 0),
           "P3": (-.36, -.40), "Pz": (0, -.40), "P4": (.36, -.40), "O1": (-.25, -.76), "O2": (.25, -.76)}
    for nome, (px, py) in pos.items():
        x, y = cx + px * R, cy + py * R
        if nome in COR_CANAL:
            ax.add_patch(Circle((x, y), 4.6, facecolor=COR_CANAL[nome], edgecolor="white", lw=2, zorder=4))
            ax.text(x, y, nome, ha="center", va="center", size=13, weight="bold", color="white", zorder=5)
        else:
            ax.add_patch(Circle((x, y), 1.3, facecolor="#c4c3bc", edgecolor="none"))
            ax.text(x, y - 3.6, nome, ha="center", va="center", size=9, color=APAGADO)
    ax.text(cx, cy + R + 8.5, "nariz", ha="center", size=11, color=TINTA2)
    ax.text(cx - R * 0.72, cy - R - 5, "lado esquerdo", ha="center", size=11, color=TINTA2)
    ax.text(cx + R * 0.72, cy - R - 5, "lado direito", ha="center", size=11, color=TINTA2)
    fig.text(0.05, 0.075, "Cabeça vista de cima. Em cinza, outras posições do sistema 10-20 que o projeto não usa.",
             size=10, color=APAGADO)

    y = paragrafo(fig, 0.52, 0.80,
                  "O sistema 10-20 dá nome a cada ponto do couro cabeludo. O projeto usa só três, "
                  "todos em cima da faixa do córtex motor.", larg=64, size=14)
    ax = fig.add_axes([0.52, y - 0.275, 0.43, 0.255])
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    linhas = [("C3", "hemisfério esquerdo", "movimento da mão direita"),
              ("C4", "hemisfério direito", "movimento da mão esquerda"),
              ("Cz", "linha do meio", "atividade motora central")]
    for cab, xx in (("canal", 0.02), ("fica sobre o", 0.19), ("e acompanha", 0.55)):
        ax.text(xx, 0.93, cab, size=12, color=APAGADO, va="center")
    ax.plot([0, 1], [0.83, 0.83], color=GRADE, lw=1.2)
    for i, (c, a, b) in enumerate(linhas):
        yy = 0.67 - i * 0.26
        ax.add_patch(Circle((0.055, yy), 0.045, facecolor=COR_CANAL[c], edgecolor="none",
                            transform=ax.transData))
        ax.text(0.055, yy, c, ha="center", va="center", size=11, weight="bold", color="white")
        ax.text(0.19, yy, a, size=13.5, color=TINTA, va="center")
        ax.text(0.55, yy, b, size=13.5, color=TINTA, va="center")
        ax.plot([0, 1], [yy - 0.13, yy - 0.13], color=GRADE, lw=1)
    ax.set_aspect(0.255 * ALT / (0.43 * LARG))
    topicos(fig, 0.52, y - 0.31, [
        "A placa também pede um eletrodo de referência e um de terra, que não entram na conta "
        "dos três canais.",
        "A ordem C3, C4, Cz é a mesma que a rede viu no treino. Trocar dois fios não gera erro "
        "nenhum na tela. O resultado só piora, em silêncio.",
        "Três eletrodos é pouco. Capacetes de pesquisa usam 32 ou 64, e a base pública usada no "
        "treino foi gravada com 64.",
    ], larg=56, size=13.5)


def p_placa(D):
    fig = nova("3 · A placa", "Do eletrodo ao computador",
               "A OpenBCI Cyton transforma microvolts em números e manda esses números por rádio. "
               "O computador recebe 250 leituras por segundo de cada canal.", ativa=2)
    ax = quadro(fig, [0.04, 0.44, 0.92, 0.27])
    H = ax.get_ylim()[1]
    itens = [("Eletrodo", "microvolts\nna pele"),
             ("Amplificador", "chip ADS1299,\nganho de 24 vezes"),
             ("Conversor A/D", "24 bits,\n250 leituras por segundo"),
             ("Rádio", "a placa fica na cabeça,\nsem fio até o PC"),
             ("Dongle USB", "aparece no Windows\ncomo a porta COM9"),
             ("BrainFlow", "biblioteca que entrega\nos números ao Python")]
    w, gap = 13.6, 3.2
    for i, (a, b) in enumerate(itens):
        x = 0.3 + i * (w + gap)
        caixa(ax, x, 3, w, H - 6, a, b, cor="#e3edfa" if i < 4 else PAINEL, size=12.5)
        if i:
            seta(ax, (x - gap + 0.3, H / 2), (x - 0.3, H / 2))
    ax.text(0.3 + 2 * (w + gap), 0.3, "dentro da placa Cyton", ha="center", size=11, color=TINTA2)

    c = D["cyton"]
    a = int(100.0 * c["fs"])
    n = int(0.24 * c["fs"])
    seg = D["limpeza"]["banda"][0, a:a + n]
    t = np.arange(n) / c["fs"] * 1000
    ax = fig.add_axes([0.07, 0.115, 0.50, 0.27])
    tf = np.linspace(t[0], t[-1], 800)
    from scipy.interpolate import CubicSpline
    ax.plot(tf, CubicSpline(t, seg)(tf), color="#b9d2f1", lw=2.2)
    ax.plot(t, seg, "o", color=AZUL, ms=6.5, mec="white", mew=1.2)
    ax.set_xlabel("tempo (milissegundos)")
    ax.set_ylabel("C3 (µV)")
    limpar(ax)
    ax.annotate("4 ms entre uma leitura e a seguinte", ((t[8] + t[9]) / 2, max(seg[8], seg[9]) + 1.2),
                (t[8] + 26, seg.max() + 6), size=11.5, color=TINTA, va="center",
                arrowprops=dict(arrowstyle="-", color=APAGADO, lw=1))
    ax.set_ylim(seg.min() - 3, seg.max() + 9)
    topicos(fig, 0.62, 0.385, [
        ("Amostrar é tirar fotos",
         "A curva clara é o sinal como ele existe na pele. Cada ponto azul é uma leitura da "
         "placa. Ao computador só chegam os pontos.", ),
        ("Resolução de sobra",
         "Com 24 bits, cada degrau do conversor vale perto de 0,02 µV. Quem limita o sistema "
         "é o ruído que chega junto com o sinal."),
    ], larg=44, size=13)


def _trecho(fig, D, chave, a, dur, passo, escala, pos, fonte="x"):
    d = D[chave]
    fs = d["fs"]
    i0, i1 = int(a * fs), int((a + dur) * fs)
    x = (d["x"] if fonte == "x" else D["limpeza"][fonte])[:, i0:i1]
    t = a + np.arange(x.shape[1]) / fs
    ax = fig.add_axes(pos)
    empilhar(ax, t, x, passo)
    faixas(ax, d["ev"])
    ax.set_xlabel("tempo dentro da gravação (s)")
    regua(ax, t[-1] + dur * 0.012, -passo * 0.25, escala)
    return ax


def p_physionet(D):
    fig = nova("3 · A placa", "Um trecho da base pública (PhysioNet)",
               "Voluntário S001, run 4: a pessoa só imagina abrir e fechar a mão quando a tela "
               "manda. Nenhum filtro foi aplicado ainda.", ativa=2)
    _trecho(fig, D, "edf", 20, 34, 330, 200, [0.09, 0.30, 0.82, 0.40])
    topicos(fig, 0.06, 0.225, [
        "A base EEG Motor Movement/Imagery tem 109 voluntários, 64 eletrodos e 160 leituras por "
        "segundo. Daqui o projeto aproveita só C3, C4 e Cz.",
    ], larg=62, size=13)
    topicos(fig, 0.53, 0.225, [
        "As faixas coloridas marcam quando a dica de mão estava na tela. A olho nu não se vê "
        "diferença entre os trechos, e esse é o tamanho do problema.",
    ], larg=62, size=13)


def p_cyton(D):
    fig = nova("3 · A placa", "Um trecho do nosso capacete (OpenBCI Cyton)",
               "Sessão S004_R1, gravada no projeto com o mesmo protocolo de dicas. Em cima o sinal "
               "como a placa entrega, embaixo o mesmo trecho depois dos filtros.", ativa=2)
    ax = _trecho(fig, D, "cyton", 99.6, 30, 260, 200, [0.09, 0.485, 0.82, 0.20])
    ax.set_xlabel("")
    ax.set_xticklabels([])
    fig.text(0.09, 0.692, "bruto", size=12, weight="bold", color=TINTA, va="bottom")
    ax = _trecho(fig, D, "cyton", 99.6, 30, 150, 50, [0.09, 0.225, 0.82, 0.20], fonte="banda")
    for txt in ax.texts[:]:
        if txt.get_text().startswith("mão"):
            txt.remove()
    fig.text(0.09, 0.432, "filtrado (0,5 a 35 Hz)", size=12, weight="bold", color=TINTA, va="bottom")
    fig.text(0.06, 0.075, textwrap.fill(
        "O sinal bruto passeia centenas de microvolts para cima e para baixo: é o contato do "
        "eletrodo com a pele variando. A atividade do cérebro é a parte miúda que aparece depois "
        "do filtro. Repare que a régua de escala mudou de 200 para 50 µV.", 150),
        size=13, color=TINTA, va="bottom", linespacing=1.45)


def p_limpeza(D):
    L = D["limpeza"]
    fs = D["cyton"]["fs"]
    fig = nova("4 · A limpeza", "Quatro passos antes de a rede ver o sinal",
               "O mesmo tratamento vale para a base pública e para o capacete. Se os dois fossem "
               "tratados de forma diferente, a rede aprenderia a distinguir a origem do dado.",
               ativa=3)
    a, dur = 100.0, 6.0
    i0, i1 = int(a * fs), int((a + dur) * fs)
    j0, j1 = int(a * cfg.TAXA_IA), int((a + dur) * cfg.TAXA_IA)
    from scipy.signal import detrend
    final = detrend(L["final"][0, j0:j1])
    final = final / final.std()
    passos = [("1  Sinal bruto da placa", L["bruto"][0, i0:i1], fs, "µV"),
              ("2  Corta os 60 Hz da rede elétrica", L["notch"][0, i0:i1], fs, "µV"),
              ("3  Deixa passar só de 0,5 a 35 Hz", L["banda"][0, i0:i1], fs, "µV"),
              ("4  Reduz para 128 Hz, tira a inclinação e põe em escala padrão", final,
               cfg.TAXA_IA, "desvios")]
    for k, (nome, x, f, un) in enumerate(passos):
        ax = fig.add_axes([0.06, 0.585 - k * 0.158, 0.50, 0.105])
        t = np.arange(len(x)) / f
        ax.plot(t, x - (np.median(x) if k < 3 else 0), color=AZUL, lw=1.0)
        ax.set_xlim(0, dur)
        ax.tick_params(length=0)
        ax.set_title(nome, loc="left", size=12.5, weight="bold", color=TINTA, pad=4)
        ax.set_ylabel(un, size=10.5)
        ax.yaxis.set_major_locator(plt.MaxNLocator(3))
        ax.grid(axis="y", color=GRADE, lw=0.8)
        if k < 3:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel("tempo (s), canal C3")

    ax = fig.add_axes([0.64, 0.30, 0.31, 0.385])
    ax.axvspan(0.5, 35, color="#e9f1fb", lw=0)
    ax.semilogy(L["f0"], L["p0"][0], color="#b9b8b2", lw=1.6)
    ax.semilogy(L["f1"], L["p1"][0], color=AZUL, lw=2)
    ax.set_xlim(0, 80)
    ax.set_ylim(1e-3, None)
    ax.set_xlabel("frequência (Hz)")
    ax.set_ylabel("potência (µV²/Hz)")
    limpar(ax)
    k60 = np.argmin(abs(L["f0"] - 60))
    ax.annotate("rede elétrica\n(60 Hz)", (60, L["p0"][0][k60]), (67, L["p0"][0][k60] * 6),
                size=11, color=TINTA, ha="center", arrowprops=dict(arrowstyle="-", color=APAGADO))
    ax.text(17.5, 2.2e-3, "faixa que fica", ha="center", size=11, color=TINTA2)
    ax.text(72, L["p0"][0][np.argmin(abs(L["f0"] - 72))] * 3, "antes", size=11.5, color=TINTA2, ha="center")
    ax.text(41.5, 3e-2, "depois", size=11.5, color=TINTA, weight="bold")
    ax.set_title("O que cada frequência carrega", loc="left", size=13, weight="bold", color=TINTA, pad=8)
    fig.text(0.64, 0.215, textwrap.fill(
        "O corte de baixo foi escolhido por medição: com 0,5 Hz a rede acerta 72,0%, com 1 Hz "
        "cai para 69,1% e com 4 Hz para 61,0%. Boa parte da informação está nas ondas lentas.", 56),
        size=12.5, color=TINTA, va="top", linespacing=1.45)


def p_janela(D):
    L = D["limpeza"]
    fs = D["cyton"]["fs"]
    fig = nova("4 · A limpeza", "A janela de 4,1 s, centrada na troca de estado",
               "A rede recebe o sinal em pedaços de 4,1 segundos. Cada pedaço tem no meio o instante em "
               "que a pessoa passou do descanso para a tarefa, ou da tarefa para o descanso.",
               ativa=3)
    ev = D["cyton"]["ev"]
    k = next(i for i, e in enumerate(ev) if e[0] > 98 and e[2] == "dir")
    a, b = ev[k - 1][0] - 1.2, ev[k + 1][1] + 0.2
    i0, i1 = int(a * fs), int(b * fs)
    t = a + np.arange(i1 - i0) / fs
    ax = fig.add_axes([0.09, 0.335, 0.82, 0.335])
    empilhar(ax, t, L["banda"][:, i0:i1], 80)
    faixas(ax, ev)
    ax.set_xlabel("tempo dentro da gravação (s)")
    topo = ax.get_ylim()[1]
    for centro, nome, cor, yb in ((ev[k][0], "janela de MOVIMENTO", TINTA, -0.20),
                                  (ev[k + 1][0], "janela de DESCANSO", TINTA2, -0.20)):
        x0, x1 = centro - cfg.MEIA_JANELA, centro + cfg.MEIA_JANELA
        ax.axvline(centro, color=TINTA, lw=1.2, ls=(0, (3, 3)))
        ax.plot([x0 + 0.04, x0 + 0.04, x1 - 0.04, x1 - 0.04], [-232, -246, -246, -232], color=cor, lw=2)
        ax.text(x0 + 0.12 if "MOV" in nome else centro + 0.12, -254, nome, ha="left", va="top",
                size=12.5, weight="bold", color=cor)
    ax.set_ylim(-300, topo)
    ax.text(ev[k - 1][0] + 2.05, 1.01, "descanso", transform=ax.get_xaxis_transform(), ha="center",
            va="bottom", size=11, color=TINTA2)
    ax.text(ev[k + 1][0] + 2.05, 1.01, "descanso", transform=ax.get_xaxis_transform(), ha="center",
            va="bottom", size=11, color=TINTA2)
    topicos(fig, 0.06, 0.225, [
        f"Cada janela tem {cfg.AMOSTRAS} leituras por canal (4,1 s a 128 Hz). Com três canais, a "
        f"rede recebe uma tabela de 3 por {cfg.AMOSTRAS} números.",
    ], larg=62, size=13)
    topicos(fig, 0.53, 0.225, [
        "As janelas encostam uma na outra sem se sobrepor. Nenhum pedaço do sinal recebe dois "
        "rótulos, e o que a rede aprende é a cara da transição.",
    ], larg=62, size=13)


def _celulas(ax, x0, y0, valores, lado=7.0, cor=PAINEL, borda=None, negrito=False, cor_txt=TINTA):
    for i, v in enumerate(valores):
        ax.add_patch(Rectangle((x0 + i * lado, y0), lado - 0.6, lado - 0.6, facecolor=cor,
                               edgecolor=borda or "none", lw=1.6))
        ax.text(x0 + i * lado + (lado - 0.6) / 2, y0 + (lado - 0.6) / 2, str(v), ha="center",
                va="center", size=16, color=cor_txt, weight="bold" if negrito else "normal")


def p_conv_numeros(D):
    fig = nova("5 · A rede", "Convolução, primeiro com números pequenos",
               "Uma convolução desliza uma régua curta de pesos (o kernel) por cima do sinal. Em cada "
               "posição ela multiplica os números que estão alinhados e soma tudo.", ativa=4)
    ax = quadro(fig, [0.05, 0.10, 0.58, 0.60])
    H = ax.get_ylim()[1]
    sinal = [1, 3, 2, 5, 4, 1, 0]
    kernel = [1, 0, -1]
    saida = [sum(sinal[i + j] * kernel[j] for j in range(3)) for i in range(5)]
    L, x0 = 9.0, 26
    pos = 3
    ax.text(x0 - 3, H - 9.5, "sinal", ha="right", va="center", size=14, weight="bold", color=TINTA)
    _celulas(ax, x0, H - 14, sinal, L, cor="#e3edfa")
    ax.add_patch(Rectangle((x0 + pos * L - 0.9, H - 14.9), 3 * L + 1.2, L + 1.2, facecolor="none",
                           edgecolor=LARANJA, lw=2.4))
    ax.text(x0 - 3, H - 25.5, "kernel", ha="right", va="center", size=14, weight="bold", color=TINTA)
    _celulas(ax, x0 + pos * L, H - 30, kernel, L, cor="#fde9df")
    for j in range(3):
        xx = x0 + (pos + j) * L + (L - 0.6) / 2
        ax.text(xx, H - 17.6, "×", ha="center", va="center", size=14, color=TINTA2)
    seta(ax, (x0 + pos * L - 12, H - 25.8), (x0 + pos * L - 2, H - 25.8), cor=LARANJA, lw=2)
    ax.text(x0 + pos * L - 13, H - 25.8, "desliza", ha="right", va="center", size=11.5, color=TINTA2)
    conta = "  +  ".join(f"({sinal[pos + j]} × {kernel[j]})" for j in range(3))
    ax.text(x0 + (pos + 1.5) * L - 0.3, H - 37, f"{conta}  =  {saida[pos]}", ha="center", va="center",
            size=15, color=TINTA)
    ax.text(x0 - 3, H - 47.5, "saída", ha="right", va="center", size=14, weight="bold", color=TINTA)
    _celulas(ax, x0 + L, H - 52, saida, L, cor=PAINEL)
    ax.add_patch(Rectangle((x0 + (pos + 1) * L - 0.9, H - 52.9), L + 1.2, L + 1.2, facecolor="none",
                           edgecolor=LARANJA, lw=2.4))
    seta(ax, (x0 + (pos + 1.5) * L - 0.3, H - 39.5), (x0 + (pos + 1.5) * L - 0.3, H - 42.6), cor=LARANJA, lw=2)
    topicos(fig, 0.66, 0.69, [
        ("O kernel é um detector de formato",
         "Este kernel (1, 0, -1) compara um número com o que vem dois passos depois. A saída "
         "fica alta onde o sinal está caindo e negativa onde está subindo."),
        ("O mesmo kernel percorre tudo",
         "Os três pesos não mudam de uma posição para outra. Por isso a rede acha o mesmo "
         "formato em qualquer ponto da janela."),
        ("Ninguém escolhe os pesos",
         "No treino a rede ajusta os pesos sozinha até os kernels detectarem os formatos que "
         "ajudam a separar descanso de movimento."),
    ], larg=38, size=13)


def p_conv_real(D):
    R = D["rede"]
    fig = nova("5 · A rede", "A mesma conta, no sinal de verdade",
               "Um kernel que a rede aprendeu, com 33 pesos (um quarto de segundo), passando sobre o "
               "canal C3 de uma janela real. Onde o sinal se parece com o kernel, a saída cresce.",
               ativa=4)
    K = R["kernels"][3]
    esp = np.abs(np.fft.rfft(K, 512, axis=1))
    fr = np.fft.rfftfreq(512, 1 / cfg.TAXA_IA)
    banda = (fr > 6) & (fr < 20)
    q = np.argmax(esp[:, banda].max(axis=1) / (esp.mean(axis=1) + 1e-9))
    w = K[q]
    x = R["x"][0]
    saida = np.correlate(x, w, mode="same")
    a, b = 150, 400
    t = np.arange(a, b) / cfg.TAXA_IA
    centros = [a + 55, a + 125, a + 195]
    ax = fig.add_axes([0.07, 0.445, 0.60, 0.215])
    ax.plot(t, x[a:b], color=AZUL, lw=1.5)
    esc = 0.8 * np.abs(x[a:b]).max() / np.abs(w).max()
    for c in centros:
        tk = (c + np.arange(-16, 17)) / cfg.TAXA_IA
        ax.axvspan(tk[0], tk[-1], color="#fde9df", lw=0, zorder=0)
        ax.plot(tk, w * esc, color=LARANJA, lw=2.2)
    ax.set_xlim(t[0], t[-1])
    ax.set_xticklabels([])
    ax.tick_params(length=0)
    ax.set_ylabel("entrada")
    ax.grid(axis="y", color=GRADE, lw=0.8)
    ax.set_title("Sinal de C3 (azul) e o kernel parado em três posições (laranja)", loc="left",
                 size=12.5, weight="bold", color=TINTA, pad=6)
    ax2 = fig.add_axes([0.07, 0.135, 0.60, 0.215])
    ax2.plot(t, saida[a:b], color=TINTA, lw=1.6)
    for c in centros:
        ax2.plot(c / cfg.TAXA_IA, saida[c], "o", ms=10, color=LARANJA, mec="white", mew=1.6, zorder=5)
        ax2.axvline(c / cfg.TAXA_IA, color=LARANJA, lw=1, ls=(0, (3, 3)), zorder=1)
    ax2.set_xlim(t[0], t[-1])
    ax2.set_xlabel("tempo dentro da janela (s)")
    ax2.set_ylabel("saída")
    ax2.tick_params(length=0)
    ax2.grid(axis="y", color=GRADE, lw=0.8)
    ax2.set_title("Resultado da convolução, com as três posições marcadas", loc="left",
                  size=12.5, weight="bold", color=TINTA, pad=6)
    pico = fr[np.argmax(esp[q])]
    topicos(fig, 0.71, 0.68, [
        ("Cada ponto laranja é uma soma",
         "A saída em cada instante é a soma dos 33 produtos entre o kernel e o pedaço de sinal "
         "que está embaixo dele naquele momento."),
        ("O kernel virou um filtro",
         f"Este aqui oscila perto de {pico:.0f} Hz. A saída fica grande quando o sinal tem essa "
         "frequência e quase some quando não tem."),
        ("Entra uma curva, sai outra",
         "O comprimento é o mesmo: 526 pontos entram e 526 saem. A rede faz isso com 100 "
         "kernels diferentes ao mesmo tempo."),
    ], larg=33, size=13)


def p_rede_mapa(D):
    R = D["rede"]
    fig = nova("5 · A rede", "A rede inteira em uma página",
               f"São {R['n_param']:,}".replace(",", ".") + " pesos, o que é pouco para uma rede neural. "
               "Ela foi mantida pequena de propósito, porque três eletrodos não têm informação para "
               "sustentar uma rede grande.", ativa=4)
    ax = quadro(fig, [0.04, 0.27, 0.92, 0.43])
    H = ax.get_ylim()[1]
    blocos = [
        ("Entrada", "uma janela\nde 4,1 s", "3 canais\n× 526 pontos", "#e3edfa"),
        ("Filtros\nno tempo", "100 kernels em\n5 comprimentos", "100 mapas\n× 3 × 526", PAINEL),
        ("Filtro\nespacial", "combina C3,\nC4 e Cz", "300 mapas\n× 526", PAINEL),
        ("Potência", "quadrado,\nmédia e log", "300 mapas\n× 131", PAINEL),
        ("Refino", "olha o tempo e\nmistura os mapas", "32 mapas\n× 131", PAINEL),
        ("Resumo", "média em\n32 trechos", "32 × 32 =\n1.024 números", PAINEL),
        ("Classificador", "uma soma\nponderada", "três\nprobabilidades", "#fde9df"),
    ]
    w, gap = 11.6, 2.75
    y, h = 0.5, H - 1
    for i, (tit, faz, forma, cor) in enumerate(blocos):
        x = 0.3 + i * (w + gap)
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.2",
                                    facecolor=cor, edgecolor="none"))
        ax.text(x + w / 2, y + h - 1.8, tit, ha="center", va="top", size=13.5, weight="bold",
                color=TINTA, linespacing=1.25)
        ax.text(x + w / 2, y + h - 9.2, faz, ha="center", va="top", size=11.5, color=TINTA2,
                linespacing=1.4)
        ax.plot([x + 1.5, x + w - 1.5], [y + 8.2, y + 8.2], color="#cfcec7", lw=1)
        ax.text(x + w / 2, y + 4.2, forma, ha="center", va="center", size=11.5, color=TINTA,
                linespacing=1.4)
        if i:
            seta(ax, (x - gap + 0.25, y + h / 2), (x - 0.25, y + h / 2))
    fig.text(0.05, 0.235, "Em cima, o que cada bloco faz. Embaixo, o tamanho do que sai dele.",
             size=11.5, color=TINTA2)
    fig.text(0.05, 0.195, textwrap.fill(
        "A sequência copia um método clássico de BCI feito à mão: filtrar em várias faixas de "
        "frequência, combinar os eletrodos, medir a potência e classificar pelo logaritmo dela. "
        "A diferença é que aqui a rede aprende sozinha os filtros e as combinações.", 140),
        size=13.5, color=TINTA, va="top", linespacing=1.45)


def p_banco(D):
    R = D["rede"]
    fig = nova("5 · A rede", "Camada 1: cem filtros no tempo, em cinco comprimentos",
               "Um kernel curto só enxerga oscilações rápidas. Um kernel longo consegue acompanhar as "
               "lentas. A rede usa cinco comprimentos em paralelo, com vinte kernels em cada um.",
               ativa=4)
    fr = np.fft.rfftfreq(512, 1 / cfg.TAXA_IA)
    for i, (k, K) in enumerate(zip(cfg.CNN_KERNELS, R["kernels"])):
        x0 = 0.065 + i * 0.181
        esp = np.abs(np.fft.rfft(K, 512, axis=1))
        ordem = np.argsort(-esp.max(axis=1))[:3]
        ax = fig.add_axes([x0, 0.455, 0.152, 0.185])
        tk = (np.arange(k) - k // 2) / cfg.TAXA_IA * 1000
        for j, q in enumerate(ordem):
            w = K[q] / np.abs(K[q]).max()
            ax.plot(tk, w - j * 2.3, color=AZUL, lw=1.8, marker="o" if k <= 15 else None, ms=3.5)
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
        ax.tick_params(length=0)
        ax.set_xlabel("ms", labelpad=1)
        ax.set_title(f"{k} pesos = {k / cfg.TAXA_IA * 1000:.0f} ms", loc="left", size=13,
                     weight="bold", color=TINTA, pad=6)
        ax = fig.add_axes([x0, 0.205, 0.152, 0.135])
        for q in range(K.shape[0]):
            ax.plot(fr, esp[q] / esp.max(), color="#9ec0ea", lw=0.9)
        ax.plot(fr, esp.mean(axis=0) / esp.max(), color="#123f7a", lw=2.2)
        ax.set_xlim(0, 40)
        ax.set_ylim(0, 1.05)
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
        ax.tick_params(length=0)
        ax.set_xlabel("Hz", labelpad=1)
    fig.text(0.065, 0.705, "Três dos vinte kernels de cada comprimento, como a rede os aprendeu",
             size=12.5, color=TINTA2)
    fig.text(0.065, 0.36, "Que frequências cada grupo deixa passar (linha escura: média dos vinte)",
             size=12.5, color=TINTA2)
    fig.text(0.065, 0.075, textwrap.fill(
        "Os kernels longos concentram a resposta nas frequências baixas, onde está a onda lenta "
        "da transição. Os curtos respondem a uma faixa larga. Somando os cinco grupos, a janela "
        "vira 100 versões filtradas de cada canal.", 150),
        size=13, color=TINTA, va="bottom", linespacing=1.45)


def p_espacial(D):
    R = D["rede"]
    W = R["espacial"] / np.linalg.norm(R["espacial"], axis=1, keepdims=True)
    fig = nova("5 · A rede", "Camada 2: juntar os três eletrodos",
               "Para cada versão filtrada, a rede faz três somas ponderadas de C3, C4 e Cz. Cada soma "
               "é um jeito diferente de comparar os lados da cabeça.", ativa=4)
    ordem = np.argsort(W[:, 0] - W[:, 1])
    ax = fig.add_axes([0.07, 0.475, 0.56, 0.19])
    im = ax.imshow(W[ordem].T, aspect="auto", cmap=DIV, vmin=-1, vmax=1, interpolation="nearest")
    ax.set_yticks(range(3))
    ax.set_yticklabels(cfg.CANAIS, size=13, weight="bold", color=TINTA)
    ax.set_xlabel("os 300 filtros espaciais, ordenados do que privilegia C4 ao que privilegia C3")
    ax.set_xticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    cax = fig.add_axes([0.43, 0.385, 0.20, 0.016])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    cb.set_ticks([-1, 0, 1])
    cb.set_ticklabels(["peso negativo", "zero", "peso positivo"])
    cb.outline.set_visible(False)
    cb.ax.tick_params(length=0, labelsize=10.5)

    # três exemplos lidos dos pesos
    d = W[:, 0] - W[:, 1]
    soma = W.sum(axis=1)
    escolhas = [("C3 contra C4", np.argmax(np.abs(d) - np.abs(W[:, 2]))),
                ("os três juntos", np.argmax(np.abs(soma))),
                ("Cz contra os lados", np.argmax(np.abs(W[:, 2] - (W[:, 0] + W[:, 1]) / 2)
                                                 - np.abs(d)))]
    for i, (nome, q) in enumerate(escolhas):
        ax = fig.add_axes([0.07 + i * 0.195, 0.125, 0.15, 0.135])
        v = W[q] * np.sign(W[q][np.argmax(np.abs(W[q]))])
        ax.bar(range(3), v, width=0.55, color=[COR_CANAL[c] for c in cfg.CANAIS])
        ax.axhline(0, color=APAGADO, lw=1)
        ax.set_xticks(range(3))
        ax.set_xticklabels(cfg.CANAIS, size=11.5)
        ax.set_yticks([])
        ax.set_ylim(-1.05, 1.05)
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.tick_params(length=0)
        ax.set_title(nome, loc="left", size=12, weight="bold", color=TINTA, pad=4)
    topicos(fig, 0.68, 0.66, [
        ("Subtrair é comparar",
         "Um filtro com peso positivo em C3 e negativo em C4 mede a diferença entre os dois "
         "hemisférios. É assim que a rede descobre de que lado o sinal mudou."),
        ("Por que só três por mapa",
         "Com três eletrodos só existem três combinações independentes. Pôr mais filtros "
         "espaciais não acrescentaria informação nenhuma."),
        ("O que sai daqui",
         "Os canais deixam de existir como canais. Ficam 300 curvas, cada uma com a sua faixa "
         "de frequência e a sua combinação de eletrodos."),
    ], larg=38, size=12.5)
    fig.text(0.07, 0.315, "Três filtros tirados da figura de cima:", size=12, color=TINTA2)


def p_potencia(D):
    E = D["rede"]["etapas"]
    q = int(np.argmax(E[3][:, 0, :].std(axis=1) * (E[6][:, 0, :].std(axis=1))))
    a = E[3][q, 0]
    quad = a ** 2
    media = E[5][q, 0]
    logv = E[6][q, 0]
    fig = nova("5 · A rede", "Camada 3: da oscilação para a potência",
               "Para a rede importa quanta energia a onda carrega em cada trecho, e pouco importa se "
               "ela está subindo ou descendo naquele instante. Três operações simples fazem essa troca.",
               ativa=4)
    t1 = np.arange(526) / cfg.TAXA_IA
    t2 = (np.arange(131) * 4 + 2) / cfg.TAXA_IA
    paineis = [("Uma das 300 curvas que saem do filtro espacial", t1, a, AZUL, "line"),
               ("1  Ao quadrado: tudo fica positivo e os picos crescem", t1, quad, AZUL, "fill"),
               ("2  Média de 4 em 4 pontos: de 526 para 131 valores", t2, media, AZUL, "step"),
               ("3  Logaritmo: achata os picos e dá destaque às variações pequenas", t2, logv, TINTA, "step")]
    for k, (nome, t, v, cor, tipo) in enumerate(paineis):
        ax = fig.add_axes([0.07, 0.575 - k * 0.153, 0.55, 0.098])
        if tipo == "line":
            ax.plot(t, v, color=cor, lw=1.0)
        elif tipo == "fill":
            ax.fill_between(t, 0, v, color=cor, lw=0, alpha=0.85)
        else:
            ax.step(t, v, where="mid", color=cor, lw=1.5)
        ax.set_xlim(0, t1[-1])
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
        ax.tick_params(length=0)
        ax.set_title(nome, loc="left", size=12.5, weight="bold", color=TINTA, pad=4)
        if k < 3:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel("tempo dentro da janela (s). A linha tracejada marca o início da tarefa")
        ax.axvline(cfg.MEIA_JANELA, color=LARANJA, lw=1.2, ls=(0, (3, 3)))
    topicos(fig, 0.67, 0.685, [
        ("Quadrado",
         "Uma onda de 10 Hz passa metade do tempo negativa. Ao quadrado sobra só o tamanho "
         "dela, que é a potência."),
        ("Média",
         "Juntar de quatro em quatro pontos alisa a curva e diminui a conta das camadas "
         "seguintes."),
        ("Logaritmo",
         "A potência do EEG varia muito de um trecho para outro. No log, cair pela metade vale "
         "o mesmo em qualquer nível."),
        ("O quadrado é a ativação",
         "Antes deste ponto a rede só soma e multiplica. O quadrado é a primeira operação que "
         "tira a rede da linha reta."),
    ], larg=40, size=12.5)


def p_saida(D):
    R = D["rede"]
    E = R["etapas"]
    fig = nova("5 · A rede", "As últimas camadas e a resposta",
               "A rede refina as 300 curvas, mistura tudo em 32 mapas e resume cada mapa em 32 "
               "trechos. Sobram 1.024 números, e uma última soma ponderada vira três probabilidades.",
               ativa=4)
    ax = fig.add_axes([0.07, 0.15, 0.40, 0.50])
    mapa = E["pool"][:, 0, :]
    ax.imshow(mapa, aspect="auto", cmap=SEQ, interpolation="nearest",
              extent=[0, cfg.JANELA_S, 32, 0])
    ax.axvline(cfg.MEIA_JANELA, color=LARANJA, lw=1.4, ls=(0, (3, 3)))
    ax.set_xlabel("tempo dentro da janela (s). A linha tracejada marca o início da tarefa")
    ax.set_ylabel("os 32 mapas")
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    ax.set_title("O que a rede enxerga nesta janela (32 × 32)", loc="left", size=13, weight="bold",
                 color=TINTA, pad=8)

    ax = fig.add_axes([0.60, 0.40, 0.33, 0.25])
    nomes = ["descanso", "mão esquerda", "mão direita"]
    p = R["prob"] * 100
    cores = ["#c4c3bc", "#c4c3bc", AZUL]
    ax.barh(range(3), p, height=0.55, color=cores)
    for i in range(3):
        ax.text(p[i] + 1.5, i, f"{p[i]:.0f}%", va="center", size=14, weight="bold", color=TINTA)
    ax.set_yticks(range(3))
    ax.set_yticklabels(nomes, size=13, color=TINTA)
    ax.invert_yaxis()
    ax.set_xlim(0, 105)
    ax.set_xticks([])
    ax.spines["bottom"].set_visible(False)
    ax.tick_params(length=0)
    ax.axvline(cfg.LIMIAR * 100, color=TINTA, lw=1.2, ls=(0, (3, 3)))
    ax.text(cfg.LIMIAR * 100, -0.62, f"limiar de {cfg.LIMIAR * 100:.0f}%", ha="center", size=11, color=TINTA2)
    ax.set_title("A resposta para esta janela", loc="left", size=13, weight="bold", color=TINTA, pad=24)
    topicos(fig, 0.57, 0.335, [
        "A janela do exemplo é de mão direita (voluntário S001, run 4). É a mesma que percorreu "
        "as páginas anteriores.",
        "As três probabilidades somam 100%. A rede sempre responde alguma coisa, mesmo quando o "
        "sinal é só ruído. Quem decide se a resposta merece virar movimento é a etapa seguinte.",
    ], larg=48, size=13)
    seta(fig.add_axes([0, 0, 1, 1], facecolor="none", frameon=False), (0.485, 0.50), (0.555, 0.50))
    fig.axes[-1].set_xlim(0, 1)
    fig.axes[-1].set_ylim(0, 1)
    fig.axes[-1].axis("off")


def p_acerto(D):
    conf = D["conf"]
    fig = nova("5 · A rede", "Quanto ela acerta",
               "O modelo distribuído foi treinado com 105 voluntários do PhysioNet e duas gravações "
               "do projeto, por 35 épocas. No melhor ponto da validação ele acertou 72,2% das janelas.",
               ativa=4)
    ax = fig.add_axes([0.10, 0.14, 0.34, 0.50])
    pct = conf / conf.sum(axis=1, keepdims=True) * 100
    ax.imshow(pct, cmap=SEQ, vmin=0, vmax=100)
    nomes = ["descanso", "mão\nesquerda", "mão\ndireita"]
    for i in range(3):
        for j in range(3):
            ax.text(j, i - 0.08, f"{pct[i, j]:.0f}%", ha="center", va="center", size=19,
                    weight="bold", color="white" if pct[i, j] > 55 else TINTA)
            ax.text(j, i + 0.26, f"{conf[i, j]} janelas", ha="center", va="center", size=10.5,
                    color="white" if pct[i, j] > 55 else TINTA2)
    ax.set_xticks(range(3))
    ax.set_xticklabels(nomes, size=12, color=TINTA)
    ax.set_yticks(range(3))
    ax.set_yticklabels(nomes, size=12, color=TINTA)
    ax.xaxis.tick_top()
    ax.set_xlabel("a rede respondeu", labelpad=10)
    ax.xaxis.set_label_position("top")
    ax.set_ylabel("o certo era", labelpad=8)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    ax.set_xticks(np.arange(-.5, 3), minor=True)
    ax.set_yticks(np.arange(-.5, 3), minor=True)
    ax.grid(which="minor", color=FUNDO, lw=3)
    ax.tick_params(which="minor", length=0)
    ac = np.trace(conf) / conf.sum() * 100
    base = conf.sum(axis=1).max() / conf.sum() * 100
    fig.text(0.10, 0.085, f"Voluntário S001, {conf.sum()} janelas dos seis runs. Cada linha soma 100%.",
             size=10.5, color=APAGADO)
    fig.text(0.55, 0.615, f"{ac:.1f}%".replace(".", ","), size=54, weight="bold", color=TINTA)
    fig.text(0.55, 0.565, "das janelas do S001 classificadas certo", size=14, color=TINTA2)
    topicos(fig, 0.55, 0.50, [
        ("Compare com o chute",
         f"Quem respondesse sempre descanso acertaria {base:.1f}%".replace(".", ",")
         + ", porque metade das janelas é descanso. A rede está bem acima disso, e longe de 100%."),
        ("Onde ela erra",
         "O erro mais comum é confundir uma mão com a outra ou tomar mão por descanso. Com três "
         "eletrodos, o sinal de um lado e do outro é parecido."),
        ("No capacete o número cai",
         "Na gravação S005 do projeto, feita com a Cyton, o acerto é de 66,7%."),
    ], larg=50, size=12.5)


def p_tempo_real(D):
    rt = D["rt"]
    inf = rt["inf"]
    fig = nova("6 · A decisão", "Ao vivo: uma janela que desliza e um decisor que segura",
               "A cada 0,1 s o programa pega os últimos 4,1 s de sinal, passa pela rede e olha a chance "
               "de mão direita. Sessão real S012_RT4, com a órtese ligada.", ativa=5)
    t = inf["t_s"].to_numpy()
    ax = fig.add_axes([0.08, 0.455, 0.86, 0.225])
    ax.set_xlim(t[0], t[-1])
    faixas(ax, rt["ev"])
    ax.plot(t, inf["p_right"] * 100, color=AZUL, lw=1.8)
    ax.axhline(cfg.LIMIAR * 100, color=TINTA, lw=1.2, ls=(0, (3, 3)))
    ax.text(t[-1], cfg.LIMIAR * 100 + 2, "limiar de 70% ", ha="right", va="bottom", size=11, color=TINTA2)
    ax.set_ylim(0, 105)
    ax.set_ylabel("chance de mão direita (%)")
    ax.set_xticklabels([])
    limpar(ax)
    fecha = inf[inf["acao"] == "FECHAR"]
    ax.plot(fecha["t_s"], fecha["p_right"] * 100, "o", ms=10, color=LARANJA, mec="white", mew=1.6, zorder=5)

    f0 = fecha.iloc[0]
    ax.annotate("fechou fora de hora", (f0["t_s"] - 0.15, f0["p_right"] * 100 + 2), (t[0] + 0.3, 97),
                size=11, color=TINTA, va="center", arrowprops=dict(arrowstyle="-", color=APAGADO, lw=1))

    ax2 = fig.add_axes([0.08, 0.315, 0.86, 0.075])
    ax2.set_xlim(t[0], t[-1])
    ax2.set_ylim(0, 1)
    abre = inf[inf["acao"] == "ABRIR"]["t_s"].to_numpy()
    for tf in fecha["t_s"]:
        ta = abre[abre > tf][0] if (abre > tf).any() else t[-1]
        ax2.add_patch(Rectangle((tf, 0.15), ta - tf, 0.7, facecolor=LARANJA, edgecolor="none"))
        ax2.add_patch(Rectangle((ta, 0.15), cfg.REFRATARIO_S, 0.7, facecolor="#f3c9b6", edgecolor="none"))
    ax2.set_yticks([])
    ax2.spines["left"].set_visible(False)
    ax2.tick_params(length=0)
    ax2.set_xlabel("tempo da sessão (s)")
    fig.text(0.08, 0.405, "órtese", size=12, weight="bold", color=TINTA)
    fig.add_artist(Rectangle((0.135, 0.406), 0.014, 0.014, transform=fig.transFigure, facecolor=LARANJA, edgecolor="none"))
    fig.text(0.153, 0.405, "fechada (trava de 2 s)", size=11, color=TINTA2)
    fig.add_artist(Rectangle((0.305, 0.406), 0.014, 0.014, transform=fig.transFigure, facecolor="#f3c9b6", edgecolor="none"))
    fig.text(0.323, 0.405, "tempo morto de 2 s, sem poder fechar", size=11, color=TINTA2)
    fig.add_artist(Circle((0.578, 0.413), 0.006, transform=fig.transFigure, facecolor=LARANJA, edgecolor="none"))
    fig.text(0.59, 0.405, "comando de fechar", size=11, color=TINTA2)
    topicos(fig, 0.06, 0.225, [
        ("Limiar",
         "Só vale como intenção se a chance passar de 70%. Abaixo disso a janela conta como "
         "descanso."),
    ], larg=38, size=12.5)
    topicos(fig, 0.37, 0.225, [
        ("Trava e tempo morto",
         "Depois de fechar, a órtese fica 2 s fechada sem perguntar nada. Depois de abrir, "
         "espera mais 2 s. Sem isso ela ficaria batendo."),
    ], larg=38, size=12.5)
    topicos(fig, 0.68, 0.225, [
        ("O atraso tem explicação",
         "O comando sai uns 2 s depois da dica. É o tempo de a transição chegar ao meio da "
         "janela, onde a rede aprendeu a vê-la."),
    ], larg=38, size=12.5)


def p_ortese(D):
    fig = nova("7 · A órtese", "Do computador ao servo",
               "O computador manda uma letra por vez em vez de um ângulo pronto. Cada letra faz o ESP32 "
               "girar o servo um passo, e o firmware responde com o ângulo em que ficou.", ativa=6)
    ax = quadro(fig, [0.04, 0.455, 0.92, 0.245])
    H = ax.get_ylim()[1]
    itens = [("Python", "nucleo/servo.py\nmanda 'L', 'R' ou 'C'", PAINEL),
             ("Bluetooth ou USB", "o mesmo protocolo\nnos dois caminhos", PAINEL),
             ("ESP32", "soma ou subtrai o passo\ne responde angle=<n>", "#fde9df"),
             ("Sinal PWM", "um pulso a cada 20 ms\nno pino 18", "#fde9df"),
             ("Servo", "a largura do pulso\ndiz o ângulo", "#fde9df"),
             ("Órtese", "a haste puxa\no dedo", "#fde9df")]
    w, gap = 13.6, 3.2
    for i, (a, b, cor) in enumerate(itens):
        x = 0.3 + i * (w + gap)
        caixa(ax, x, 2, w, H - 4, a, b, cor=cor, size=12.5)
        if i:
            seta(ax, (x - gap + 0.3, H / 2), (x - 0.3, H / 2))

    # movimento: degraus de 1 grau contra o tempo
    ax = fig.add_axes([0.07, 0.115, 0.38, 0.245])
    for intervalo, cor, nome in ((0.01, AZUL, "espera de 10 ms: 100°/s"), (0.03, "#9ec0ea", "espera de 30 ms: 33°/s")):
        n = 30
        tt = np.arange(n + 1) * intervalo
        ang = 90 - np.arange(n + 1)
        ax.step(np.append(tt, 1.0), np.append(ang, ang[-1]), where="post", color=cor, lw=2)
        ax.text(tt[-1] + 0.02 if intervalo < 0.02 else tt[18] + 0.03, 62.5 if intervalo < 0.02 else 73.5,
                nome, size=11, color=TINTA, va="center")
    ax.set_xlim(0, 1.0)
    ax.set_ylim(57, 92)
    ax.set_xlabel("tempo (s)")
    ax.set_ylabel("ângulo pedido (graus)")
    limpar(ax)
    ax.set_title("Fechar 30 graus em passos de 1 grau (calculado)", loc="left", size=12.5,
                 weight="bold", color=TINTA, pad=8)

    # PWM: dois pulsos
    ax = fig.add_axes([0.53, 0.115, 0.19, 0.245])
    for k, (ang, cor) in enumerate(((90, "#b9b8b2"), (60, LARANJA))):
        larg_ms = (500 + ang / 180 * 2000) / 1000
        t = [0, 0, larg_ms, larg_ms, 4]
        v = [0, 1, 1, 0, 0]
        ax.plot(t, np.array(v) * 0.7 + (1 - k) * 1.1, color=cor, lw=2.4)
        ax.text(larg_ms + 0.15, (1 - k) * 1.1 + 0.38, f"{ang}°: pulso de {larg_ms:.2f} ms".replace(".", ","),
                size=11, color=TINTA, va="center")
    ax.set_xlim(-0.2, 4)
    ax.set_ylim(-0.2, 2.1)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.tick_params(length=0)
    ax.set_xlabel("tempo (ms)")
    ax.set_title("O que o servo recebe", loc="left", size=12.5, weight="bold", color=TINTA, pad=8)
    topicos(fig, 0.755, 0.40, [
        "Repouso é 90°, o centro do curso. Fechar é andar para um dos lados.",
        "A velocidade é o passo dividido pela espera entre as letras.",
        "Se o Bluetooth cai, o firmware volta ao repouso sozinho.",
    ], larg=27, size=12)


def p_limites(D):
    fig = nova("Resultados", "O que já funciona e o que ainda falha",
               "É um protótipo de pesquisa. A órtese fecha quando deve na maior parte das vezes, e "
               "também fecha sozinha de vez em quando.")
    ax = fig.add_axes([0.07, 0.19, 0.40, 0.44])
    lim = ["0,60", "0,70", "0,80"]
    det = [73, 68, 43]
    falso = [60 / 16, 60 / 29, 60 / 53]
    ax.bar(np.arange(3) - 0.0, det, width=0.5, color=AZUL)
    for i in range(3):
        ax.text(i, det[i] + 2, f"{det[i]}%", ha="center", size=15, weight="bold", color=TINTA)
        ax.text(i, -19, "fecha sozinha\n" + f"{falso[i]:.1f}".replace(".", ",") + " vezes por min",
                ha="center", va="top", size=12, color=TINTA2, linespacing=1.4)
    ax.set_xticks(range(3))
    ax.set_xticklabels([f"limiar {x}" for x in lim], size=12.5, color=TINTA)
    ax.set_ylim(0, 100)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.tick_params(length=0)
    ax.set_title("Intenções detectadas", loc="left", size=13, weight="bold", color=TINTA, pad=10)
    ax.xaxis.set_tick_params(pad=6)
    fig.text(0.07, 0.065, "Medido em 6 runs contínuos do PhysioNet (44 dicas, 740 s).", size=10, color=APAGADO)
    ax.set_position([0.07, 0.27, 0.40, 0.36])
    topicos(fig, 0.54, 0.69, [
        ("O limiar é uma troca",
         "Baixar o limiar detecta mais intenções e também faz a mão fechar sozinha mais vezes. "
         "Numa órtese, fechar sem querer é o erro que machuca."),
        ("Com uma pessoa de verdade",
         "Na sessão S012_RT4, com limiar de 0,70, a órtese fechou em 3 de 5 tentativas, com "
         "atraso mediano de 2,4 s, e fechou uma vez fora de hora."),
        ("O que limita",
         "São só três eletrodos, e o sinal de imagética é fraco e muda de pessoa para pessoa. O "
         "modelo foi treinado quase todo com outras cabeças."),
        ("Regras de uso",
         "A órtese tem de sair da mão com facilidade. Ninguém fica sozinho com o sistema ligado. "
         "O curso do servo é conferido com a órtese vazia antes de cada sessão."),
    ], larg=52, size=12.5)


def p_resumo(D):
    fig = nova("Resumo", "Dois segundos, do pensamento ao movimento",
               "A mesma história em ordem de tempo, para uma única intenção de fechar a mão direita.")
    ax = quadro(fig, [0.05, 0.33, 0.90, 0.37])
    H = ax.get_ylim()[1]
    y = H - 7
    ax.plot([3, 97], [y, y], color=TINTA2, lw=2)
    marcos = [(4, "instante zero", "A pessoa começa a\nimaginar o movimento.\nO córtex motor esquerdo\nmuda de ritmo."),
              (23, "a cada 4 ms", "A Cyton mede C3, C4\ne Cz e manda cada\nleitura por rádio\nao computador."),
              (42, "espera de uns 2 s", "A transição chega ao\nmeio da janela de 4,1 s.\nO sinal é filtrado e\nreduzido a 128 Hz."),
              (61, "milissegundos", "A rede devolve três\nprobabilidades. A de mão\ndireita passa de 70% e\no decisor manda fechar."),
              (80, "0,3 s de giro", "O ESP32 recebe uma\nletra por grau (30, num\ncurso de 30 graus) e a\nórtese fecha a mão."),
              ]
    for x, tempo, txt in marcos:
        ax.add_patch(Circle((x, y), 1.3, facecolor=AZUL, edgecolor="white", lw=2, zorder=4))
        ax.text(x - 1, y + 3, tempo, ha="left", va="bottom", size=14, weight="bold", color=TINTA)
        ax.text(x - 1, y - 3.5, txt, ha="left", va="top", size=12.5, color=TINTA2, linespacing=1.45)
    fig.text(0.05, 0.27, textwrap.fill(
        "Depois disso a órtese fica fechada por 2 s, abre e espera mais 2 s antes de aceitar um "
        "novo comando. O atraso de 2 s vem do método, já que a conta no computador leva "
        "milissegundos: a rede reconhece a transição quando ela está no meio da janela.", 135),
        size=14, color=TINTA, va="top", linespacing=1.5)


PAGINAS = [p_capa, p_visao, p_comando, p_neuronio, p_imagetica, p_eletrodos, p_placa,
           p_physionet, p_cyton, p_limpeza, p_janela, p_rede_mapa, p_conv_numeros, p_conv_real,
           p_banco, p_espacial, p_potencia, p_saida, p_acerto, p_tempo_real, p_ortese,
           p_limites, p_resumo]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--saida", default=os.path.join(RAIZ, "apresentacao",
                                                    "do_pensamento_ao_movimento.pdf"))
    ap.add_argument("--png", default=None, help="pasta para salvar uma imagem por página")
    a = ap.parse_args()

    print("lendo os dados e o modelo...")
    D = carregar_tudo()
    for p in PAGINAS:
        p(D)
    with PdfPages(a.saida) as pdf:
        for fig in _paginas:
            pdf.savefig(fig, facecolor=FUNDO)
        info = pdf.infodict()
        info["Title"] = "Do pensamento ao movimento"
        info["Subject"] = "BCI de imagética motora para órtese de mão"
    if a.png:
        os.makedirs(a.png, exist_ok=True)
        for i, fig in enumerate(_paginas, 1):
            fig.savefig(os.path.join(a.png, f"p{i:02d}.png"), dpi=75, facecolor=FUNDO)
    print(f"{len(_paginas)} páginas em {a.saida}")


if __name__ == "__main__":
    main()
