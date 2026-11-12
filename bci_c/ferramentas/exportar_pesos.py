# -*- coding: utf-8 -*-
"""exportar_pesos.py -- .pt do PyTorch -> modelo.bin que o C le.

    python ferramentas/exportar_pesos.py
    python ferramentas/exportar_pesos.py --pt ../modelos/cnn2d_transicao.pt --saida artefatos/modelo.bin

Roda SO no PC, uma vez. O Pi nunca ve um .pt, nem o torch.

Duas coisas acontecem aqui, e as duas sao de proposito:

1) Os BatchNorm sao colapsados de 4 tensores para 2. Em eval,
       y = gama*(x - mu)/sqrt(var + eps) + beta
   e uma afim, entao basta gravar
       s = gama/sqrt(var + eps)      b = beta - s*mu
   e o C aplica `s*x + b`. Isso NAO e dobrar o BN na convolucao: cada BN continua
   sendo uma camada propria no rede.c. Dobrar economizaria ~3% do tempo e exigiria
   algebra entre camadas (o bias de um BN atravessando a conv seguinte), que e
   exatamente onde esse tipo de port costuma ganhar um bug silencioso.

2) A geometria nao e chutada: cada numero e lido do modulo correspondente do
   modelo carregado (kernel do AvgPool, largura da conv separavel, output_size do
   AdaptiveAvgPool, in_features do Linear). Se alguem mudar o modelo.py, este
   script quebra alto em vez de gravar um binario sutilmente errado.
"""
from __future__ import annotations

import argparse
import os
import struct
import sys

import numpy as np
import torch
import torch.nn as nn

AQUI = os.path.dirname(os.path.abspath(__file__))
PROJETO = os.path.dirname(AQUI)                 # bci_c/
RAIZ = os.path.dirname(PROJETO)                 # repositorio Python
sys.path.insert(0, RAIZ)

MAGIA = b"BCIC0001"


def f32(t):
    """Tensor ou array -> numpy float32 contiguo e achatado."""
    if isinstance(t, torch.Tensor):
        t = t.detach().cpu().numpy()
    return np.ascontiguousarray(np.asarray(t, dtype=np.float32)).ravel()


def colapsar_bn(bn):
    """BatchNorm2d em eval -> (s, b) tais que y = s*x + b, por canal."""
    assert isinstance(bn, nn.BatchNorm2d), f"esperava BatchNorm2d, achei {type(bn)}"
    gama = bn.weight.detach().cpu().numpy().astype(np.float64)
    beta = bn.bias.detach().cpu().numpy().astype(np.float64)
    mu = bn.running_mean.detach().cpu().numpy().astype(np.float64)
    var = bn.running_var.detach().cpu().numpy().astype(np.float64)
    s = gama / np.sqrt(var + bn.eps)
    b = beta - s * mu
    return s.astype(np.float32), b.astype(np.float32)


def main():
    ap = argparse.ArgumentParser(description="Exporta um .pt para o modelo.bin do C.")
    ap.add_argument("--pt", default=os.path.join(RAIZ, "modelos", "cnn2d_transicao.pt"))
    ap.add_argument("--saida", default=os.path.join(PROJETO, "artefatos", "modelo.bin"))
    a = ap.parse_args()

    from nucleo.modelo import carregar

    modelo, amostras, esquema = carregar(a.pt)
    ck = torch.load(a.pt, map_location="cpu", weights_only=False)

    bloco, cabeca = modelo.bloco, modelo.cabeca

    # --- geometria, lida dos modulos, nao chutada -------------------------
    banco = bloco[0]
    assert hasattr(banco, "ramos"), "bloco[0] deveria ser o BancoTemporal"
    kernels = [int(r.weight.shape[3]) for r in banco.ramos]
    n_por_escala = int(banco.ramos[0].weight.shape[0])
    n_escalas = len(kernels)
    F = n_por_escala * n_escalas

    bn1, conv_esp, bn2 = bloco[1], bloco[2], bloco[3]
    pool_mod, conv_dep, conv_pw, bn3 = bloco[5], bloco[8], bloco[9], bloco[10]
    linear = cabeca[1]

    assert isinstance(pool_mod, nn.AvgPool2d), f"bloco[5] = {type(pool_mod)}"
    assert isinstance(conv_esp, nn.Conv2d) and isinstance(conv_dep, nn.Conv2d)
    assert isinstance(conv_pw, nn.Conv2d) and isinstance(linear, nn.Linear)

    ksz = pool_mod.kernel_size
    pool = int(ksz[1] if isinstance(ksz, (tuple, list)) else ksz)
    ksep = int(conv_dep.weight.shape[3])
    n_canais = int(conv_esp.weight.shape[2])
    Fs = int(conv_esp.weight.shape[0])
    depth = Fs // F
    n_saida = int(conv_pw.weight.shape[0])
    ad_saida = int(modelo.pool.output_size[1])
    n_flat = int(linear.in_features)
    n_classes = int(linear.out_features)

    assert Fs == F * depth, f"Fs={Fs} nao e F*depth={F}*{depth}"
    assert n_flat == n_saida * ad_saida, f"n_flat={n_flat} != {n_saida}*{ad_saida}"
    assert all(k % 2 == 1 for k in kernels), "kernels tem que ser impares"
    assert conv_esp.bias is None and conv_dep.bias is None and conv_pw.bias is None, \
        "as convolucoes deveriam ser bias=False; o C nao le bias nelas"

    # --- normalizacao do checkpoint --------------------------------------
    mu, sd = ck.get("norm_mu"), ck.get("norm_sd")
    assert mu is not None and sd is not None, \
        "o checkpoint nao tem norm_mu/norm_sd -- a inferencia ficaria sem escala"
    mu, sd = f32(mu), f32(sd)
    assert mu.size == n_canais and sd.size == n_canais, \
        f"norm tem {mu.size} valores para {n_canais} canais"

    # --- cabecalho --------------------------------------------------------
    cab = bytearray(MAGIA)
    def i32(v):
        cab.extend(struct.pack("<i", int(v)))
    for v in (n_canais, n_classes, amostras, n_por_escala, n_escalas):
        i32(v)
    for k in kernels:
        i32(k)
    for v in (depth, n_saida, ksep, pool, ad_saida, n_flat):
        i32(v)

    # --- pesos, na ORDEM que o rede.c le --------------------------------
    partes = [mu, sd]
    for r in banco.ramos:                       # (n_por_escala,1,1,k) -> [f][k]
        partes.append(f32(r.weight))
    s1, b1 = colapsar_bn(bn1);   partes += [s1, b1]
    partes.append(f32(conv_esp.weight))         # (Fs,1,C,1) -> [o][c]
    s2, b2 = colapsar_bn(bn2);   partes += [s2, b2]
    partes.append(f32(conv_dep.weight))         # (Fs,1,1,ksep) -> [o][j]
    partes.append(f32(conv_pw.weight))          # (n_saida,Fs,1,1) -> [o][i]
    s3, b3 = colapsar_bn(bn3);   partes += [s3, b3]
    partes.append(f32(linear.weight))           # (K,n_flat)
    partes.append(f32(linear.bias))

    esperado = (2 * n_canais
                + sum(n_por_escala * k for k in kernels)
                + 2 * F
                + Fs * n_canais
                + 2 * Fs
                + Fs * ksep
                + n_saida * Fs
                + 2 * n_saida
                + n_classes * n_flat + n_classes)
    corpo = np.concatenate(partes)
    assert corpo.size == esperado, f"gravei {corpo.size} floats, o C espera {esperado}"
    assert np.isfinite(corpo).all(), "ha NaN ou inf nos pesos exportados"

    os.makedirs(os.path.dirname(a.saida) or ".", exist_ok=True)
    with open(a.saida, "wb") as f:
        f.write(bytes(cab))
        f.write(corpo.tobytes())

    n_par = sum(p.numel() for p in modelo.parameters())
    print("=" * 70)
    print(f"  origem ....: {os.path.relpath(a.pt, RAIZ)}")
    print(f"  esquema ...: {esquema.get('nome', '?')} | treinado em "
          f"{esquema.get('treinado_em', '?')}")
    print(f"  geometria .: C={n_canais} T={amostras} K={n_classes} | "
          f"{n_escalas} escalas x {n_por_escala} filtros, kernels {kernels}")
    print(f"             : depth={depth} Fs={Fs} n_saida={n_saida} ksep={ksep} "
          f"pool={pool} ad={ad_saida} n_flat={n_flat}")
    print(f"  normaliz. .: mu={np.array2string(mu, precision=3)} "
          f"sd={np.array2string(sd, precision=3)}")
    print(f"  parametros : {n_par:,} no .pt | {corpo.size:,} floats no .bin "
          f"(+{corpo.size - n_par} = norm_mu e norm_sd, que nao sao parametros)")
    print(f"  saida .....: {a.saida}  ({len(cab) + corpo.nbytes:,} bytes)")
    print("=" * 70)


if __name__ == "__main__":
    main()
