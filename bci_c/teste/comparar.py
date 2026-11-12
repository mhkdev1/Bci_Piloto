# -*- coding: utf-8 -*-
"""comparar.py -- o C e o PyTorch veem os MESMOS pesos. As contas tem que bater.

    python teste/comparar.py

Este e o unico Python do projeto C, e existe por um motivo so: dar uma
verdade-base para o forward. Com os pesos do cnn2d_transicao.pt carregados dos
dois lados, qualquer divergencia acima do erro de arredondamento esta no rede.c
-- nao no pre-processamento, nao no treino, nao na aquisicao. Nao ha o que
interpretar.

O que NAO se exige aqui e igualdade bit a bit: a ordem das somas nas convolucoes
difere entre o C e o BLAS do torch, e isso muda os ultimos bits. O que se exige e
que a diferenca seja irrelevante para a decisao -- contra um LIMIAR de 0,65, uma
discrepancia de 1e-4 nunca inverte um acionamento.

As janelas de teste incluem quatro casos de borda de proposito:
  [0] tudo zero          -- o log de potencia nula bate no piso do SafeLog
  [1] tudo um            -- sinal constante, resposta so dos termos de bias
  [2] escala 100x        -- confere que nada estoura em float32
  [3] escala 1e-4        -- forca o clamp de 1e-6 do SafeLog nos dois lados
"""
from __future__ import annotations

import argparse
import os
import struct
import subprocess
import sys

import numpy as np
import torch

AQUI = os.path.dirname(os.path.abspath(__file__))
PROJETO = os.path.dirname(AQUI)
RAIZ = os.path.dirname(PROJETO)
sys.path.insert(0, RAIZ)


def achar_binario(dado):
    if dado:
        return dado
    for nome in ("bancada.exe", "bancada"):
        p = os.path.join(PROJETO, "artefatos", nome)
        if os.path.exists(p):
            return p
    return None


def main():
    ap = argparse.ArgumentParser(description="Paridade do forward: C vs PyTorch.")
    ap.add_argument("--pt", default=os.path.join(RAIZ, "modelos", "cnn2d_transicao.pt"))
    ap.add_argument("--bin", default=os.path.join(PROJETO, "artefatos", "modelo.bin"))
    ap.add_argument("--bancada", default=None)
    ap.add_argument("--n", type=int, default=96, help="janelas de teste")
    ap.add_argument("--tol", type=float, default=1e-4,
                    help="tolerancia absoluta nas probabilidades")
    a = ap.parse_args()

    bancada = achar_binario(a.bancada)
    if not bancada:
        sys.exit("bancada nao compilada. Rode `make` em bci_c/ primeiro.")
    if not os.path.exists(a.bin):
        sys.exit(f"{a.bin} nao existe. Rode ferramentas/exportar_pesos.py primeiro.")

    from nucleo.modelo import carregar
    modelo, T, _ = carregar(a.pt)
    C, K = modelo.n_canais, modelo.n_classes

    # ---- janelas de teste, deterministicas ------------------------------
    rng = np.random.default_rng(0)
    X = rng.standard_normal((a.n, C, T)).astype(np.float32)
    X[0] = 0.0
    X[1] = 1.0
    X[2] = (rng.standard_normal((C, T)) * 100.0).astype(np.float32)
    X[3] = (rng.standard_normal((C, T)) * 1e-4).astype(np.float32)

    # ---- lado PyTorch ----------------------------------------------------
    with torch.no_grad():
        p_py = torch.softmax(modelo(torch.from_numpy(X)), dim=1).numpy()

    # ---- lado C ----------------------------------------------------------
    tmp = os.path.join(PROJETO, "artefatos")
    os.makedirs(tmp, exist_ok=True)
    f_ent = os.path.join(tmp, "paridade_entrada.bin")
    f_sai = os.path.join(tmp, "paridade_saida.bin")
    with open(f_ent, "wb") as f:
        f.write(struct.pack("<iii", a.n, C, T))
        f.write(np.ascontiguousarray(X, dtype=np.float32).tobytes())

    r = subprocess.run([bancada, "rede", a.bin, f_ent, f_sai],
                       capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit(f"bancada falhou ({r.returncode}):\n{r.stdout}{r.stderr}")

    with open(f_sai, "rb") as f:
        n_c, k_c = struct.unpack("<ii", f.read(8))
        p_c = np.frombuffer(f.read(), dtype=np.float32).reshape(n_c, k_c)
    if (n_c, k_c) != (a.n, K):
        sys.exit(f"o C devolveu ({n_c},{k_c}), esperava ({a.n},{K})")

    # ---- comparacao ------------------------------------------------------
    dif = np.abs(p_py - p_c)
    pior = int(np.unravel_index(np.argmax(dif), dif.shape)[0])
    concorda = int((p_py.argmax(1) == p_c.argmax(1)).sum())
    soma_c = p_c.sum(1)

    print("=" * 72)
    print(f"  janelas .........: {a.n}  (C={C}, T={T}, K={K})")
    print(f"  max |dif| .......: {dif.max():.3e}   (tolerancia {a.tol:.0e})")
    print(f"  media |dif| .....: {dif.mean():.3e}")
    print(f"  argmax igual ....: {concorda}/{a.n}")
    print(f"  softmax soma 1 ..: desvio max {np.abs(soma_c - 1.0).max():.3e}")
    print("-" * 72)
    print(f"  pior janela (#{pior}):")
    print(f"    PyTorch : {np.array2string(p_py[pior], precision=6)}")
    print(f"    C       : {np.array2string(p_c[pior], precision=6)}")
    print("-" * 72)
    for i, rotulo in ((0, "tudo zero"), (1, "tudo um"),
                      (2, "escala 100x"), (3, "escala 1e-4")):
        print(f"  [{i}] {rotulo:<12s} max|dif| {dif[i].max():.3e}   "
              f"argmax py={p_py[i].argmax()} c={p_c[i].argmax()}")
    print("=" * 72)

    ok = dif.max() <= a.tol and concorda == a.n
    print("PARIDADE OK" if ok else "PARIDADE FALHOU")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
