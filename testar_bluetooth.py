# -*- coding: utf-8 -*-
"""
testar_bluetooth.py — confere, com a órtese de verdade, se o controle pelo Bluetooth está certo.

    python testar_bluetooth.py                # acha o ESP32 "BCI_Ortese" sozinho
    python testar_bluetooth.py --porta COM6   # força uma porta
    python testar_bluetooth.py --ciclos 5     # mais ciclos de fechar e abrir
    python testar_bluetooth.py --detalhe      # mostra cada byte trocado
    python testar_bluetooth.py --amplitude 21 # confere um curso menor que o do config
    python testar_bluetooth.py --intervalo 0.02   # servo mais lento
    python testar_bluetooth.py --porta COM4 --ciclos 5 --detalhe   # combinando; serve no cabo também

Usa as MESMAS peças do tempo_real.py: `nucleo.servo.Ortese` fala com o ESP32 e
`nucleo.servo.Decisor` decide quando fechar. Só o capacete é trocado por probabilidades
inventadas. Se este teste passa, o que falta para o sistema completo é o EEG.

Quatro etapas, e cada uma diz o que você deve VER a órtese fazer:

  1. conexão    acha a porta, conecta e mede quanto o eco do firmware demora
  2. movimento  fecha e abre algumas vezes, conferindo o ângulo que o firmware informa
  3. limite     pede além da amplitude, para os dois lados, e confere que o servo não vai
  4. decisão    roda o Decisor em tempo real: fecha, trava, abre, ignora uma intenção no
                refratário e fecha de novo depois dele

No fim — ou no Ctrl-C — a órtese volta ao repouso e a conexão é fechada.
Mantenha a órtese VAZIA durante o teste.
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time

import numpy as np

import config as cfg
from nucleo import servo

REPOUSO, FECHADO = cfg.SERVO_REPOUSO, cfg.ANGULO_FECHADO
falhas = []


def checar(descricao, ok, se_falhar=""):
    print(f"  {'ok   ' if ok else 'FALHA'} {descricao}" + ("" if ok else f"  ({se_falhar})"))
    if not ok:
        falhas.append(descricao)
    return ok


def etapa(titulo, veja):
    print(f"\n--- {titulo} " + "-" * (69 - len(titulo)))
    print(f"  veja: {veja}")


def angulo_no_firmware(o):
    """Pergunta ao ESP32 onde o servo está, sem mover: reenvia o dígito do passo, e o
    firmware só devolve o eco `angle=<n>`. None se ele não respondeu."""
    o._limpar_entrada()
    o._enviar(str(o.passo).encode())
    return o._ler_eco()


def conexao(o, segundos):
    bt = servo.portas_bluetooth()
    print(f"  conectou em {segundos:.1f} s")
    if o.porta in bt:
        print(f"  pelo Bluetooth: '{bt[o.porta]}' na {o.porta}")
    else:
        print(f"  [aviso] {o.porta} não é porta Bluetooth: o teste vai rodar pelo cabo")

    tempos, angulo = [], None
    for _ in range(5):
        t0 = time.monotonic()
        angulo = angulo_no_firmware(o)
        if angulo is None:
            break
        tempos.append(time.monotonic() - t0)
    if not checar("o firmware responde", len(tempos) == 5,
                  "sem eco: confira a porta e se o firmware com Bluetooth está gravado"):
        return False
    print(f"  eco em {statistics.median(tempos) * 1000:.0f} ms (pior {max(tempos) * 1000:.0f} ms, "
          f"prazo {cfg.SERVO_TIMEOUT_ECO_S * 1000:.0f} ms)")
    checar(f"o servo está no repouso, firmware em {angulo}°", angulo == REPOUSO,
           f"esperado {REPOUSO}°")
    return True


def movimento(o, ciclos):
    etapa("2. movimento", f"fecha até {FECHADO}° e abre até {REPOUSO}°, {ciclos} vez(es)")
    for i in range(1, ciclos + 1):
        for nome, comando, alvo in (("fechou", o.fechar, FECHADO), ("abriu", o.abrir, REPOUSO)):
            t0 = time.monotonic()
            comando()
            dt = time.monotonic() - t0
            fw = angulo_no_firmware(o)
            checar(f"ciclo {i}: {nome} em {dt:.2f} s, firmware em {fw}°", fw == alvo,
                   f"esperado {alvo}°")
            time.sleep(1.0)                    # tempo para você ver a posição


def limite(o):
    alem = FECHADO + (60 if FECHADO > REPOUSO else -60)
    espelho = 2 * REPOUSO - FECHADO            # o mesmo tanto, para o lado que abre
    etapa("3. limite", f"fecha uma vez e para em {FECHADO}°; abre e não passa do repouso")
    o.fechar()
    o.ir_para(alem)
    fw = angulo_no_firmware(o)
    checar(f"pedir {alem}° parou em {fw}°", fw == FECHADO, f"esperado {FECHADO}°")
    time.sleep(1.0)
    o.abrir()
    o.ir_para(espelho)
    fw = angulo_no_firmware(o)
    checar(f"pedir {espelho}° parou em {fw}°", fw == REPOUSO, f"esperado {REPOUSO}°")


def decisao(o):
    seg, ref = cfg.SEGURA_S, cfg.REFRATARIO_S
    i1 = 1.0                                   # 1ª intenção: fecha
    i2 = i1 + seg + ref / 2                    # 2ª, no meio do refratário: tem que ser ignorada
    i3 = i1 + seg + ref + 1.0                  # 3ª, depois do refratário: fecha de novo
    fim = i3 + seg + 1.5
    intencoes = (i1, i2, i3) if ref >= 1.0 else (i1, i3)
    etapa("4. decisão, igual ao tempo_real.py",
          f"fecha, fica {seg:g} s fechada e abre; depois, de novo ({fim:.0f} s)")

    dec = servo.Decisor()
    idx = {v: k for k, v in cfg.CLASSES.items()}[dec.classe]
    acoes, avisou = [], False
    t0 = time.monotonic()
    while (t := time.monotonic() - t0) < fim:
        # o "capacete": 0,95 na classe que aciona durante cada intenção, senão descanso
        intencao = any(ini <= t < ini + 0.5 for ini in intencoes)
        prob = np.full(len(cfg.CLASSES), 0.025)
        prob[idx if intencao else 0] = 0.95
        _, acao = dec.passo(prob, time.monotonic())
        if acao == "FECHAR":
            print(f"  t={t:4.1f} s  intenção '{dec.classe}' ({prob[idx]:.2f})   >>> FECHA")
            acoes.append((acao, o.fechar(), FECHADO))
        elif acao == "ABRIR":
            print(f"  t={t:4.1f} s  fim da trava              <<< abre")
            acoes.append((acao, o.abrir(), REPOUSO))
        elif intencao and dec.travado_ate is None and not avisou:
            print(f"  t={t:4.1f} s  intenção no refratário    (ignorada, como deve)")
            avisou = True
        time.sleep(cfg.PASSO_S)

    nomes = [a for a, _, _ in acoes]
    checar("ações: FECHAR, ABRIR, FECHAR, ABRIR", nomes == ["FECHAR", "ABRIR", "FECHAR", "ABRIR"],
           f"veio {nomes}")
    checar("cada ação chegou ao ângulo certo", all(ang == alvo for _, ang, alvo in acoes),
           f"{[(a, ang) for a, ang, _ in acoes]}")
    fw = angulo_no_firmware(o)
    checar(f"terminou aberta, firmware em {fw}°", fw == REPOUSO, f"esperado {REPOUSO}°")


def main():
    ap = argparse.ArgumentParser(description="Testa o controle da órtese pelo Bluetooth.")
    ap.add_argument("--porta", default=None, help="força uma porta (padrão: acha o BCI_Ortese)")
    ap.add_argument("--ciclos", type=int, default=3, help="ciclos de fechar e abrir (etapa 2)")
    ap.add_argument("--detalhe", action="store_true", help="mostra cada byte trocado")
    ap.add_argument("--intervalo", type=servo.arg_intervalo, default=None,
                    help=servo.AJUDA_INTERVALO)
    ap.add_argument("--amplitude", type=servo.arg_amplitude, default=None,
                    help="quanto fecha, em graus (teto: config.SERVO_AMPLITUDE)")
    a = ap.parse_args()
    global FECHADO                             # as etapas conferem contra ele
    amp = servo.amplitude_efetiva(a.amplitude)
    FECHADO = REPOUSO + (amp if cfg.SERVO_SENTIDO_FECHA == "R" else -amp)

    print("=" * 74)
    print("  TESTE DA ÓRTESE PELO BLUETOOTH")
    print("=" * 74)
    print(f"  aparelho ..: '{cfg.SERVO_BT_NOME}'")
    print(f"  curso .....: {REPOUSO}° -> {FECHADO}° ({amp}°) | passo {cfg.SERVO_PASSO_GRAUS}° | "
          f"fecha com '{cfg.SERVO_SENTIDO_FECHA}'")
    print(f"  decisão ...: limiar {cfg.LIMIAR:g} | trava {cfg.SEGURA_S:g} s | "
          f"refratário {cfg.REFRATARIO_S:g} s | passo {cfg.PASSO_S:g} s")
    try:
        input("\n  Órtese VAZIA? Enter para começar, Ctrl-C para cancelar... ")
    except KeyboardInterrupt:
        print("\n  Cancelado.")
        return 1

    o = None
    try:
        etapa("1. conexão", "nada se move além de, no máximo, ir para o repouso")
        t0 = time.monotonic()
        o = servo.Ortese(porta=a.porta, detalhe=a.detalhe, amplitude=a.amplitude,
                         intervalo=a.intervalo)
        if conexao(o, time.monotonic() - t0):
            movimento(o, a.ciclos)
            limite(o)
            decisao(o)
    except KeyboardInterrupt:
        falhas.append("teste interrompido antes do fim")
    except OSError as e:                       # a conexão caiu e não voltou
        falhas.append(f"a conexão caiu: {e}")
    finally:
        if o is not None:
            o.fechar_conexao()                 # volta ao repouso antes de soltar a porta
            print("\n  Órtese no repouso, conexão fechada.")

    print("\n" + "=" * 74)
    if falhas:
        print(f"  {len(falhas)} falha(s):")
        for f in falhas:
            print(f"    - {f}")
        print("  NÃO use com a órtese vestida até resolver.")
    else:
        print("  Tudo certo: o PC comanda a órtese pelo Bluetooth com a lógica do tempo_real.py.")
        print("  Falta só o capacete:  python tempo_real.py")
    print("=" * 74)
    return 1 if falhas else 0


if __name__ == "__main__":
    sys.exit(main())
