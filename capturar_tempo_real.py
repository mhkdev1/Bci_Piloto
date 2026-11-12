# -*- coding: utf-8 -*-
"""
capturar_tempo_real.py — sessão guiada que aciona a órtese E grava o EEG para treino.

    python capturar_tempo_real.py --sujeito S006                      # sessão real
    python capturar_tempo_real.py --sujeito S006 --simular-ortese     # EEG real, órtese só no terminal
    python capturar_tempo_real.py --ensaio --sem-ortese               # sem hardware nenhum

É a junção do `tempo_real.py --protocolo` com o `capturar.py`, sem mudar nenhum dos dois:

    - as dicas e a agenda são as do modo --protocolo (descanso inicial + N tentativas de
      DESCANSE 4,1 s | IMAGINE A MÃO 4,1 s), e a órtese é acionada pelo mesmo Decisor;
    - o EEG é drenado continuamente, como no capturar.py, e salvo no MESMO formato:

        dados/gravacoes/<sujeito>_<run>/
            continuo.csv      igual ao do capturar.py  -> entra no treinar.py sozinho
            eventos.csv       igual ao do capturar.py
            inferencias.csv   uma linha por inferência (o log do tempo_real.py)
            resumo.txt        detecção, latência e disparos falsos

`--run` não informado vira RT1, RT2, ... (o primeiro livre), então repetir o comando nunca
sobrescreve uma sessão. Com `--ensaio` nada vai para dados/gravacoes/: o EEG é falso e não
pode cair no treino; a saída vai para logs/ENSAIO_<sujeito>_<run>/.

------------------------------------------------------------------------------
DUAS DIFERENÇAS EM RELAÇÃO AO tempo_real.py
------------------------------------------------------------------------------
1. Um único leitor do stream. O tempo_real espia o buffer (get_current_board_data) e o
   capturar o esvazia (get_board_data). Aqui o stream é esvaziado a cada volta do laço,
   tudo vai para a gravação, e a janela da rede é a cauda do que já foi lido.

2. A órtese roda numa thread. `Ortese.fechar()` manda os pulsos um a um e espera o eco,
   o que leva de 1 a 3 s. No laço principal isso atrasaria as dicas e, com elas, os
   instantes gravados em eventos.csv, que são os rótulos do treino.

------------------------------------------------------------------------------
ANTES DE TREINAR COM ESTAS SESSÕES
------------------------------------------------------------------------------
Os rótulos seguem a DICA, não o que a rede detectou: uma tentativa em que a órtese não
fechou continua sendo "mão". Mas o sinal tem algo que a gravação do capturar.py não tem:
a pessoa VÊ a mão fechar (ou não), e o servo e o movimento passivo do dedo podem aparecer
no EEG. Para comparar, treine com e sem estas pastas; para tirar uma do treino, basta
movê-la para fora de dados/gravacoes/.
"""
from __future__ import annotations

import argparse
import csv
import os
import queue
import sys
import threading
import time

import numpy as np
import pandas as pd
import torch

import config as cfg
from capturar import DICAS, cortes_por_janela, detrend_por_janela, mostrar
from nucleo import preproc, servo
from nucleo.modelo import carregar
from tempo_real import CODIGO_TAREFA, agenda_protocolo, dica_em, resumir, texto_resumo


class Acionador(threading.Thread):
    """Executa os comandos da órtese em ordem, fora do laço de captura."""

    def __init__(self, ortese):
        super().__init__(daemon=True)
        self.ortese, self.fila = ortese, queue.Queue()

    def run(self):
        while True:
            cmd = self.fila.get()
            if cmd is None:
                return
            try:
                cmd()
            except Exception as e:
                print(f"  [órtese] comando falhou: {e}")

    def parar(self):
        self.fila.put(None)
        self.join(timeout=10)


def proximo_run(base, sujeito):
    k = 1
    while os.path.exists(os.path.join(base, f"{sujeito}_RT{k}")):
        k += 1
    return f"RT{k}"


def salvar_gravacao(pasta, blocos, eventos, linha_tempo, canais_eeg, taxa):
    """Mesmo formato e mesmo tratamento do capturar.py."""
    bruto = np.concatenate(blocos, axis=1)
    tempos = bruto[linha_tempo, :].astype(np.float64)
    n = bruto.shape[1]
    print(f"Amostras: {n} | {n/taxa:.1f} s de sinal | taxa medida "
          f"{n/(tempos[-1]-tempos[0]):.1f} Hz")

    # Base de tempo pela contagem de amostras, como no capturar.py.
    inicios = [max(0, min(n - 1, int(round((e["t_onset"] - tempos[0]) * taxa)))) for e in eventos]
    espac = np.diff(inicios[1:]) / taxa if len(inicios) > 2 else np.array([])
    if len(espac):
        print(f"Espaçamento entre eventos (sem o bloco inicial): mediana {np.median(espac):.3f} s "
              f"(nominal {cfg.DUR_TAREFA:g} s), desvio {espac.std():.3f} s")
        if np.abs(np.median(espac) - cfg.DUR_TAREFA) > 0.3:
            print("  [aviso] espaçamento fora do nominal — confira a carga da bateria e o link.")

    canais = np.stack([bruto[canais_eeg[cfg.MAPA_CANAIS[c]], :] for c in cfg.CANAIS], axis=1)
    limpo, _ = detrend_por_janela(canais, cortes_por_janela(inicios, n))

    rotulos = np.empty(n, dtype=object)
    rotulos[:] = ""
    for j, ini in enumerate(inicios):
        fim = inicios[j + 1] if j + 1 < len(inicios) else n
        rotulos[ini:fim] = eventos[j]["code"]

    saida = pd.DataFrame({"sample_index": np.arange(n), "timestamp": tempos, "label": rotulos})
    for k, c in enumerate(cfg.CANAIS):
        saida[f"{c}_uV"] = np.round(limpo[:, k], 1)
    saida.to_csv(os.path.join(pasta, "continuo.csv"), index=False)
    pd.DataFrame([{"trial_index": e["trial_index"], "code": e["code"],
                   "amostra_inicio": ini, "t_s": round(ini / taxa, 4)}
                  for e, ini in zip(eventos, inicios)]).to_csv(
        os.path.join(pasta, "eventos.csv"), index=False)


def main():
    ap = argparse.ArgumentParser(description="Sessão guiada: aciona a órtese e grava para treino.")
    ap.add_argument("--sujeito", required=True, help="código do participante, ex.: S006")
    ap.add_argument("--run", default=None, help="nome da sessão (padrão: RT1, RT2, ... livre)")
    ap.add_argument("--modelo", default=cfg.MODELO_PADRAO)
    ap.add_argument("--porta", default=None, help="porta da Cyton (padrão: config.PORTA_EEG)")
    ap.add_argument("--sem-ortese", action="store_true", help="não aciona nada, só imprime")
    ap.add_argument("--simular-ortese", action="store_true", help="imprime os comandos, sem serial")
    ap.add_argument("--passo", type=float, default=cfg.PASSO_S)
    ap.add_argument("--limiar", type=float, default=cfg.LIMIAR)
    ap.add_argument("--classe", default=cfg.CLASSE_ACIONA, choices=sorted(CODIGO_TAREFA))
    ap.add_argument("--amplitude", type=servo.arg_amplitude, default=None,
                    help=f"graus de fechamento (até {cfg.SERVO_AMPLITUDE})")
    ap.add_argument("--intervalo", type=servo.arg_intervalo, default=None,
                    help=servo.AJUDA_INTERVALO)
    ap.add_argument("--tentativas", type=int, default=20)
    ap.add_argument("--descanso-inicial", type=float, default=120.0,
                    help="bloco de descanso inicial, em s (0 desliga)")
    ap.add_argument("--sem-tela", action="store_true", help="dicas só no terminal")
    ap.add_argument("--ensaio", action="store_true", help="placa sintética: EEG falso, sem capacete")
    a = ap.parse_args()

    from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds

    base = os.path.join(cfg.DIR_LOGS, "ENSAIO") if a.ensaio else cfg.DIR_GRAVACOES
    a.run = a.run or proximo_run(base, a.sujeito)
    pasta = os.path.join(base, f"{a.sujeito}_{a.run}")
    if os.path.exists(pasta):
        sys.exit(f"{pasta} já existe. Use outro --run (ou omita para pegar o próximo livre).")

    modelo, amostras, esquema = carregar(a.modelo)
    if amostras != cfg.AMOSTRAS:
        sys.exit(f"O modelo espera {amostras} amostras e o config monta {cfg.AMOSTRAS}.")

    params = BrainFlowInputParams()
    params.serial_port = a.porta or cfg.PORTA_EEG
    placa_id = (BoardIds.SYNTHETIC_BOARD if a.ensaio else BoardIds.CYTON_BOARD).value
    placa = BoardShim(placa_id, params)
    canais_eeg = BoardShim.get_eeg_channels(placa_id)
    linha_tempo = BoardShim.get_timestamp_channel(placa_id)
    linhas = [canais_eeg[cfg.MAPA_CANAIS[c]] for c in cfg.CANAIS]
    taxa = float(cfg.TAXA_PLACA)
    largura = int(np.ceil(amostras / cfg.TAXA_IA * taxa))

    agenda = agenda_protocolo(a.classe, a.tentativas, a.descanso_inicial)
    total = agenda[-1][0] + agenda[-1][2]
    amplitude = servo.amplitude_efetiva(a.amplitude)

    print("=" * 74)
    if a.ensaio:
        print("  [ENSAIO] placa sintética — o EEG é falso, nada vai para dados/gravacoes/.")
    print(f"  {a.sujeito} {a.run} | modelo {os.path.basename(a.modelo)} | {esquema.get('nome', '?')}")
    print(f"  decisão ..: fecha em '{a.classe}' acima de {a.limiar:g} | passo {a.passo:g} s")
    print(f"  órtese ...: amplitude {amplitude}°, trava {cfg.SEGURA_S:g} s após fechar")
    print(f"  protocolo : {a.descanso_inicial:g} s de descanso + {a.tentativas} tentativas "
          f"= {total/60:.1f} min")
    print(f"  saída ....: {pasta}")
    print("=" * 74)

    janela = None
    if not a.sem_tela:
        try:
            import cv2
            janela = "Motor Imagery"
            cv2.namedWindow(janela, cv2.WINDOW_NORMAL)
            cv2.setWindowProperty(janela, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        except ImportError:
            print("[aviso] opencv-python não instalado; dicas só no terminal.")

    dec = servo.Decisor(classe=a.classe, limiar=a.limiar)
    ortese, acionador = None, None
    if not a.sem_ortese:
        ortese = servo.Ortese(simular=a.simular_ortese, amplitude=a.amplitude,
                                  intervalo=a.intervalo)
        if not ortese.simular and not ortese.ping():
            print("  [aviso] o firmware não respondeu ao ping. Seguindo mesmo assim.")
        acionador = Acionador(ortese)
        acionador.start()

    # Primeira passada da cadeia (2 a 11 s) feita antes do stream, como no tempo_real.py.
    with torch.no_grad():
        modelo(torch.from_numpy(preproc.normalizar_com_modelo(
            modelo, preproc.janela_bruta(np.zeros((len(linhas), largura)), taxa, amostras))).float())

    os.makedirs(pasta)
    arq = open(os.path.join(pasta, "inferencias.csv"), "w", newline="", encoding="utf-8")
    log = csv.writer(arq)
    log.writerow(["t_s", "dica"] + [f"p_{cfg.CLASSES[i]}" for i in sorted(cfg.CLASSES)]
                 + ["rotulo", "acao"])

    blocos, eventos, fechamentos = [], [], []
    cauda = np.zeros((len(linhas), 0))
    dica, t, t_zero = "", 0.0, None
    try:
        placa.prepare_session()
        placa.start_stream()
        mostrar(janela, *DICAS["RE"])
        print(f"Estabilizando {cfg.PAUSA_ESTABILIZACAO}s... (ESC ou Ctrl-C encerra)")
        time.sleep(cfg.PAUSA_ESTABILIZACAO)
        placa.get_board_data()                      # único descarte da sessão

        t_zero = time.time()
        ultima_inf = 0.0
        while True:
            time.sleep(0.02)
            agora = time.time()
            t = agora - t_zero
            nova = dica_em(agenda, t)
            if nova is None:
                break
            if nova != dica:
                eventos.append({"trial_index": len(eventos) + 1, "code": nova, "t_onset": agora})
                print(f"  {t:7.1f}s  >>> {DICAS[nova][0]}")
                dica = nova
            if janela is not None:
                mostrar(janela, *DICAS[nova])

            pedaco = placa.get_board_data()
            if pedaco.shape[1]:
                blocos.append(pedaco)
                cauda = np.concatenate([cauda, pedaco[linhas]], axis=1)[:, -largura:]

            if cauda.shape[1] < largura or agora - ultima_inf < a.passo:
                continue
            ultima_inf = agora

            x = preproc.janela_bruta(cauda * 1e-6, taxa, amostras)
            with torch.no_grad():
                prob = torch.softmax(modelo(torch.from_numpy(
                    preproc.normalizar_com_modelo(modelo, x)).float()), dim=1).numpy()[0]

            rot, acao = dec.passo(prob, agora)
            if acao == "FECHAR":
                fechamentos.append(t)
                print(f"  {t:7.1f}s {dica:2s} | {rot:<5s} | p({a.classe})="
                      f"{prob[[k for k, v in cfg.CLASSES.items() if v == a.classe][0]]:.2f}"
                      f"   >>> FECHA")
                if acionador:
                    acionador.fila.put(ortese.fechar)
            elif acao == "ABRIR":
                print(f"  {t:7.1f}s {dica:2s} |   <<< abre")
                if acionador:
                    acionador.fila.put(ortese.abrir)
            log.writerow([f"{t:.3f}", dica] + [f"{p:.4f}" for p in prob] + [rot, acao or ""])
            arq.flush()

        resto = placa.get_board_data()
        if resto.shape[1]:
            blocos.append(resto)
    except KeyboardInterrupt:
        print("\nInterrompido — salvando o que já foi gravado.")
    finally:
        # A mão primeiro, como no tempo_real.py: se a placa falhar, a órtese já abriu.
        if acionador:
            acionador.parar()
        if ortese:
            ortese.fechar_conexao()
        for encerrar in (placa.stop_stream, placa.release_session):
            try:
                encerrar()
            except Exception:
                pass
        arq.close()
        if janela is not None:
            import cv2
            cv2.destroyAllWindows()
        print("Sessão encerrada, órtese aberta.")

    print("-" * 74)
    if not blocos or not eventos:
        sys.exit(f"Nenhuma amostra capturada. Só o log ficou em {pasta}.")
    salvar_gravacao(pasta, blocos, eventos, linha_tempo, canais_eeg, taxa)

    r = resumir(agenda, fechamentos, t, bloco_s=a.descanso_inicial)
    resumo = texto_resumo(r, a)
    print("=" * 74 + "\n" + resumo + "\n" + "=" * 74)
    with open(os.path.join(pasta, "resumo.txt"), "w", encoding="utf-8") as f:
        f.write(resumo + "\n")
    print(f"Salvo em {pasta}")
    if not a.ensaio:
        print(f"Para conferir:  python inferir_offline.py --gravacao {a.sujeito}_{a.run}")


if __name__ == "__main__":
    main()
