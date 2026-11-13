# -*- coding: utf-8 -*-
"""
tempo_real.py — capacete ligado, órtese acionada pela intenção de movimento.

------------------------------------------------------------------------------
EXEMPLOS — um por opção
------------------------------------------------------------------------------
    python tempo_real.py                             # capacete + órtese, padrões do config
    python tempo_real.py --sem-ortese                # só imprime; não abre a órtese
    python tempo_real.py --simular-ortese            # imprime o que mandaria à órtese
    python tempo_real.py --ensaio --sem-ortese       # EEG falso: sem capacete e sem órtese
    python tempo_real.py --modelo modelos/cnn2d_SK.pt   # outro modelo (padrão: config.MODELO_PADRAO)
    python tempo_real.py --porta COM5                # porta da CYTON (padrão: config.PORTA_EEG)
    python tempo_real.py --sujeito SK                # nome do log em logs/
    python tempo_real.py --limiar 0.7                # probabilidade mínima para fechar (config.LIMIAR)
    python tempo_real.py --classe left               # qual classe fecha (config.CLASSE_ACIONA)
    python tempo_real.py --passo 0.25                # segundos entre inferências (config.PASSO_S)
    python tempo_real.py --amplitude 60              # fecha só 60° (teto: config.SERVO_AMPLITUDE)
    python tempo_real.py --intervalo 0.02            # espera entre pulsos do servo: mais lento
    python tempo_real.py --protocolo --sujeito SK    # sessão guiada, com dicas e resultado
    python tempo_real.py --protocolo --tentativas 10 # 10 tentativas em vez de 20
    python tempo_real.py --protocolo --descanso-inicial 0   # sem o bloco de descanso inicial
    python tempo_real.py --protocolo --sem-tela      # dicas só no terminal

    # primeiro teste de um modelo novo: sem órtese; depois com ela
    python tempo_real.py --modelo modelos/cnn2d_SK.pt --sujeito SK --sem-ortese
    python tempo_real.py --modelo modelos/cnn2d_SK.pt --sujeito SK

A porta da ÓRTESE não é argumento: vem de config.PORTA_SERVO; vazio, o ESP32 é achado
pelo nome Bluetooth, ou o programa lista as portas e pergunta. O --porta é o da Cyton.

O laço é simples: a cada `config.PASSO_S` segundos, pega os últimos 4,1 s do stream,
passa pela mesma cadeia do treino, classifica, e entrega a probabilidade ao `Decisor`,
que decide se manda fechar. Toda a lógica de acionamento está em `nucleo/servo.py`.

Toda sessão grava `logs/<sujeito>_<data>_<hora>.csv`, uma linha por inferência: tempo,
dica na tela, probabilidades, rótulo e ação.

------------------------------------------------------------------------------
O MODO --protocolo
------------------------------------------------------------------------------
Mostra as dicas na tela (as mesmas do capturar.py) e mede o sistema sozinho:

    bloco de descanso (--descanso-inicial, 120 s)   -> só conta disparos falsos
    --tentativas vezes (20): DESCANSE 4,1 s | IMAGINE A MÃO 4,1 s

Uma tentativa é DETECTADA se sair um FECHA entre o início da dica de mão e
`DUR_TAREFA + MEIA_JANELA` depois dele: a rede só vê a transição no centro da janela uns
2 s depois do início, e o comando pode chegar já no descanso seguinte. Qualquer FECHA
fora dessas faixas é DISPARO FALSO. O resumo sai na tela e em `<log>_resumo.txt`.

Os dois erros não são independentes: um disparo falso no fim do descanso trava o Decisor
por SEGURA_S + REFRATARIO_S e pode esconder a detecção da tentativa seguinte.

------------------------------------------------------------------------------
O QUE ESPERAR, E O QUE NÃO ESPERAR
------------------------------------------------------------------------------
Este é o ponto do projeto onde a diferença entre o número de validação e o uso real
aparece inteira, e vale ser explícito sobre isso antes de alguém colocar a mão na órtese.

A rede foi treinada com janelas centradas na transição descanso -> tarefa. Ao vivo, a
janela deslizante só reproduz essa geometria quando a transição cai no centro dela — ou
seja, cerca de 2 segundos depois de o usuário começar a imaginar o movimento. Antes
disso a janela é quase toda descanso; depois, o transiente sai pela borda e o que sobra
é imagética sustentada, que separa bem pior. A detecção tem, portanto, uma janela útil
de mais ou menos um segundo, e é por isso que o `Decisor` TRAVA o estado depois de
fechar em vez de exigir confirmação contínua.

Medido em 6 runs contínuos do PhysioNet (44 tentativas de mão direita, 740 s) com
limiar 0,80: cerca de 43% das tentativas geram um comando, com um disparo
fora de hora a cada 53 segundos. Ou seja: mais da metade das tentativas não fecha a mão,
e a mão fecha sozinha aproximadamente uma vez por minuto. Baixar o limiar aumenta a
detecção e piora os disparos falsos na mesma medida.

Um sistema com esse desempenho serve para demonstração e pesquisa. Não serve para uso
assistivo sem supervisão, e a órtese deve estar sempre acessível para ser removida.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time

import numpy as np
import torch

import config as cfg
from nucleo import preproc, servo
from nucleo.modelo import carregar

CODIGO_TAREFA = {"left": "LH", "right": "RH"}


def agenda_protocolo(classe, tentativas, descanso_inicial):
    """[(início_s, código, duração_s)]: bloco de descanso, depois descanso/tarefa alternados."""
    agenda, t = [], 0.0
    if descanso_inicial > 0:
        agenda.append((0.0, "RE", float(descanso_inicial)))
        t = float(descanso_inicial)
    for _ in range(tentativas):
        agenda.append((t, "RE", cfg.DUR_DESCANSO))
        t += cfg.DUR_DESCANSO
        agenda.append((t, CODIGO_TAREFA[classe], cfg.DUR_TAREFA))
        t += cfg.DUR_TAREFA
    return agenda


def dica_em(agenda, t):
    """Código da dica no instante t, ou None depois do fim."""
    for ini, cod, dur in agenda:
        if ini <= t < ini + dur:
            return cod
    return None


def resumir(agenda, fechamentos, duracao, bloco_s=0.0, tolerancia=cfg.MEIA_JANELA):
    """Detecção, latência e disparos falsos de uma sessão guiada.

    `fechamentos`: instantes (s, na base da agenda) em que saiu FECHAR. `duracao`: até onde
    a sessão foi — se interrompida, a detecção só conta as tentativas completas.
    """
    faixas = [(ini, ini + dur + tolerancia, ini + dur)
              for ini, cod, dur in agenda if cod != "RE" and ini < duracao]
    latencias = []
    for a, b, fim in faixas:
        if fim > duracao:
            continue
        dentro = [t for t in fechamentos if a <= t < b]
        if dentro:
            latencias.append(dentro[0] - a)
    falsos = [t for t in fechamentos if not any(a <= t < b for a, b, _ in faixas)]
    return {
        "tentativas": sum(1 for _, _, fim in faixas if fim <= duracao),
        "detectadas": len(latencias),
        "latencias": latencias,
        "falsos": len(falsos),
        "tempo_fora_s": duracao - sum(max(0.0, min(b, duracao) - a) for a, b, _ in faixas),
        "bloco_s": min(bloco_s, duracao),
        "falsos_bloco": sum(1 for t in falsos if t < bloco_s),
    }


def texto_resumo(r, a):
    n, d = r["tentativas"], r["detectadas"]
    lin = [
        "RESULTADO DA SESSÃO GUIADA",
        f"  sujeito ..........: {a.sujeito} | classe '{a.classe}' | limiar {a.limiar:g} "
        f"| passo {a.passo:g} s | amplitude {servo.amplitude_efetiva(a.amplitude)}°",
        f"  tentativas .......: {n} completas",
        f"  detectadas .......: {d}" + (f" ({100 * d / n:.0f}%)" if n else ""),
    ]
    if r["latencias"]:
        lat = np.array(r["latencias"])
        lin.append(f"  latência .........: mediana {np.median(lat):.1f} s, de {lat.min():.1f} "
                   f"a {lat.max():.1f} s (do início da dica ao FECHA)")
    minutos = r["tempo_fora_s"] / 60
    taxa = f" = {r['falsos'] / minutos:.2f} por minuto" if minutos > 0 else ""
    lin.append(f"  disparos falsos ..: {r['falsos']} em {minutos:.1f} min fora das "
               f"tentativas{taxa}")
    if r["bloco_s"] > 0:
        lin.append(f"    só no bloco de descanso: {r['falsos_bloco']} em "
                   f"{r['bloco_s'] / 60:.1f} min")
    return "\n".join(lin)


def main():
    ap = argparse.ArgumentParser(description="Inferência ao vivo com acionamento da órtese.")
    ap.add_argument("--modelo", default=cfg.MODELO_PADRAO)
    ap.add_argument("--porta", default=None, help="porta da Cyton (padrão: config.PORTA_EEG)")
    ap.add_argument("--sem-ortese", action="store_true", help="não aciona nada, só imprime")
    ap.add_argument("--simular-ortese", action="store_true", help="imprime os comandos, sem serial")
    ap.add_argument("--passo", type=float, default=cfg.PASSO_S)
    ap.add_argument("--limiar", type=float, default=cfg.LIMIAR)
    ap.add_argument("--classe", default=cfg.CLASSE_ACIONA)
    ap.add_argument("--amplitude", type=servo.arg_amplitude, default=None,
                    help=f"quanto a órtese fecha, em graus (padrão e teto: config.SERVO_AMPLITUDE = "
                         f"{cfg.SERVO_AMPLITUDE})")
    ap.add_argument("--intervalo", type=servo.arg_intervalo, default=None,
                    help=servo.AJUDA_INTERVALO)
    ap.add_argument("--sujeito", default="teste", help="código do participante, nome do log")
    ap.add_argument("--protocolo", action="store_true", help="sessão guiada com dicas na tela")
    ap.add_argument("--tentativas", type=int, default=20, help="com --protocolo")
    ap.add_argument("--descanso-inicial", type=float, default=120.0,
                    help="com --protocolo: bloco de descanso inicial, em s (0 desliga)")
    ap.add_argument("--sem-tela", action="store_true", help="dicas só no terminal")
    ap.add_argument("--ensaio", action="store_true", help="placa sintética: EEG falso, sem capacete")
    a = ap.parse_args()

    from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds

    modelo, amostras, esquema = carregar(a.modelo)
    if amostras != cfg.AMOSTRAS:
        sys.exit(f"O modelo espera {amostras} amostras e o config monta {cfg.AMOSTRAS}.")
    if a.classe not in cfg.CLASSES.values():
        sys.exit(f"--classe deve ser uma de {sorted(cfg.CLASSES.values())}")
    if a.protocolo and a.classe not in CODIGO_TAREFA:
        sys.exit("--protocolo precisa de uma classe de mão: left ou right.")

    params = BrainFlowInputParams()
    params.serial_port = a.porta or cfg.PORTA_EEG
    placa_id = (BoardIds.SYNTHETIC_BOARD if a.ensaio else BoardIds.CYTON_BOARD).value
    placa = BoardShim(placa_id, params)
    canais_eeg = BoardShim.get_eeg_channels(placa_id)
    linhas = [canais_eeg[cfg.MAPA_CANAIS[c]] for c in cfg.CANAIS]
    taxa = float(cfg.TAXA_PLACA)                # a sintética também é 250 Hz
    # O buffer é dimensionado a partir do que o MODELO espera, não de um número fixo:
    # assim ele acompanha automaticamente qualquer mudança de janela ou de taxa.
    largura = int(np.ceil(amostras / cfg.TAXA_IA * taxa))

    agenda = agenda_protocolo(a.classe, a.tentativas, a.descanso_inicial) if a.protocolo else None
    os.makedirs(cfg.DIR_LOGS, exist_ok=True)
    base_log = os.path.join(cfg.DIR_LOGS, f"{a.sujeito}_{time.strftime('%Y%m%d_%H%M%S')}")

    print("=" * 74)
    if a.ensaio:
        print("  [ENSAIO] placa sintética — o EEG é falso, os números não significam nada.")
    print(f"  modelo ...: {os.path.basename(a.modelo)} | {esquema.get('nome', '?')}")
    print(f"  janela ...: {largura} amostras @ {taxa:g} Hz  ->  {amostras} @ {cfg.TAXA_IA} Hz "
          f"({amostras/cfg.TAXA_IA:.2f} s)")
    print(f"  decisão ..: fecha em '{a.classe}' acima de {a.limiar:g} | passo {a.passo:g} s")
    amplitude = servo.amplitude_efetiva(a.amplitude)
    fechado = cfg.SERVO_REPOUSO + (amplitude if cfg.SERVO_SENTIDO_FECHA == "R" else -amplitude)
    print(f"  órtese ...: curso {cfg.ANGULO_ABERTO}°->{fechado}° ({amplitude}°), "
          f"trava {cfg.SEGURA_S:g} s após fechar")
    if agenda:
        total = agenda[-1][0] + agenda[-1][2]
        print(f"  protocolo : {a.descanso_inicial:g} s de descanso + {a.tentativas} tentativas "
              f"= {total/60:.1f} min")
    print(f"  log ......: {base_log}.csv")
    print("=" * 74)

    janela = None
    if agenda:
        from capturar import DICAS, mostrar
        if not a.sem_tela:
            try:
                import cv2
                janela = "Motor Imagery"
                cv2.namedWindow(janela, cv2.WINDOW_NORMAL)
                cv2.setWindowProperty(janela, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            except ImportError:
                print("[aviso] opencv-python não instalado; dicas só no terminal.")

    dec = servo.Decisor(classe=a.classe, limiar=a.limiar)
    ortese = None
    if not a.sem_ortese:
        ortese = servo.Ortese(simular=a.simular_ortese, amplitude=a.amplitude,
                               intervalo=a.intervalo)
        if not ortese.simular and not ortese.ping():
            print("  [aviso] o firmware não respondeu ao ping. Seguindo mesmo assim.")

    # A primeira passada pela cadeia carrega bibliotecas e projeta os filtros: medido de 2 a
    # 11 s. Feita aqui, com sinal nulo, ela não atrasa a primeira dica nem abre um buraco
    # no log e no bloco de descanso.
    with torch.no_grad():
        modelo(torch.from_numpy(preproc.normalizar_com_modelo(
            modelo, preproc.janela_bruta(np.zeros((len(linhas), largura)), taxa, amostras))).float())

    arq = open(base_log + ".csv", "w", newline="", encoding="utf-8")
    log = csv.writer(arq)
    log.writerow(["t_s", "dica"] + [f"p_{cfg.CLASSES[i]}" for i in sorted(cfg.CLASSES)]
                 + ["rotulo", "acao"])
    fechamentos, t0, t, dica = [], None, 0.0, ""
    try:
        placa.prepare_session()
        placa.start_stream()
        print(f"Stream ligado. Estabilizando {cfg.PAUSA_ESTABILIZACAO}s... (Ctrl-C encerra)")
        if agenda:
            mostrar(janela, *DICAS["RE"])
        time.sleep(cfg.PAUSA_ESTABILIZACAO)

        while True:
            time.sleep(a.passo)
            bruto = placa.get_current_board_data(largura)
            if bruto.shape[1] < largura:
                if janela is not None:
                    mostrar(janela, *DICAS["RE"])   # mantém a janela respondendo
                continue                            # buffer ainda enchendo
            agora = time.monotonic()
            if t0 is None:                          # o relógio parte da 1ª janela cheia
                t0 = agora
            t = agora - t0
            if agenda:
                nova = dica_em(agenda, t)
                if nova is None:
                    break
                if janela is not None or nova != dica:   # no terminal, só na troca
                    mostrar(janela, *DICAS[nova])
                dica = nova

            sinal = np.stack([bruto[r, :] for r in linhas]) * 1e-6
            x = preproc.janela_bruta(sinal, taxa, amostras)
            xn = preproc.normalizar_com_modelo(modelo, x)
            with torch.no_grad():
                prob = torch.softmax(modelo(torch.from_numpy(xn).float()), dim=1).numpy()[0]

            rot, acao = dec.passo(prob, agora)
            probs = "  ".join(f"{cfg.CLASSES[i]}={prob[i]:.2f}" for i in sorted(cfg.CLASSES))
            marca = ""
            if acao == "FECHAR":
                fechamentos.append(t)
                marca = f"   >>> FECHA ({fechado}°)"
                if ortese:
                    ortese.fechar()
            elif acao == "ABRIR":
                marca = "   <<< abre"
                if ortese:
                    ortese.abrir()
            elif dec.travado_ate is not None:
                marca = "   [travado]"
            log.writerow([f"{t:.3f}", dica] + [f"{p:.4f}" for p in prob] + [rot, acao or ""])
            arq.flush()                             # se o programa cair, o log fica
            print(f"  {t:7.1f}s {dica:2s} | {rot:<5s} | {probs}{marca}")
    except KeyboardInterrupt:
        print("\nEncerrando.")
    finally:
        # A mão primeiro: se o capacete falhou (dongle solto, bateria), stop_stream também
        # falha, e nada depois dele rodaria — a órtese ficaria fechada no dedo.
        if ortese:
            ortese.fechar_conexao()      # abre a mão antes de soltar a serial
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

    if agenda and t0 is not None:
        r = resumir(agenda, fechamentos, t, bloco_s=a.descanso_inicial)
        resumo = texto_resumo(r, a)
        print("=" * 74 + "\n" + resumo + "\n" + "=" * 74)
        with open(base_log + "_resumo.txt", "w", encoding="utf-8") as f:
            f.write(resumo + "\n")
        print(f"Log: {base_log}.csv\nResumo: {base_log}_resumo.txt")


if __name__ == "__main__":
    main()
