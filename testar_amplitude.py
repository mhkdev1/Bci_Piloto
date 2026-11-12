# -*- coding: utf-8 -*-
"""
testar_amplitude.py — confere, na órtese de verdade, que `--amplitude` fecha o quanto pede.

    python testar_amplitude.py --porta COM11                    # 9°, 18°, 27° e o teto
    python testar_amplitude.py --porta COM11 --amplitudes 12 21 # escolhe os valores
    python testar_amplitude.py --porta COM11 --pausa 3          # mais tempo para olhar

Para cada amplitude, conecta como o tempo_real.py faz (`Ortese(amplitude=...)`), fecha,
pergunta ao FIRMWARE onde o servo está, pede 60° além do fechado para conferir que não
passa, e abre. A posição vem do eco `angle=<n>` do ESP32, não da conta do PC: mandar só o
dígito do passo não move o servo, mas o firmware responde com o ângulo atual.

Mantenha a órtese VAZIA. Ctrl-C devolve ao repouso.
"""
from __future__ import annotations

import argparse
import time

import config as cfg
from nucleo import servo


def angulo_firmware(o):
    """Pergunta ao ESP32 onde o servo está, sem movê-lo."""
    o._limpar_entrada()
    o._enviar(str(o.passo).encode())
    return o._ler_eco()


def main():
    ap = argparse.ArgumentParser(description="Testa a amplitude da órtese no hardware.")
    ap.add_argument("--porta", default=None, help="padrão: config.PORTA_SERVO ou o Bluetooth")
    ap.add_argument("--amplitudes", type=servo.arg_amplitude, nargs="+", default=None,
                    help="em graus; padrão: 9, 18, 27 e config.SERVO_AMPLITUDE")
    ap.add_argument("--intervalo", type=servo.arg_intervalo, default=None,
                    help=servo.AJUDA_INTERVALO)
    ap.add_argument("--pausa", type=float, default=2.0, help="segundos fechada e aberta")
    a = ap.parse_args()
    amplitudes = a.amplitudes or sorted({x for x in (9, 18, 27) if x < cfg.SERVO_AMPLITUDE}
                                     | {cfg.SERVO_AMPLITUDE})

    print("=" * 74)
    print("  TESTE DE AMPLITUDE — órtese VAZIA")
    print(f"  repouso {cfg.SERVO_REPOUSO}° | fecha com '{cfg.SERVO_SENTIDO_FECHA}' | passo "
          f"{cfg.SERVO_PASSO_GRAUS}° | teto config.SERVO_AMPLITUDE = {cfg.SERVO_AMPLITUDE}°")
    print(f"  amplitudes: {', '.join(f'{x}°' for x in amplitudes)}")
    print("=" * 74)

    falhas = []

    def checar(nome, obtido, esperado):
        ok = obtido == esperado
        print(f"    {'ok   ' if ok else 'FALHA'} {nome}: firmware em {obtido}°"
              + ("" if ok else f" (esperado {esperado}°)"))
        if not ok:
            falhas.append(f"{nome}: {obtido}° em vez de {esperado}°")

    for amp in amplitudes:
        print(f"\n--- amplitude {amp}° " + "-" * 56)
        o = servo.Ortese(porta=a.porta, amplitude=amp, intervalo=a.intervalo)
        try:
            alvo = o.fechado_graus
            checar("partiu do repouso", angulo_firmware(o), cfg.SERVO_REPOUSO)

            t = time.monotonic()
            o.fechar()
            print(f"    fechou em {time.monotonic() - t:.2f} s")
            checar(f"fechou {o.amplitude}°", angulo_firmware(o), alvo)
            time.sleep(a.pausa)

            alem = alvo + (60 if alvo > cfg.SERVO_REPOUSO else -60)
            o.ir_para(alem)
            checar(f"pedir {alem}° não passa do fechado", angulo_firmware(o), alvo)

            o.abrir()
            checar("abriu", angulo_firmware(o), cfg.SERVO_REPOUSO)
            time.sleep(a.pausa)
        except KeyboardInterrupt:
            falhas.append("interrompido")
            break
        finally:
            o.fechar_conexao()

    print("\n" + "=" * 74)
    if falhas:
        print(f"  {len(falhas)} falha(s):")
        for f in falhas:
            print(f"    - {f}")
    else:
        print("  Tudo certo: a órtese fechou exatamente a amplitude pedida em todos os casos.")
    print("=" * 74)
    return 1 if falhas else 0


if __name__ == "__main__":
    raise SystemExit(main())
