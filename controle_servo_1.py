# -*- coding: utf-8 -*-
"""
controle_servo_1.py — controle manual da órtese pelo teclado.

    python controle_servo_1.py

Não tem argumentos. A porta vem de config.PORTA_SERVO; vazio, o ESP32 é achado pelo nome
Bluetooth (config.SERVO_BT_NOME), ou o programa lista as portas e pergunta.

    seta ESQUERDA   manda 'L' enquanto pressionada: o ângulo diminui um passo por vez
    seta DIREITA    manda 'R' enquanto pressionada: o ângulo aumenta
    seta CIMA       manda 'C': volta ao repouso
    0 a 9           tamanho do passo, em graus (0 vira 1)
    ESC             sai

Este script fala direto com o ESP32, sem passar pelo `nucleo.servo.Ortese`. Então ele NÃO
manda repouso nem batente do config.py: valem os que estão salvos no ESP32 (firmware
livre) ou os fixos do .ino (firmware original). Para ajustar esses valores à mão, use o
console do firmware livre:

    python -m serial.tools.miniterm COM4 115200
    ?  A90  P15  H140  N20  X140  U500,2500      (cada um seguido de Enter; Ctrl+] sai)
"""
import time
import serial
from pynput import keyboard

import config as cfg
from nucleo import servo

# A porta vem de config.PORTA_SERVO. Vazio = acha sozinho o ESP32 pareado por Bluetooth
# (pelo nome em config.SERVO_BT_NOME), ou pergunta. Para forçar uma porta: 'COM11'.
SERIAL_PORT = cfg.PORTA_SERVO or servo.achar_porta()
BAUD = cfg.BAUD_SERVO

print(f"Conectando em {SERIAL_PORT}...")
try:
    ser = serial.Serial(SERIAL_PORT, BAUD, timeout=0)
except serial.SerialException as e:
    raise SystemExit(f"Não consegui abrir {SERIAL_PORT}: {e}")
pressed = set()

print("Controles:")
print("← = L (movimenta mão esquerda)")
print("→ = R (movimenta mão direita)")
print("↑ = C (descansa mão)")
print("0..9 = define o passo (0=1°, 9=9°)")
print("ESC para sair")

# envia comandos contínuos a cada 50 ms enquanto teclas estiverem pressionadas
def pump():
    ultimo, resto = None, b""
    while listener.is_alive():
        if keyboard.Key.left in pressed:
            ser.write(b'L')
        if keyboard.Key.right in pressed:
            ser.write(b'R')
        # o firmware responde "angle=<n>" a cada byte; o eco tem que ser consumido,
        # senão se acumula no buffer durante a sessão inteira
        resto += ser.read_all() or b""
        eco = servo._RE_ANGULO.findall(resto)
        resto = resto[resto.rfind(b"\n") + 1:]      # guarda só a linha incompleta
        if eco and eco[-1] != ultimo:
            ultimo = eco[-1]
            print(f"\rângulo: {int(ultimo):4d}°", end="", flush=True)
        time.sleep(0.05)  # 50 ms por passo (ajuste se quiser mais rapido/lento)

def on_press(key):
    if key == keyboard.Key.esc:
        return False  # para o listener; o pump sai do laço e a porta fecha no finally

    # setas “one-shot”
    if key == keyboard.Key.up:
        ser.write(b'C')

    # contínuas
    if key in (keyboard.Key.left, keyboard.Key.right):
        pressed.add(key)

    # números mudam o passo no ESP32
    try:
        if hasattr(key, 'char') and key.char is not None and key.char.isdigit():
            ser.write(key.char.encode('ascii'))
    except:
        pass

def on_release(key):
    if key in (keyboard.Key.left, keyboard.Key.right):
        pressed.discard(key)

if __name__ == "__main__":
    # Thread do leitor de teclado
    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()

    try:
        pump()
    except KeyboardInterrupt:
        pass
    except serial.SerialException as e:
        print(f"\nA conexão com o ESP32 caiu: {e}")
    finally:
        if ser and ser.is_open:
            ser.close()
        print()
