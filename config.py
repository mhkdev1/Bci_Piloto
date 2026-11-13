# -*- coding: utf-8 -*-
"""
config.py — todos os parâmetros do projeto, num arquivo só.

Nenhum script tem número solto: se você quer mudar alguma coisa, muda aqui. Os valores
abaixo são os que produziram o modelo em `modelos/cnn2d_transicao.pt`, e vários deles
foram escolhidos por medição — quando foi o caso, o número medido está no comentário.
"""
import os

RAIZ = os.path.dirname(os.path.abspath(__file__))

# =====================================================================
# 1) HARDWARE
# =====================================================================
PORTA_EEG = "COM5"          # dongle da Cyton. Linux: "/dev/ttyUSB0"
# ESP32 da órtese, pelo Bluetooth ou pelo cabo — o protocolo é o mesmo. Vazio: acha
# sozinho o ESP32 pareado com o nome SERVO_BT_NOME; se não achar, lista e pergunta.
# Para forçar uma porta (ex.: o cabo USB), escreva-a aqui: "COM11".
PORTA_SERVO = ""
SERVO_BT_NOME = "BCI_Ortese" # nome Bluetooth do ESP32; tem que ser o mesmo do .ino
BAUD_SERVO = 115200         # só conta no cabo; a porta Bluetooth ignora
SERVO_TIMEOUT_ECO_S = 1.5   # prazo para o eco `angle=<n>` chegar. No cabo chega em ms; no
                            # Bluetooth, com o link ocioso, pode passar de 0,5 s. A leitura
                            # termina assim que o eco chega, então não atrasa o uso normal.
TAXA_PLACA = 250            # Cyton no dongle USB é 250 Hz fixo

# Quais eletrodos, e em que ordem. A ordem importa: é a mesma que a rede viu no treino.
CANAIS = ["C3", "C4", "Cz"]
# Índice de cada canal na lista de canais EEG que o BrainFlow devolve para a Cyton.
MAPA_CANAIS = {"C3": 0, "C4": 1, "Cz": 2}

# =====================================================================
# 2) PROTOCOLO
# =====================================================================
# Durações medidas nos EDFs do eegmmidb (1800 segmentos), não tiradas do artigo —
# a página do PhysioNet não publica os tempos por 
# trial.
DUR_DESCANSO = 4.1          # anotação T0
DUR_TAREFA = 4.1            # anotações T1 (mão esquerda) e T2 (mão direita)
N_TAREFAS = 3              # trials de tarefa por sessão de captura
CAPTURA_MODO = "ambas"     # capturar.py: "ambas", "esquerda", "direita" ou "descanso".
                           # Sempre alterna com descanso; "descanso" grava só descanso.
CAPTURA_TIPO = "imaginar"  # "imaginar" (imagética) ou "executar" (movimento real). Os
                           # rótulos são os mesmos; o tipo fica na coluna `tipo` do eventos.csv.
PAUSA_ESTABILIZACAO = 2     # segundos de descarte no início, antes do primeiro trial

# =====================================================================
# 3) PRÉ-PROCESSAMENTO
# =====================================================================
TAXA_IA = 128               # o EDF vem a 160 Hz e a Cyton a 250; ambos caem para 128
NOTCH = 60.0                # rede elétrica
PASSA_BANDA = (0.5, 35.0)
# O corte inferior foi varrido nesta arquitetura (platô das últimas 10 épocas):
#   0,5 Hz -> 72,0%  |  1 Hz -> 69,1%  |  2 Hz -> 68,5%  |  4 Hz -> 61,0%
# Quase toda a perda acontece já entre 0,5 e 1 Hz. O corte superior nunca mudou nada.

DETREND = "linear"          # remove offset e inclinação. Obrigatório, em todos os pontos.
PAD_FILTRO_S = 4.0          # espelho temporário nas bordas, só para o FIR de 0,5 Hz não
                            # distorcer a janela curta. É recortado antes do modelo ver.

# =====================================================================
# 4) JANELA — o coração do método
# =====================================================================
# Cada janela é centrada numa TRANSIÇÃO e leva metade de cada segmento vizinho:
#
#   centro em descanso -> tarefa   = MOVIMENTO
#   centro em tarefa -> descanso   = DESCANSO
#
#   descanso  |##############|
#   movimento                |##############|
#   ciclo     TTTTTTTTTTTTTTT...............TTTTTTTTTTTTTT
#                            ^ início da tarefa
#
# Como os segmentos duram o mesmo, as duas janelas ladrilham a linha do tempo: encostam
# uma na outra, sem sobreposição e sem buraco. Nenhum trecho de sinal recebe dois rótulos.
MEIA_JANELA = DUR_TAREFA / 2.0            # 2,05 s de cada lado
JANELA_S = DUR_TAREFA                     # 4,1 s no total
AMOSTRAS = int(round(JANELA_S * TAXA_IA)) + 1     # 526, contando as duas pontas

CLASSES = {0: "rest", 1: "left", 2: "right"}
CODIGO_CLASSE = {"RE": 0, "LH": 1, "RH": 2, "BEO": 0}
# T1 = mão esquerda, T2 = mão direita (runs 3,4,7,8,11,12 do eegmmidb)
ANOTACAO_CLASSE = {"T0": 0, "T1": 1, "T2": 2}

# =====================================================================
# 5) TREINO
# =====================================================================
RUNS_EDF = [3, 4, 7, 8, 11, 12]   # runs de mão: 3,7,11 movimento real; 4,8,12 imagética
SUJEITOS_RUINS = [88, 89, 92, 100]  # gravados a 128 Hz ou com run corrompido
EPOCAS = 35
LOTE = 16
LOTE_VALIDACAO = 64               # só afeta o pico de RAM na validação
TAXA_APRENDIZADO = 1e-3
FRACAO_VALIDACAO = 0.2
TREINO_FINAL_COM_TUDO = True      # após a validação escolher as épocas, treina de novo com
                                  # treino + validação e salva ESSE modelo. --no-final-com-tudo
                                  # salva o modelo da melhor época, sem rever a validação.
PESO_POR_CLASSE = True            # compensa o desbalanceamento (rest é ~50% dos dados)
SEMENTE = 40

AUGMENT = True
AUG_PROB = 0.5                    # probabilidade de aplicar CADA transformação
AUG_RUIDO = 0.10                  # desvio do ruído gaussiano, em unidades de z-score
AUG_ESCALA = (0.9, 1.1)           # escala de amplitude por janela
AUG_DESLOCAMENTO = 16             # deslocamento temporal máximo, em amostras

# =====================================================================
# 6) ÓRTESE
# =====================================================================
CLASSE_ACIONA = "right"           # qual classe fecha a órtese

# ---- Servo: os valores abaixo espelham o firmware gravado no ESP32 ----
# O firmware é INCREMENTAL e trabalha na escala do próprio servo, com o repouso no
# CENTRO do curso. Ver firmware_esp32/firmware_esp32.ino:
#
#   'L'        angle = max(-30, angle - passo)
#   'R'        angle = min(180, angle + passo)
#   'C'        angle = 90            <- repouso é 90°, não 0°
#   '0'..'9'   passo = dígito, com '0' virando 1  (então 1 a 9 graus)
#   e imprime "angle=<n>" a cada byte recebido
#
# Com o firmware_esp32_livre gravado (SERVO_FIRMWARE_LIVRE = True), o PC manda repouso e
# batente ao ESP32 a cada conexão: mude só aqui, sem regravar. Com o firmware antigo
# (False), mudar REPOUSO exige mudar o .ino junto.
SERVO_FIRMWARE_LIVRE = True       # False se o ESP32 estiver com o firmware_esp32 original
SERVO_REPOUSO = 140               # para onde o 'C' leva o servo
SERVO_AMPLITUDE = 120             # TETO do curso, em graus a partir do repouso. --amplitude só diminui
SERVO_PASSO_GRAUS = 10             # 1 a 9. 1° é o menor passo que o protocolo do firmware aceita
SERVO_INTERVALO_S = 0.001          # espera entre pulsos; a velocidade é passo / intervalo.
                                  # Na linha de comando: --intervalo 0.02 (não muda o firmware)
SERVO_SENTIDO_FECHA = "L"         # 'R' ou 'L' — descubra qual fecha a SUA montagem

# Ângulos absolutos, na escala do firmware. Derivados, não edite direto.
ANGULO_ABERTO = SERVO_REPOUSO
ANGULO_FECHADO = (SERVO_REPOUSO + SERVO_AMPLITUDE if SERVO_SENTIDO_FECHA == "R"
                  else SERVO_REPOUSO - SERVO_AMPLITUDE)
# Os 39° são 30% a mais que os 30° usados até aqui, que eram o limite confortável do dedo
# indicador na montagem antiga. Confira o batente físico com a órtese VAZIA antes de
# vestir: acima do limite mecânico a haste força a articulação. Para 40%, use 42 (14
# pulsos); o passo de 3° só permite múltiplos de 3.
LIMIAR = 0.80                     # probabilidade mínima para acionar
# Medido em 6 runs contínuos do PhysioNet (44 cues, 740 s), passo de 0,25 s:
#   limiar 0,60 -> 73% de detecção, 1 disparo falso a cada 16 s
#   limiar 0,70 -> 68%,             1 a cada 29 s
#   limiar 0,80 -> 43%,             1 a cada 53 s   <- o mais seguro dos medidos
# Numa órtese, disparo falso é a mão fechando sozinha; é o erro que machuca. A tabela foi
# medida com passo de 0,25 s; com um PASSO_S menor a rede é consultada mais vezes.

PASSO_S = 0.1                    # intervalo entre inferências no tempo real
SEGURA_S = 2.0                    # tempo travado depois de fechar
REFRATARIO_S = 2.0                # tempo morto depois de soltar, antes de poder fechar

# =====================================================================
# 7) CAMINHOS
# =====================================================================
DIR_PHYSIONET = os.path.join(RAIZ, "dados", "physionet")
DIR_GRAVACOES = os.path.join(RAIZ, "dados", "gravacoes")
DIR_MODELOS = os.path.join(RAIZ, "modelos")
DIR_FIGURAS = os.path.join(RAIZ, "figuras")
DIR_LOGS = os.path.join(RAIZ, "logs")           # logs do tempo_real.py, um por sessão
MODELO_PADRAO =os.path.join(DIR_MODELOS, "cnn2d_transicao.pt")

# =====================================================================
# 8) ARQUITETURA DA REDE
# =====================================================================
CNN_KERNELS = (5, 11, 15, 33, 65)   # escalas temporais; a 128 Hz cobrem 25,6 a 2,0 Hz
CNN_FILTROS = 20                    # filtros por escala
CNN_DEPTH = 3                       # filtros espaciais por mapa. Com 3 eletrodos, 3 é o
                                    # teto útil: não há mais direções independentes.
CNN_SAIDA = 32                      # mapas depois do pointwise
