# -*- coding: utf-8 -*-
"""
servo.py — comando da órtese, do lado do PC.

Este módulo é a única parte do projeto que fala com o hardware da órtese; tudo que
decide *quando* acionar está fora daqui, o que permite testar a lógica inteira sem nada
plugado (`Ortese(simular=True)`).

------------------------------------------------------------------------------
O PROTOCOLO É O DO FIRMWARE, NÃO O CONTRÁRIO
------------------------------------------------------------------------------
Há dois firmwares, com o mesmo protocolo de um byte. Ele é INCREMENTAL e trabalha na
escala do próprio servo:

    'L'        angle = max(MIN, angle - passo)
    'R'        angle = min(MAX, angle + passo)
    'C'        angle = REPOUSO
    '0'..'9'   passo = dígito, com '0' virando 1  (portanto 1 a 9 graus)

Um byte por comando, sem terminador. Depois de cada byte recebido o firmware imprime
`angle=<n>`.

    firmware_esp32/        o original. REPOUSO = 90 fixo no .ino, MAX = 180. O MIN está
                           como -30 no .ino, mas a biblioteca do servo trava em 0°: abaixo
                           de zero o eco passa a mentir. config.SERVO_FIRMWARE_LIVRE = False.

    firmware_esp32_livre/  curso 0..180°. REPOUSO, MIN e MAX são ajustáveis pela serial
                           (H<n>, N<n>, X<n> + Enter) e ficam salvos na flash do ESP32.
                           Com config.SERVO_FIRMWARE_LIVRE = True, `_configurar_firmware()`
                           manda a cada conexão o repouso e o batente tirados do config.py
                           — não é preciso regravar o ESP32 para mudar ângulos.

    Exemplo: SERVO_REPOUSO = 140, SERVO_SENTIDO_FECHA = "L", SERVO_AMPLITUDE = 120
             -> o PC manda N20, X140, H140: a mão abre em 140° e fecha até 20°.

Três consequências que o código abaixo trata explicitamente:

1. O repouso não é 0°: é `config.SERVO_REPOUSO` (90° no firmware original). "Fechar 30°"
   com repouso em 90 quer dizer ir de 90 para 120 (ou para 60, se a sua montagem fechar
   no outro sentido). É por isso que `config.SERVO_AMPLITUDE` existe separado do ângulo
   absoluto.

2. O dígito É o passo, não passo-1. Mandar '3' dá passos de 3°. Como o máximo é '9', não
   existe passo de 10°.

3. O firmware informa o ângulo a cada comando. Isso é melhor do que contar às cegas, e a
   leitura é obrigatória de qualquer jeito: se ninguém consumir o eco, ele se acumula no
   buffer do host durante a sessão toda. Aqui o eco é lido e usado como posição real.

------------------------------------------------------------------------------
BLUETOOTH OU CABO — O MESMO CÓDIGO
------------------------------------------------------------------------------
O firmware aceita o protocolo pelo cabo USB e pelo Bluetooth (perfil serial, SPP). Depois
de pareado, o ESP32 vira uma porta COM no Windows e o resto deste módulo não enxerga a
diferença. Três cuidados existem por causa do Bluetooth:

1. A porta. O Windows cria DUAS portas COM por aparelho pareado, e só a de saída conecta.
   Fones e mouses pareados também criam portas. Com `config.PORTA_SERVO` vazio, a porta é
   achada pelo nome do aparelho (`config.SERVO_BT_NOME`), nunca por "a única porta
   Bluetooth" — ver `achar_porta()`.

2. A latência. O eco chega com atraso variável, e ler depois de um tempo fixo pegaria um
   ângulo velho: a órtese em 120° com o PC achando 114°, e o próximo comando mandaria
   pulsos a mais, passando da amplitude. `_ler_eco()` espera um eco por byte enviado.

3. A queda. Um link sem fio cai sem aviso, e o PC não consegue mandar o 'C' por um link
   que já caiu. Por isso o firmware devolve a órtese ao repouso sozinho quando percebe a
   queda — na hora, se o programa fecha a porta; em até uns 20 s, se o link some sem se
   desconectar (é o tempo de supervisão do Bluetooth). E `fechar()`/`abrir()` tentam
   reconectar antes de desistir.

------------------------------------------------------------------------------
POR QUE 30 GRAUS DE AMPLITUDE
------------------------------------------------------------------------------
O mecânico é o que manda: a haste da montagem usada força a articulação do dedo acima
disso, e o batente físico não perdoa erro de software.

O segundo motivo é metodológico. O que a rede detecta não é contração muscular: é a
dessincronização sensório-motora (ERD) sobre o córtex, medida em C3/C4/Cz. Esse sinal
indica *intenção* de mover, com latência de ~2 s, e não carrega amplitude — não há como
inferir dele "quanto" fechar. Um arco curto e fixo é o comando honesto para um detector
binário: sinaliza que a intenção foi reconhecida, sem fingir uma proporcionalidade que a
medida não tem.
"""
from __future__ import annotations

import re
import time

import config as cfg

# Só linhas completas: no Bluetooth o eco chega picado, e "angle=12" sem o fim da linha
# seria lido como 12 quando o firmware está mandando 120.
_RE_ANGULO = re.compile(rb"angle\s*=\s*(-?\d+)\r?\n")
_RE_BT_SAIDA = re.compile(r"BTHENUM\\.*&([0-9A-F]{12})_", re.I)


def portas_bluetooth():
    """{porta: nome do aparelho} das portas Bluetooth de SAÍDA.

    O Windows cria duas portas COM para cada aparelho serial pareado. A de saída traz o
    endereço do aparelho no hwid e é a que abre a conexão; a de entrada traz zeros e só
    serve para o aparelho chamar o PC, o que o ESP32 não faz. O pyserial não informa o
    nome do aparelho, que sai do registro do Windows.
    """
    import serial.tools.list_ports
    saida = {}
    for p in serial.tools.list_ports.comports():
        m = _RE_BT_SAIDA.search(p.hwid or "")
        if m and int(m[1], 16):
            saida[p.device] = _nome_bluetooth(m[1].lower())
    return saida


def _nome_bluetooth(mac):
    try:
        import winreg
        chave = "SYSTEM\\CurrentControlSet\\Services\\BTHPORT\\Parameters\\Devices\\" + mac
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, chave) as k:
            nome = winreg.QueryValueEx(k, "Name")[0]
    except (ImportError, OSError):
        return ""
    if not isinstance(nome, str):
        nome = bytes(nome).decode(errors="ignore")
    return nome.split("\0")[0]


def achar_porta():
    """Porta do ESP32 quando `config.PORTA_SERVO` está vazio.

    Procura a porta Bluetooth do aparelho chamado `config.SERVO_BT_NOME`. Pelo nome, e
    não por "a única porta Bluetooth": a porta de um fone pareado pode até conectar, e aí
    simplesmente nunca responde. Se não achar exatamente uma, lista as portas e pergunta.
    """
    import serial.tools.list_ports
    bt = portas_bluetooth()
    achadas = [p for p, nome in bt.items() if nome == cfg.SERVO_BT_NOME]
    if len(achadas) == 1:
        print(f"ESP32 '{cfg.SERVO_BT_NOME}' encontrado no Bluetooth, porta {achadas[0]}")
        return achadas[0]

    portas = list(serial.tools.list_ports.comports())
    if not portas:
        raise SystemExit(f"Nenhuma porta serial encontrada. Pareie o ESP32 "
                         f"'{cfg.SERVO_BT_NOME}' no Bluetooth do Windows, ou confira o cabo.")
    print(f"  {'Mais de uma' if achadas else 'Nenhuma'} porta Bluetooth com o nome "
          f"'{cfg.SERVO_BT_NOME}'. Portas disponíveis:")
    for i, p in enumerate(portas):
        if p.device in bt:
            extra = f"  <- Bluetooth: {bt[p.device] or '?'}"
        elif "BTHENUM" in (p.hwid or "").upper():
            extra = "  <- Bluetooth de entrada, não conecta"
        else:
            extra = ""
        print(f"  [{i}] {p.device} — {p.description}{extra}")
    # Sem "Enter = 0": a primeira porta costuma ser a de outro aparelho Bluetooth, que abre
    # sem erro e nunca responde — a sessão seguiria com a órtese parada.
    while True:
        escolha = input("Índice da porta: ").strip()
        if escolha.isdigit() and int(escolha) < len(portas):
            return portas[int(escolha)].device
        print(f"  digite um número de 0 a {len(portas) - 1}")


def amplitude_efetiva(graus=None, passo=None):
    """Quanto a órtese sai do repouso ao fechar, em graus, já em múltiplo do passo.

    `graus` vem da linha de comando (`--amplitude`) e só pode DIMINUIR o curso:
    `config.SERVO_AMPLITUDE` é o teto, o limite mecânico conferido na montagem. Para
    passar dele, mude o config.py de propósito, depois de conferir a órtese vazia.
    Nenhum dos dois exige regravar o ESP32 — mas o firmware_esp32_seguro tem o próprio
    batente (AMPLITUDE no .ino), e acima dele o servo para no batente.
    """
    passo = passo or cfg.SERVO_PASSO_GRAUS
    teto = int(cfg.SERVO_AMPLITUDE)
    graus = teto if graus is None else int(graus)
    if not passo <= graus <= teto:
        raise ValueError(f"amplitude deve ficar entre {passo}° (um passo) e {teto}° "
                         f"(config.SERVO_AMPLITUDE); pedido: {graus}°")
    return graus // passo * passo


def arg_intervalo(texto):
    """Tipo para argparse: `--intervalo 0.02`, a espera entre pulsos do servo, em segundos."""
    import argparse
    try:
        valor = float(texto)
    except ValueError:
        valor = -1.0
    if not 0 <= valor <= 1:
        raise argparse.ArgumentTypeError(f"intervalo deve ficar entre 0 e 1 segundo; pedido: {texto}")
    return valor


AJUDA_INTERVALO = (f"espera entre os pulsos do servo, em segundos (padrão: config.SERVO_INTERVALO_S "
                   f"= {cfg.SERVO_INTERVALO_S:g}). Maior = fecha mais devagar")


def arg_amplitude(texto):
    """Tipo para argparse: `--amplitude 20` com mensagem clara fora da faixa."""
    import argparse
    try:
        amplitude_efetiva(int(texto))
    except ValueError as e:
        raise argparse.ArgumentTypeError(str(e))
    return int(texto)


class Ortese:
    """Conexão com a órtese, pelo Bluetooth ou pelo cabo. Use como context manager para ela
    voltar ao repouso no fim.

        with Ortese() as o:
            o.fechar()
    """

    def __init__(self, porta=None, simular=False, passo_graus=None, sentido=None,
                 verboso=True, detalhe=False, amplitude=None, intervalo=None):
        self.simular = simular
        self.verboso = verboso
        self.detalhe = detalhe               # True imprime cada byte e cada eco
        self.passo = max(1, min(9, int(passo_graus or cfg.SERVO_PASSO_GRAUS)))
        self.sentido = (sentido or cfg.SERVO_SENTIDO_FECHA).upper()
        if self.sentido not in ("L", "R"):
            raise ValueError("sentido deve ser 'L' ou 'R'")
        self.oposto = "L" if self.sentido == "R" else "R"
        self.repouso_graus = int(cfg.SERVO_REPOUSO)
        self.amplitude = amplitude_efetiva(amplitude, self.passo)
        self.fechado_graus = (self.repouso_graus + self.amplitude if self.sentido == "R"
                              else self.repouso_graus - self.amplitude)
        if amplitude is not None and self.amplitude != amplitude:
            self._log(f"  amplitude pedida {amplitude}° vira {self.amplitude}°: o servo só anda "
                      f"em múltiplos do passo de {self.passo}°")
        self.angulo = self.repouso_graus
        self.ser = None
        # Espera entre pulsos. É ela que dá a velocidade: passo / intervalo, em graus por segundo.
        self.intervalo = float(cfg.SERVO_INTERVALO_S if intervalo is None else intervalo)

        if simular:
            self._log(f"órtese em SIMULAÇÃO — nada é enviado | repouso {self.repouso_graus}°, "
                      f"fechado {self.fechado_graus}°, passo {self.passo}°")
            self._configurar_passo()
            return

        self.porta = porta or cfg.PORTA_SERVO or achar_porta()
        self._conectar()
        self._log(f"órtese em {self.porta} | repouso {self.repouso_graus}°, "
                  f"fechado {self.fechado_graus}° (amplitude {self.amplitude}°), "
                  f"passo {self.passo}°, fecha com '{self.sentido}', "
                  f"intervalo {self.intervalo:g} s ({self._velocidade()})")

    # ---------------- interface ----------------
    def fechar(self):
        """Fecha a mão até a amplitude configurada."""
        return self._com_reconexao(self.ir_para, self.fechado_graus)

    def abrir(self):
        """Devolve a mão ao repouso."""
        return self._com_reconexao(self.repouso)

    def repouso(self):
        """Comando 'C'. É o único ponto em que a posição é conhecida sem depender da conta."""
        antes = self.angulo
        self._limpar_entrada()
        self._enviar(b"C")
        lido = self._ler_eco()
        self.angulo = lido if lido is not None else self.repouso_graus
        if antes != self.angulo:
            self._log(f"  órtese -> repouso ({self.angulo}°)")
        return self.angulo

    def ir_para(self, graus):
        """Move até o ângulo pedido, em passos, sem passar da amplitude permitida.

        Devolve o ângulo alcançado, que pode não ser exatamente o pedido: o servo só anda
        em múltiplos do passo, e a conta trunca para não ultrapassar o alvo. Com passo de
        3° e amplitude de 30°, os 10 pulsos chegam exatos.
        """
        alvo = self._seguro(graus)
        delta = alvo - self.angulo
        if abs(delta) < self.passo:
            return self.angulo
        letra = self.sentido if (delta > 0) == (self.sentido == "R") else self.oposto
        n = int(abs(delta) // self.passo)
        self._limpar_entrada()
        for _ in range(n):
            self._enviar(letra.encode())
            time.sleep(self.intervalo)
        lido = self._ler_eco(n)
        esperado = self.angulo + (n * self.passo if delta > 0 else -n * self.passo)
        self.angulo = lido if lido is not None else esperado
        if lido is not None and lido != esperado:
            self._log(f"  [aviso] o firmware reportou {lido}° e a conta dava {esperado}°. "
                      f"Seguindo pelo firmware.")
        self._log(f"  órtese -> {self.angulo}° ({n}x '{letra}', passo {self.passo}°)")
        return self.angulo

    def ping(self):
        """True se o firmware respondeu o eco. Manda 'C', que é inofensivo."""
        if self.simular:
            return True
        try:
            self.ser.reset_input_buffer()
            self._enviar(b"C")
            return self._ler_eco() is not None
        except Exception:
            return False

    # ---------------- interno ----------------
    def _conectar(self, tentativas=1):
        """Abre a porta e leva a órtese a uma posição conhecida. Vale para cabo e Bluetooth.

        `tentativas` > 1 serve à reconexão: se o ESP32 reiniciou, o Bluetooth dele leva
        alguns segundos para voltar, e uma tentativa só falharia cedo demais.
        """
        import serial
        for tentativa in range(1, tentativas + 1):
            self._log(f"conectando à órtese em {self.porta} "
                      f"(pelo Bluetooth leva alguns segundos)...")
            try:
                self.ser = serial.Serial(self.porta, cfg.BAUD_SERVO, timeout=0.2,
                                         write_timeout=2.0)
                try:                         # pulso DTR/RTS reinicia o ESP32 no cabo;
                    self.ser.setDTR(False); self.ser.setRTS(False); time.sleep(0.2)
                    self.ser.setDTR(True); self.ser.setRTS(True)   # no Bluetooth não faz nada
                except Exception:
                    pass
                time.sleep(1.8)              # espera o boot (cabo) ou o link assentar (Bluetooth)
                self.ser.reset_input_buffer(); self.ser.reset_output_buffer()
                self._configurar_firmware()
                self._configurar_passo()
                self.repouso()               # parte de uma posição conhecida
                return
            except OSError as e:             # serial.SerialException herda de OSError
                self._fechar_porta()
                if tentativa == tentativas:
                    raise SystemExit(
                        f"Não consegui conectar em {self.porta}: {e}\n"
                        f"  Bluetooth: o ESP32 está ligado, por perto e pareado como "
                        f"'{cfg.SERVO_BT_NOME}'?\n"
                        f"  Cabo: confira o cabo e o driver (CP210x/CH340).\n"
                        f"  Nos dois: feche o monitor serial da IDE do Arduino, se estiver "
                        f"aberto.") from e
                self._log(f"  [aviso] não conectou ({e}); tentando de novo em 2 s...")
                time.sleep(2.0)

    def _com_reconexao(self, comando, *args):
        """Executa o comando; se a conexão caiu, reconecta e segue a sessão.

        No Bluetooth a queda é um evento normal: distância, interferência, o ESP32
        reiniciou no pico de corrente do servo. O comando interrompido não é repetido: ao
        perder o cliente o firmware já devolveu a órtese ao repouso, e a reconexão parte do
        repouso. São três tentativas; se todas falharem, o erro sobe e quem chamou encerra
        pelo caminho normal.
        """
        try:
            return comando(*args)
        except OSError as e:                 # serial.SerialException herda de OSError
            if self.ser is None:
                raise
            self._log(f"  [aviso] a conexão com a órtese caiu ({e}). Reconectando...")
            self._fechar_porta()
            self._conectar(tentativas=3)
            self._log(f"  reconectado em {self.porta}, órtese no repouso ({self.angulo}°).")
            return self.angulo

    def _configurar_firmware(self):
        """Só no firmware_esp32_livre: manda repouso e limites do config.py para o ESP32.

        Os limites N/X viram batente no próprio firmware, entre o repouso e o fechado desta
        sessão. Primeiro abre 0..180, para N novo nunca ficar acima de X antigo.
        """
        if not getattr(cfg, "SERVO_FIRMWARE_LIVRE", False):
            return
        piso, teto = sorted((self.repouso_graus, self.fechado_graus))
        linhas = ["N0", "X180", f"N{piso}", f"X{teto}", f"H{self.repouso_graus}"]
        self._limpar_entrada()
        for l in linhas:
            self._enviar(f"{l}\n".encode())
        prazo = time.monotonic() + cfg.SERVO_TIMEOUT_ECO_S
        bruto = b""
        while bruto.count(b"repouso=") < len(linhas) and time.monotonic() < prazo:
            bruto += self.ser.read_all() or b""
            time.sleep(0.01)
        esperado = f"repouso={self.repouso_graus} min={piso} max={teto}".encode()
        if esperado not in bruto:
            raise SystemExit(f"O ESP32 não confirmou repouso/limites ({esperado.decode()}).\n"
                             f"  Gravou o firmware_esp32_livre? Se é o firmware antigo, ponha "
                             f"SERVO_FIRMWARE_LIVRE = False no config.py.")
        self._log(f"  firmware: repouso {self.repouso_graus}°, batente {piso}..{teto}°")

    def _configurar_passo(self):
        """Manda o dígito do passo. No firmware o dígito É o passo (só '0' vira 1)."""
        self._limpar_entrada()
        self._enviar(str(self.passo).encode())
        self._ler_eco()

    def _seguro(self, graus):
        """Recorta ao intervalo entre repouso e fechado. Último ponto antes do hardware."""
        alvo = int(round(float(graus)))
        piso, teto = sorted((self.repouso_graus, self.fechado_graus))
        return max(piso, min(teto, alvo))

    def _limpar_entrada(self):
        """Descarta eco atrasado de um comando anterior, para não ser lido como resposta deste."""
        if not self.simular:
            self.ser.reset_input_buffer()

    def _fechar_porta(self):
        try:
            if self.ser is not None:
                self.ser.close()
        except Exception:
            pass

    def _enviar(self, token):
        if self.detalhe:
            print(f"    {'[sim] ' if self.simular else ''}TX {token.decode()}")
        if self.simular:
            return
        self.ser.write(token)                # byte solto, sem terminador
        self.ser.flush()

    def _ler_eco(self, n=1):
        """Espera os `n` ecos `angle=<n>` (um por byte enviado) e devolve o último, ou None.

        Ler não é opcional: o firmware responde a cada byte, e se ninguém consumir, o
        buffer do host acumula durante a sessão inteira. Aproveitamos para usar o valor
        como posição real, em vez de confiar só na contagem local.

        A espera termina quando os `n` ecos chegam, ou em `config.SERVO_TIMEOUT_ECO_S`.
        Se faltar algum, devolve None: o último eco que chegou não é a posição final, e
        confiar nele faria o próximo comando passar da amplitude. Quem chama cai na conta.
        """
        if self.simular:
            return None
        # O prazo cresce com o número de pulsos: medido na órtese, o ESP32 leva uns 40 ms por
        # comando (o PWM de 50 Hz segura cada escrita no servo), então 80 pulsos pedem ~3,2 s.
        prazo = time.monotonic() + cfg.SERVO_TIMEOUT_ECO_S + 0.06 * n
        bruto = b""
        while True:
            bruto += self.ser.read_all() or b""
            achados = _RE_ANGULO.findall(bruto)
            if len(achados) >= n or time.monotonic() >= prazo:
                break
            time.sleep(0.01)
        if self.detalhe and bruto.strip():
            print(f"    RX {bruto.strip().decode(errors='ignore')}")
        if len(achados) < n:
            if achados:
                self._log(f"  [aviso] chegaram {len(achados)} de {n} ecos em "
                          f"{cfg.SERVO_TIMEOUT_ECO_S:g} s; seguindo pela conta do PC.")
            return None
        return int(achados[-1])

    def _velocidade(self):
        return f"{self.passo / self.intervalo:.0f}°/s" if self.intervalo > 0 else "sem espera"

    def _log(self, msg):
        if self.verboso:
            print(msg)

    def fechar_conexao(self):
        """Devolve a mão ao repouso e desliga a serial. Chamada ao sair do `with`.

        Voltar ao repouso antes de fechar a porta não é cortesia: se o processo terminar
        com a órtese fechada, ela fica fechada no dedo de alguém até o próximo comando.
        """
        if self.ser is None:
            return
        try:
            self.repouso()
            time.sleep(0.3)
        except Exception:
            pass                             # link morto: no Bluetooth o firmware já voltou
        self._fechar_porta()                 # fecha mesmo se o 'C' falhou

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.fechar_conexao()
        return False


class Decisor:
    """Transforma a decisão ruidosa de cada janela num comando estável para a órtese.

    Três mecanismos, e o segundo é o que mais importa:

    1. LIMIAR — a janela só conta como movimento se a probabilidade passar de
       `config.LIMIAR`. Abaixo disso é tratada como descanso.

    2. TRAVA — depois de fechar, o estado fica travado por `config.SEGURA_S` sem
       reavaliar. Isso não é conveniência, é necessidade: a evidência que a rede usa é o
       transiente da transição, e ele some da janela deslizante poucos segundos depois do
       início do movimento. Exigir confirmação contínua faria a mão reabrir sozinha
       enquanto o usuário ainda está tentando fechá-la.

    3. REFRATÁRIO — depois de soltar, um tempo morto antes de poder acionar de novo,
       para a órtese não ficar batendo.

    A votação por maioria (M de N janelas) foi testada e descartada: medida em 6 runs
    contínuos, ela derruba a detecção junto com o ruído, porque o pico de evidência dura
    cerca de uma janela e os disparos falsos também são isolados. Exigir 3 de 4 zerava as
    duas coisas. O parâmetro continua exposto para quem quiser re-testar com outro modelo.
    """

    def __init__(self, classe=None, limiar=None, segura=None, refratario=None, votos=(1, 1)):
        import collections
        self.classe = classe or cfg.CLASSE_ACIONA
        self.limiar = cfg.LIMIAR if limiar is None else limiar
        self.segura = cfg.SEGURA_S if segura is None else segura
        self.refratario = cfg.REFRATARIO_S if refratario is None else refratario
        self.m, n = votos
        self.hist = collections.deque(maxlen=n)
        self.travado_ate = None
        self.livre_em = 0.0

    def rotulo(self, prob):
        """Rótulo da janela isolada: movimento só acima do limiar, senão descanso."""
        for i, nome in sorted(cfg.CLASSES.items()):
            if nome != "rest" and prob[i] > self.limiar:
                return nome
        return "rest"

    def passo(self, prob, agora):
        """(rótulo da janela, ação) com ação em {None, 'FECHAR', 'ABRIR'}."""
        r = self.rotulo(prob)
        if self.travado_ate is not None:
            if agora >= self.travado_ate:
                self.travado_ate = None
                self.livre_em = agora + self.refratario
                self.hist.clear()
                return r, "ABRIR"
            return r, None
        self.hist.append(r)
        if agora < self.livre_em:
            return r, None
        if self.hist.count(self.classe) >= self.m:
            self.travado_ate = agora + self.segura
            self.hist.clear()
            return r, "FECHAR"
        return r, None
