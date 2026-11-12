# BCI de imagética motora para acionamento de órtese

Interface cérebro-computador que lê EEG de três eletrodos, classifica a intenção de
movimento em **descanso / mão esquerda / mão direita**, e usa essa decisão para fechar
uma órtese de mão acionada por servo.

O pacote traz o código completo, os dados para reproduzir os resultados e um modelo já
treinado. Dá para rodar as três coisas — treino, inferência offline e inferência ao vivo
com a órtese — sem baixar mais nada, tirando as bibliotecas Python.

---

## O que tem aqui

```
bci_ortese/
├── config.py               todos os parâmetros, num arquivo só
├── verificar.py            confere se o Python tem as bibliotecas necessárias
├── treinar.py              treina a rede
├── inferir_offline.py      roda o modelo em dados gravados e mede a acurácia
├── tempo_real.py           capacete ligado, órtese acionada ao vivo
├── capturar.py             grava uma sessão nova
├── capturar_tempo_real.py  sessão guiada que aciona a órtese E grava o EEG
├── controle_servo_1.py     controle manual da órtese pelo teclado
├── testar_ortese.py        confere o protocolo PC <-> ESP32 (emulado ou com hardware)
├── testar_bluetooth.py     confere o controle da órtese pelo Bluetooth
├── testar_amplitude.py     confere que a órtese fecha o quanto foi pedido
├── firmware_esp32/
│   └── firmware_esp32.ino  o que roda DENTRO do ESP32 (servo no GPIO 18, Bluetooth e USB)
├── nucleo/
│   ├── preproc.py          filtros, janelamento e normalização
│   ├── modelo.py           a rede convolucional
│   ├── dados.py            leitura do PhysioNet e das gravações
│   └── servo.py            comando da órtese e lógica de acionamento
├── dados/
│   ├── physionet/          10 sujeitos do eegmmidb (6 runs cada)
│   └── gravacoes/          sessões gravadas com a OpenBCI Cyton
├── modelos/
│   └── cnn2d_transicao.pt  modelo treinado, pronto para uso
├── figuras/                curvas de treino e diagramas da rede
├── docs/                   logs de treino originais
├── apresentacao/           PDF de aula com o caminho inteiro, e o script que o gera
├── logs/                   um CSV por sessão de tempo real (criado ao rodar)
├── bci_c/                  reimplementação em C, com README próprio
├── requirements.txt        bibliotecas Python
└── RELATORIO.md            o método descrito passo a passo
```

O [RELATORIO.md](RELATORIO.md) explica como cada etapa funciona e por quê. Este arquivo
é só o manual de uso.

---

## Instalação

Python 3.9 ou mais novo.

```bash
pip install -r requirements.txt
python verificar.py          # confere se este Python tem tudo
```

Comece pelo `verificar.py`. Ele lista o que falta, mostra qual interpretador está sendo
usado e monta a linha de instalação correta.

**Se você tem mais de um Python instalado** — situação comum no Windows, onde `python` no
PATH pode não ser o mesmo que a sua IDE usa —, instale e execute sempre com o caminho
completo do mesmo interpretador:

```powershell
& "C:\caminho\para\python.exe" -m pip install -r requirements.txt
& "C:\caminho\para\python.exe" inferir_offline.py --sujeito S001
```

O sintoma de estar no interpretador errado é um `ModuleNotFoundError: No module named
'torch'` logo na primeira linha, mesmo tendo instalado tudo.

Para rodar apenas o treino e a inferência offline, bastam `numpy`, `scipy`, `torch`,
`mne`, `pandas` e `matplotlib`. As demais só são necessárias com hardware:
`brainflow` (capacete), `pyserial` (órtese), `opencv-python` (tela de dicas) e `pynput`
(controle manual da órtese).

---

## Uso rápido

### 1. Inferência offline — não precisa de hardware

Roda o modelo em dados já gravados e mostra acurácia, F1 por classe e matriz de confusão.

```bash
python inferir_offline.py --listar                    # o que existe em disco
python inferir_offline.py --sujeito S001              # um sujeito do PhysioNet
python inferir_offline.py --sujeito S001 --run 4      # um run só
python inferir_offline.py --gravacao S005_R1_cortado  # uma sessão da Cyton
python inferir_offline.py --gravacao S004_R1 --detalhe
```

Nas gravações contínuas ele faz uma segunda leitura, varrendo o sinal com janela
deslizante — é a simulação do que a órtese enfrentaria em tempo real. Para ver os
comandos que seriam enviados, sem hardware nenhum:

```bash
python inferir_offline.py --gravacao S005_R1_cortado --ortese --simular-ortese
```

Com a órtese ligada, tire o `--simular-ortese`.

### 2. Treino

```bash
python treinar.py --conferir            # monta o dataset e mostra as contas, sem treinar
python treinar.py                       # PhysioNet + gravações, 35 épocas
python treinar.py --sujeitos 5          # subconjunto, para um teste rápido
python treinar.py --sem-gravacoes       # só PhysioNet
```

Roda na CPU. Com os 10 sujeitos deste pacote, leva alguns minutos; o modelo distribuído
foi treinado com 105 sujeitos e levou cerca de duas horas.

### 3. Tempo real com a órtese

```bash
python tempo_real.py --sem-ortese       # só imprime, para testar o capacete sozinho
python tempo_real.py --simular-ortese   # imprime os comandos que mandaria
python tempo_real.py                    # capacete + órtese
python tempo_real.py --amplitude 21     # fecha só 21°
python tempo_real.py --intervalo 0.02   # fecha mais devagar (ver "Velocidade e suavidade")
```

Para testar com uma pessoa, use a **sessão guiada**. Ela mostra as dicas na tela (2 min de
descanso e depois 20 tentativas de descanso/mão, 4,1 s cada) e calcula no fim a detecção,
a latência e os disparos falsos por minuto:

```bash
python tempo_real.py --protocolo --sujeito S006 --sem-ortese   # primeiro sem órtese
python tempo_real.py --protocolo --sujeito S006                # depois com ela
python tempo_real.py --protocolo --ensaio --sem-ortese         # sem capacete, placa sintética
```

Toda sessão, guiada ou não, grava `logs/<sujeito>_<data>_<hora>.csv` com uma linha por
inferência, e a guiada grava também `_resumo.txt`. Os critérios de contagem estão no
cabeçalho do `tempo_real.py`. `--tentativas` e `--descanso-inicial` mudam os tamanhos.

Antes, ajuste as portas em `config.py`:

```python
PORTA_EEG   = "COM9"     # dongle da Cyton      (Linux: /dev/ttyUSB0)
PORTA_SERVO = ""         # ESP32 da órtese: vazio = Bluetooth; "COM11" = cabo USB
```

O ESP32 aceita os dois caminhos ao mesmo tempo, e quem escolhe é o `PORTA_SERVO`:

- **vazio** — o programa acha sozinho o ESP32 pareado por Bluetooth, pelo nome do aparelho
  (ver [Conectando por Bluetooth](#conectando-por-bluetooth));
- **`"COM11"`** (a porta do cabo no seu PC) — usa o cabo USB e nem procura o Bluetooth.

Para listar as portas:

```bash
python -c "import serial.tools.list_ports as p; [print(x.device,'-',x.description) for x in p.comports()]"
```

### 4. Gravar uma sessão nova

```bash
python capturar.py --ensaio                  # sem capacete: confere tempos e formato
python capturar.py --sujeito S006 --run R1   # ~4 minutos
```

A gravação vai para `dados/gravacoes/S006_R1/` e já pode ser usada no treino e na
inferência offline.

### 5. Sessão guiada que também grava

O `capturar_tempo_real.py` junta os dois anteriores: mostra as dicas da sessão guiada,
aciona a órtese pela decisão da rede e grava o EEG no mesmo formato do `capturar.py`.

```bash
python capturar_tempo_real.py --ensaio --sem-ortese            # sem hardware nenhum
python capturar_tempo_real.py --sujeito S006 --simular-ortese  # EEG real, órtese só no terminal
python capturar_tempo_real.py --sujeito S006                   # sessão real
```

A sessão vai para `dados/gravacoes/<sujeito>_RT1/` (depois `RT2`, `RT3`, ... — repetir o
comando nunca sobrescreve) com `continuo.csv`, `eventos.csv`, `inferencias.csv` e
`resumo.txt`. Aceita `--amplitude` e `--intervalo` como o `tempo_real.py`.

Essas pastas entram no próximo `treinar.py` sozinhas. Os rótulos seguem a dica, não o que
a rede detectou, e nelas a pessoa vê a mão fechar — para treinar sem uma sessão, mova a
pasta para fora de `dados/gravacoes/`. Os detalhes estão no cabeçalho do script.

---

## Montagem dos eletrodos

Três canais no sistema 10-20, sobre o córtex motor:

| canal | posição | o que capta |
|---|---|---|
| C3 | hemisfério esquerdo | movimento da mão **direita** |
| C4 | hemisfério direito | movimento da mão **esquerda** |
| Cz | linha média | referência de atividade motora central |

A ordem em `config.CANAIS` precisa bater com a ordem física na Cyton — é a mesma ordem
que a rede viu no treino, e trocá-la degrada o resultado sem gerar erro nenhum.

---

## A órtese

O ESP32 recebe **um byte por comando, sem terminador**, e o movimento é **incremental**:

| byte | efeito |
|---|---|
| `R` | `angle = min(180, angle + passo)` |
| `L` | `angle = max(-30, angle - passo)` |
| `C` | `angle = 90` — o repouso é o **centro** do curso, não zero |
| `0`–`9` | tamanho do passo, em graus (o dígito *é* o passo; `0` vira 1) |

A cada byte o firmware imprime `angle=<n>`. O lado do PC lê esse eco e o usa como
posição real, em vez de contar pulsos às cegas — e precisa lê-lo de qualquer forma,
senão a saída se acumula no buffer do host durante a sessão inteira.

Fechar 30° quer dizer ir de 90° para 120° (ou para 60°, se a sua montagem fechar no
outro sentido). Com passo de 1°, são 30 pulsos; com passo de 3°, 10.

Há dois lados, e eles não se confundem:

| arquivo | onde roda | papel |
|---|---|---|
| `firmware_esp32/firmware_esp32.ino` | **dentro do ESP32** | recebe os bytes (Bluetooth ou USB) e move o servo |
| `firmware_esp32_seguro/` | dentro do ESP32 | variante com batente e retorno automático |
| `firmware_esp32_livre/` | dentro do ESP32 | curso livre 0..180° e comandos de console (`A<n>`, `P<n>`, `N/X<n>`, `U<a>,<b>`, `?`) |
| `controle_servo_1.py` | no PC | controle manual por teclado |
| `nucleo/servo.py` | no PC | é quem a inferência usa |
| `testar_ortese.py`, `testar_bluetooth.py`, `testar_amplitude.py` | no PC | testes da órtese |

### Gravando o firmware

Abra `firmware_esp32/firmware_esp32.ino` na IDE do Arduino, instale a biblioteca
**ESP32Servo** pelo Gerenciador de Bibliotecas, selecione a sua placa ESP32 e grave. A
biblioteca do Bluetooth (`BluetoothSerial`) já vem com o pacote de placas esp32.

Ligação:

```
servo (sinal)  ->  GPIO 18
servo (VCC)    ->  fonte externa 5 V   (não use o 5V do ESP32)
servo (GND)    ->  GND comum com o ESP32
```

O GND precisa ser comum, senão o servo treme ou não responde. E alimentar o servo pelo
regulador da placa costuma derrubá-la no pico de corrente da partida.

### Conectando por Bluetooth

O firmware fala o mesmo protocolo pelo Bluetooth e pelo cabo USB, ao mesmo tempo. Pelo
Bluetooth o ESP32 aparece como **`BCI_Ortese`** (perfil serial, SPP), e o resto não muda:
mesmos comandos, mesma ordem, mesma saída.

1. Grave o firmware e ligue o ESP32 — pelo cabo ou por uma fonte.
2. No Windows: **Configurações → Dispositivos → Bluetooth e outros dispositivos →
   Adicionar Bluetooth ou outro dispositivo → Bluetooth → BCI_Ortese**. O ESP32 aceita o
   pareamento sozinho, sem PIN.
3. Deixe `PORTA_SERVO = ""` no `config.py` e rode como sempre:

```bash
python testar_ortese.py --real     # com a órtese VAZIA
python testar_bluetooth.py         # conexão, movimento, limite e decisão ao vivo
python tempo_real.py
```

```
ESP32 'BCI_Ortese' encontrado no Bluetooth, porta COM5
conectando à órtese em COM5 (pelo Bluetooth leva alguns segundos)...
órtese em COM5 | repouso 90°, fechado 120° (amplitude 30°), passo 1°, fecha com 'R', intervalo 0.01 s (100°/s)
```

O Windows mostra o aparelho como "Pareado", não "Conectado", e está certo assim: a conexão
abre quando um dos programas abre a porta, e fecha quando ele termina.

O Bluetooth muda três coisas, e o código já trata as três:

- **A porta.** O Windows cria **duas** portas COM por aparelho pareado e só a de saída
  conecta — e fones e mouses pareados também criam portas. Por isso a porta é achada pelo
  nome (`SERVO_BT_NOME`), nunca por "a única porta Bluetooth". Se não achar, o programa
  lista as portas, marca quais são Bluetooth e pergunta. Para forçar uma porta, escreva-a
  em `PORTA_SERVO`.
- **A latência.** O eco `angle=<n>` chega com atraso variável. O PC espera um eco por byte
  enviado, até `SERVO_TIMEOUT_ECO_S`, em vez de ler depois de um tempo fixo: ler cedo
  pegaria o ângulo de um pulso do meio, e o comando seguinte passaria da amplitude. O teste
  6 do `testar_ortese.py` reproduz isso sem hardware.
- **A queda.** Se o link cair, o **firmware devolve a órtese ao repouso sozinho** — na
  hora, se o programa fecha a porta; em até uns 20 s, se o link some sem desconectar — e o
  PC tenta reconectar (três tentativas) e segue a sessão. Se não conseguir, encerra pelo
  caminho normal. Com a órtese vestida, prefira o `firmware_esp32_seguro/`, que também
  volta ao repouso após 10 s sem comando.

Se trocar o nome, troque nos dois lados: `NOME_BT` no `.ino` e `SERVO_BT_NOME` no
`config.py`. Só o ESP32 clássico (WROOM/WROVER, DevKit V1) tem Bluetooth serial; S2, S3,
C3 e C6 não têm, e o firmware para com um erro claro na compilação. Use o pacote de placas
esp32 **3.x** (compilado na 3.3.2: 81% da flash do ESP32 Dev Module) — a `BluetoothSerial`
está marcada para deixar de vir por padrão na 4.0, e o *warning* de compilação que avisa
isso pode ser ignorado.

O firmware em uso vai até 180° — quem limita o curso à amplitude da órtese é o PC
(`config.SERVO_AMPLITUDE = 30`). Isso basta enquanto o PC for quem comanda, mas não
protege contra um terminal aberto por engano nem contra o programa travar com a mão
fechada.

Para uso com a órtese **vestida**, grave a variante `firmware_esp32_seguro/`: ela põe o
batente também no firmware e devolve a órtese ao repouso sozinha se nenhum comando
chegar por 10 s. Protocolo e Bluetooth são os mesmos, então o lado do PC não muda. Ajuste
`SENTIDO_FECHA` e `AMPLITUDE` no topo do `.ino` para baterem com o `config.py`, e descubra
o sentido que fecha **antes**, com o firmware principal: o batente não deixa o servo passar
do repouso para o outro lado.

### Conferindo a ligação

Primeiro o teste de comunicação, que roda **sem hardware nenhum** — ele emula o ESP32 em
software e confere se o PC e o firmware concordam em todas as regras:

```bash
python testar_ortese.py            # emulado
python testar_ortese.py --real     # com a órtese ligada, e VAZIA
```

Depois o controle manual, que é como você descobre qual sentido fecha a sua montagem:

```bash
python controle_servo_1.py       # setas ← → movem, ↑ volta ao repouso, ESC sai
```

Depois ajuste em `config.py`:

```python
SERVO_SENTIDO_FECHA = "R"    # qual letra fecha a SUA montagem
SERVO_PASSO_GRAUS   = 1      # 1 a 9 graus por pulso; 1 é o menor que o firmware aceita
SERVO_AMPLITUDE     = 30     # curso máximo, em graus a partir do repouso
SERVO_INTERVALO_S   = 0.01   # espera entre pulsos, em segundos
```

Nenhum desses exige regravar o ESP32: o PC manda o passo ao conectar, e a amplitude e o
intervalo só existem do lado do PC.

**O curso é limitado a 30°.** É o limite mecânico confortável da montagem usada: acima
disso a haste força a articulação do dedo. O módulo nunca envia pulsos além do teto, e
trunca para baixo quando o ângulo não é múltiplo do passo — pedir 30° com passo de 4°
para em 28°, nunca em 32°.

Para fechar **menos** numa sessão, sem editar nada nem regravar o ESP32, use `--amplitude`
no `tempo_real.py`, no `capturar_tempo_real.py`, no `testar_bluetooth.py` ou no
`inferir_offline.py --ortese`:

```bash
python testar_bluetooth.py --amplitude 21   # confere o curso menor com a órtese vazia
python tempo_real.py --amplitude 21         # a sessão fecha só 21°
```

Para conferir vários cursos de uma vez, com a órtese **vazia**, o `testar_amplitude.py`
fecha em cada um e pergunta ao firmware onde o servo parou:

```bash
python testar_amplitude.py --porta COM11                     # 9°, 18°, 27° e o teto
python testar_amplitude.py --porta COM11 --amplitudes 12 21  # escolhe os valores
```

O `SERVO_AMPLITUDE` do `config.py` é o teto: pedir mais que ele dá erro. O valor é
arredondado para baixo, para um múltiplo do passo (20° com passo de 3° vira 18°). Com o
`firmware_esp32_seguro`, o `AMPLITUDE` gravado no `.ino` também limita: acima dele, o servo
para no batente.

### Velocidade e suavidade

O movimento é uma sequência de pulsos, e dois números o definem:

- **o passo** (`SERVO_PASSO_GRAUS`) — quantos graus cada pulso anda. Quanto menor, mais
  suave. O mínimo é 1°: o firmware guarda o ângulo como inteiro.
- **o intervalo** (`SERVO_INTERVALO_S`) — a espera entre um pulso e o seguinte. É o PC que
  espera, não o ESP32, então muda sem regravar nada.

A velocidade é passo ÷ intervalo. Para mudar o intervalo numa sessão, sem editar o
`config.py`, use `--intervalo` (em segundos) no `tempo_real.py`, no
`capturar_tempo_real.py`, no `inferir_offline.py --ortese`, no `testar_bluetooth.py` ou no
`testar_amplitude.py`:

```bash
python tempo_real.py --intervalo 0.08        # 1° a cada 80 ms = 12,5°/s
python testar_bluetooth.py --intervalo 0.05  # confere a velocidade com a órtese vazia
```

**O firmware tem um teto de velocidade.** Medido na órtese, pelo cabo: o ESP32 leva cerca de
40 ms para executar cada comando, porque o PWM de 50 Hz segura cada escrita no servo. São
uns 25 comandos por segundo, e esperar menos que isso entre as letras não adianta: elas
ficam na fila do ESP32.

| passo | velocidade máxima | tempo para fechar 80° |
|---|---|---|
| 1° | cerca de 25°/s | 3,2 s (medido) |
| 2° | cerca de 50°/s | 1,6 s (estimado) |
| 3° | cerca de 75°/s | 1,1 s (estimado) |

Na prática:

- para fechar **mais devagar**, aumente o `--intervalo` acima de `0.04`;
- para fechar **mais rápido**, aumente o `SERVO_PASSO_GRAUS`. O movimento fica menos suave,
  em degraus maiores;
- no `tempo_real.py` o programa fica parado enquanto manda os pulsos, e não faz inferência
  nesse tempo. Com passo de 1° e 80° de curso, são mais de 3 s por fechamento.

Como o firmware é incremental, a posição é mantida no PC — conferida pelo eco `angle=<n>`
— e ressincronizada a cada `C`. Toda sequência de fechamento parte do repouso, e o
programa devolve a órtese ao repouso ao terminar, inclusive se for interrompido. Sem
isso, um pulso perdido faria o erro acumular e o limite de 30° deixaria de valer.

## Antes de usar em alguém

Este sistema é um protótipo de pesquisa. Com limiar 0,80, cerca de 43% das tentativas
geram um comando e a órtese dispara sozinha aproximadamente uma vez por minuto. O
`config.py` está com 0,70, mais permissivo: na mesma medição, esse limiar detectou 68% das
tentativas e disparou sozinho uma vez a cada 29 s. Os números
completos, e como eles foram medidos, estão no [RELATORIO.md](RELATORIO.md).

Na prática isso significa: a órtese precisa estar sempre removível, o usuário nunca deve
ficar sozinho com o sistema em funcionamento, e o curso mecânico deve ser conferido antes
de cada sessão.

---

## Dados incluídos

**PhysioNet — EEG Motor Movement/Imagery Dataset (eegmmidb).** Dez sujeitos, seis runs
cada: 3, 7 e 11 (movimento real da mão) e 4, 8 e 12 (imagética). Base pública, disponível
em <https://physionet.org/content/eegmmidb/1.0.0/>. O modelo distribuído foi treinado com
105 sujeitos; os dez aqui bastam para reproduzir o pipeline e rodar um treino de
demonstração.

**Gravações próprias.** Sessões com a OpenBCI Cyton, três eletrodos, 250 Hz. As duas
abaixo foram gravadas com o `capturar.py`; as pastas `<sujeito>_RT<n>` são sessões do
`capturar_tempo_real.py`, gravadas com a órtese em uso.

| sessão | duração | segmentos | observação |
|---|---|---|---|
| `S004_R1` | 248 s | 60 | sessão completa |
| `S005_R1_cortado` | 193 s | 47 | a bateria acabou no meio; a parte instável foi removida |

O corte do S005 não foi arbitrário: a partir do evento 48 os segmentos passaram a mapear
para a mesma amostra, sinal de que a placa parou de transmitir. Tudo até ali está íntegro.

---

## Reprodutibilidade

Os resultados abaixo saem direto dos comandos indicados, sem nenhum ajuste:

| comando | acurácia | macro-F1 |
|---|---|---|
| `python inferir_offline.py --sujeito S001` | 73,6% | 0,723 |
| `python inferir_offline.py --gravacao S005_R1_cortado` | 66,7% | 0,634 |

O treino usa semente fixa (`config.SEMENTE`), mas o resultado exato ainda pode variar um
pouco entre máquinas por causa de diferenças de BLAS e de versão do PyTorch.

---

## Citação

Se este material for útil no seu trabalho, cite o DOI do depósito no Zenodo. Os dados do
PhysioNet têm citação própria:

> Goldberger, A., et al. (2000). PhysioBank, PhysioToolkit, and PhysioNet: Components of
> a New Research Resource for Complex Physiologic Signals. *Circulation*, 101(23),
> e215–e220.

> Schalk, G., McFarland, D.J., Hinterberger, T., Birbaumer, N., Wolpaw, J.R. (2004).
> BCI2000: A General-Purpose Brain-Computer Interface (BCI) System. *IEEE Transactions on
> Biomedical Engineering*, 51(6), 1034–1043.
