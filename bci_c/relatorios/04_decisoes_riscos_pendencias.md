# 04 — Decisões, riscos e pendências

## Decisões tomadas, e o que cada uma custa

### Treinar em C também — e isso é obrigatório, não opcional

**Decisão.** O treino inteiro vai para C (leitura de EDF, backprop, Adam), rodando
no PC, com o mesmo `src/rede.c` do alvo embarcado.

**Consequência.** É o que dissolve o maior risco técnico do porte. Com treino e
inferência na mesma cadeia, o DSP em C não precisa reproduzir o `mne.filter`
numericamente — basta ser correto e consistente consigo mesmo. Sem o treino em C,
seria obrigatório imitar bit a bit o resampler FFT do MNE, porque o modelo
treinado espera exatamente aquela distribuição.

**O que custa.** O critério de aceitação deixa de ser "diferença numérica por
etapa" e passa a ser **acurácia de validação equivalente**. E o `rede_grad.c` é
o maior volume de código do projeto (**estimado** em ~1000–1500 linhas).

### Baseline: comparar contra o `.pt` commitado

**Decisão** (escolha do usuário, entre três opções apresentadas). O
`modelos/cnn2d_transicao.pt` é a referência, em vez de retreinar o Python nos 10
sujeitos locais ou baixar os 109 do eegmmidb.

**Consequência boa, que não estava prevista.** Habilitou um teste mais forte do
que o planejado: com os **mesmos pesos** nos dois lados, o forward em C ganhou uma
verdade-base absoluta. Esse é o teste que já passou (ver `01`), e ele não existiria
se o baseline fosse um modelo treinado em C.

**O risco que isso carrega.** Só existem 10 sujeitos em disco
(`dados/physionet/S001..S010`, **medido**: 60 EDFs, 148 MB) e o
`SUJEITOS_RUINS = [88, 89, 92, 100]` do `config.py` só faz sentido com os 109 do
eegmmidb. Se o `.pt` foi treinado com mais dado do que há aqui, **um modelo C
treinado só nesses 10 vai parecer pior sem estar errado**.

Há um indício em sentido contrário: um comentário no `treinar.py` cita "3.700
janelas", compatível com ~60 arquivos × ~61 janelas. **Não confirmado.** Resolve-se
com um comando, que é o passo zero da fase de treino:

```bash
python treinar.py --conferir     # monta o dataset e imprime as contas, não treina
```

**Mitigação já decidida.** O `treinar.c` avaliará **os dois** modelos no mesmo
conjunto de validação gerado pelo C. Assim a comparação de acurácia é maçã com
maçã, independente de com o que cada um foi treinado.

### Escopo: treino + pré-processamento + inferência

**Decisão** (escolha do usuário). Dentro: EDF, DSP, treino, inferência offline,
tempo real lendo a Cyton e imprimindo a classe no stdout. **Fora**: tela de dicas
(OpenCV) e acionamento do servo/órtese.

**Consequência.** O `nucleo/servo.py` (480 linhas, **medido**) e o `capturar.py`
(289 linhas) seguem sendo Python. Gravar sujeito novo e acionar a órtese continuam
dependendo do PC. O projeto C entrega classificação, não atuação.

### Tempo real também no Windows

**Decisão** (escolha do usuário, contra a recomendação de "só no Pi").

**Consequência.** A aquisição precisa de **duas** camadas serial atrás da mesma
interface: Win32 (`CreateFile`/`SetCommState`) e POSIX (`termios.h`).
**Estimado**: ~150 linhas extras e uma superfície a mais para testar. Em troca, o
tempo real em C pode ser exercitado com a Cyton no COM9 antes de haver um Pi.

Isto também determinou a escolha de toolchain: WSL2 não acessa porta COM
nativamente (precisaria de `usbipd-win`), então WSL estava descartado.

### Toolchain: MSYS2 + mingw-w64

**Decisão** (escolha do usuário). **Medido**: não havia nenhum compilador C nesta
máquina antes — nem gcc, nem clang, nem MSVC, nem MinGW, nem MSYS2; o `wsl.exe`
existia sem distribuição instalada. Instalado MSYS2 em `C:\msys64` com gcc 16.2.0
e make 4.4.1.

**Consequência.** gcc nativo Windows, enxerga COM9, sem VM. O custo é a
peculiaridade do `TMP` documentada em `01` — o build tem que passar pelo shell de
login do MSYS2.

### BatchNorm colapsado, mas não dobrado nas convoluções

**Decisão.** 4 tensores → 2 (`s`, `b`) no exportador; cada BN continua uma camada
no `rede.c`.

**Consequência.** Abre mão de ~3% de desempenho (**derivado**) para não abrir a
superfície de erro da álgebra entre camadas. A álgebra completa está escrita em
`03_mecanismos.md`, seção 2, pronta para ser usada se o número do Pi exigir.

### Reamostragem polifásica, e reamostrar antes de filtrar

**Decisão.** Polifásico racional (4/5 para o EDF a 160 Hz, 64/125 para a Cyton a
250 Hz) em vez do resampler FFT do MNE. E reamostrar **primeiro**, filtrar depois,
a 128 Hz.

**Consequência.** ≈3,8× menos trabalho de filtragem (**derivado**: 3301 taps ×
3028 amostras contra 1690 × 1550), estrutura *streaming-friendly* para o tempo
real, e muito mais simples de acertar. É um desvio deliberado da ordem do Python,
válido porque o passa-baixas do polifásico já faz o anti-aliasing até 64 Hz, bem
acima dos 35 Hz de interesse.

**O que custa.** Divergência numérica mensurável contra o Python, que será
quantificada pelo arnês e julgada pela acurácia — não escondida.

### Convolução direta antes de FFT

**Decisão.** A primeira versão do `dsp.c` usa convolução direta, sem biblioteca de
FFT.

**Consequência.** Mantém o projeto em **zero dependências** por mais tempo e
remove uma variável do primeiro teste de DSP. O custo **derivado** é
1690 × 1550 × 3 ≈ 7,9 M MACs por janela só de filtragem — mais que a rede inteira.
Se o número do Pi apertar, a convolução por FFT (via `kissfft`, BSD, 4 arquivos)
é a otimização evidente, com queda estimada para ~10% disso.

### Não usar biblioteca de *projeto* de filtro

**Decisão.** `firwin` próprio (sinc janelada por Hamming, ~40 linhas). Não usar
`liquid-dsp`, `iir1`, CMSIS-DSP ou KFR para **projetar** o filtro.

**Consequência, e é a mais importante do DSP.** Pedir "passa-banda 0,5–35 Hz" a
qualquer dessas bibliotecas devolve um filtro com resposta de **fase** diferente.
O método inteiro deste projeto depende do transitório: a janela é centrada na
transição descanso→tarefa, logo o que a rede classifica *é a forma do
transitório*. Mudar a fase do filtro muda precisamente a feature aprendida.
Bibliotecas são bem-vindas para **aplicar** (dado um vetor de coeficientes), nunca
para projetar.

## Pendências, em ordem de risco decrescente

| # | peça | por que nesta posição |
|---|---|---|
| 1 | `dsp.c` + `preproc.c` | é onde a divergência contra o Python aparece; define se a acurácia fecha |
| 2 | `edf.c` | EDF+ e as anotações T0/T1/T2; sem isso não há dataset |
| 3 | `rede_grad.c` + `adam.c` | maior volume; backprop errado é difícil de notar sem gradiente numérico |
| 4 | `alvos/treinar.c` | aqui sai o número de acurácia que fecha o eixo A |
| 5 | `cyton.c` | duas plataformas; a escala de contagens→Volts é o detalhe perigoso |
| 6 | `inferir.c`, `tempo_real.c` | montagem, pouco risco próprio |

O próximo marco natural é **1 + 2**, que termina em algo mensurável: o C monta o
dataset dos 10 sujeitos e as janelas que ele produz são comparadas, etapa por
etapa, contra as do `nucleo/dados.py`.

## Riscos, e o que sinalizaria cada um

| risco | gravidade | como apareceria |
|---|---|---|
| backprop em C com erro sutil | **alta** | treino converge, mas acurácia fica 5+ pp abaixo. Mitigação obrigatória: checagem por **gradiente numérico** por camada antes de confiar no `treinar.c` |
| escala da Cyton errada | **alta** | tempo real "funciona" mas classifica mal: z-score do checkpoint deixa de servir. Conferir contra uma gravação do BrainFlow, não contra a aritmética |
| acurácia do C abaixo do `.pt` por falta de dado, não por bug | média | indistinguível de bug sem a mitigação do conjunto de validação comum |
| `logf` caro no ARMv6 | média | 39.300 chamadas por janela (**derivado**); pode dominar o tempo. Só aparece medindo |
| FIR mais longo que o sinal no Python | baixa | **derivado, não confirmado**: ≈3301 taps sobre 3028 amostras. Se confirmado, a cadeia Python já opera em regime degenerado — interessante, mas não bloqueia o C |
| resync de pacote na serial | baixa | o BrainFlow esconde isso hoje; em C é código explícito a escrever e testar |

## Afirmações ainda não verificadas

Lista explícita, para que nada aqui seja confundido com medição:

1. **Latência no Pi Zero v1.3.** Não medida. Estimativa de 0,12–0,25 s para o
   forward sozinho pode errar por um fator.
2. **Se o `.pt` foi treinado com mais de 10 sujeitos.** Desconhecido.
   `treinar.py --conferir` resolve.
3. **O comprimento real do FIR do MNE** (≈3301 taps) e o regime de "filtro mais
   longo que o sinal". Derivado por leitura da regra `6,6/transição`.
4. **A escala contagens→Volts da Cyton** (≈2,235e-8 V/contagem). Derivada de
   4,5 V / ganho 24 / (2²³−1); não conferida contra o que o BrainFlow aplica.
5. **O custo de `logf` no ARMv6.** Não medido.
6. **Os internals de `mne.filter.resample`.** Não investigados — e, pela decisão
   de treinar em C, não precisam mais ser.

## O que já dá para medir no Pi

Com o que existe hoje, **dois dos três eixos** de comparação já podem ser
fechados — inclusive o eixo C, que é o número que falta:

| eixo | o que o Pi responde | status |
|---|---|---|
| **A. acurácia** | o C treinado bate o `.pt`? | falta `treinar.c` |
| **B. identidade numérica** | C no PC e C no Pi dão o mesmo? | **testável agora** |
| **C. latência** | quantos ms por janela no ARMv6? | **testável agora** |

Não dá para testar: Cyton, pré-processamento, classificação de EEG real, loop de
tempo real. O que roda é: janela já normalizada entra, probabilidade sai.

### Procedimento

O pacote `artefatos/para_o_pi.tar.gz` (**medido**: 644.749 bytes) já contém
`Makefile`, `CONFIG.h`, `src/`, `alvos/`, o `modelo.bin` e os vetores de
referência gerados no PC.

**1. Sistema operacional — atenção, é o inverso do caso do Zero 2 W.** No Zero
**v1.3** é obrigatório o **Raspberry Pi OS 32 bits**: a imagem 64 bits **não dá
boot** num ARMv6. (A regra dos 64 bits valia para o Zero 2 W, que é ARMv8.)

**2. Transferência.** O v1.3 não tem Wi-Fi nem Bluetooth — o rádio entrou no Zero
W. Então: cartão SD direto no PC, modo gadget USB (`g_ether`/`g_serial`), ou
adaptador USB-ethernet.

**3. Compilar:**

```bash
sudo apt install -y build-essential          # já vem na imagem completa
tar -xzf para_o_pi.tar.gz
make CFLAGS="-std=c99 -O2 -Wall -mcpu=arm1176jzf-s -mfpu=vfp"
```

**4. Latência** — use 20 repetições, não 200 (a 6,1 ms no PC e talvez 30× mais
lento, 200 passariam de meio minuto):

```bash
./artefatos/bancada tempo artefatos/modelo.bin 20
```

**5. Identidade numérica PC ↔ Pi:**

```bash
./artefatos/bancada rede artefatos/modelo.bin \
    artefatos/paridade_entrada.bin artefatos/saida_pi.bin

./artefatos/bancada comparar artefatos/paridade_saida.bin artefatos/saida_pi.bin
```

Nenhum Python necessário no Pi.

### Como ler os resultados

**Identidade:** esperar `max |dif|` da ordem de 1e-7 e `argmax igual 96/96`.
Divergência maior é algo real a investigar — suspeito provável é a ordem de
avaliação de ponto flutuante no ARMv6 sem NEON, mas aí o número importa.

**Latência**, que decide o `PASSO_S`:

| se der | significa |
|---|---|
| < 50 ms | confortável; sobra orçamento para o pré-processamento |
| 50–150 ms | `PASSO_S` de 0,25 s viável, 0,1 s não |
| > 250 ms | o forward sozinho come o passo; a conversa vira otimização (FFT na filtragem, BN dobrado, `logf` tabelado) |

Qualquer um dos três é um bom resultado, porque é **medida** em vez de
estimativa. Os dois números transformam o planejamento de `dsp.c` e `preproc.c`
de orçamento suposto em orçamento conhecido.
