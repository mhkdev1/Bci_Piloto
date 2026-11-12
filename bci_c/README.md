# bci_c — o projeto em C, treinável no PC e embarcável no Pi

Reimplementação em C do caminho completo: leitura, pré-processamento, treino e
inferência. Zero dependências externas — C99 e `libm`. Sem torch, sem MNE, sem
BrainFlow. O projeto Python ao lado permanece intocado e serve de referência.

O `src/rede.c` é **o mesmo arquivo** nos três alvos (treino no PC, inferência
offline, tempo real no Pi). Não existe caminho de código que só o PC exercite —
é isso que faz "pronto para embarcar" ser verdade e não intenção.

## Documentação

Os relatórios em [relatorios/](relatorios/) são a referência completa — o que
existe, como foi verificado, a diferença peça por peça do projeto Python, o
mecanismo de cada parte, e as decisões com suas consequências. Este README é só
o cartão de entrada.

| documento | responde |
|---|---|
| [01 — Estado e verificação](relatorios/01_estado_e_verificacao.md) | o que existe e todos os números medidos |
| [02 — C versus Python](relatorios/02_c_versus_python.md) | a diferença peça por peça, e a consequência de cada uma |
| [03 — Mecanismos](relatorios/03_mecanismos.md) | como cada peça funciona por dentro |
| [04 — Decisões, riscos e pendências](relatorios/04_decisoes_riscos_pendencias.md) | por quê, o que falta, o que pode dar errado |

## Estado atual

| peça | estado |
|---|---|
| `src/rede.c` — forward da CNN2D | **pronto e verificado** |
| `ferramentas/exportar_pesos.py` — `.pt` → `modelo.bin` | **pronto** |
| `teste/comparar.py` — paridade C vs PyTorch | **pronto, passando** |
| `alvos/bancada.c` — arnês de comparação e medição | **pronto** |
| `src/dsp.c` — firwin, convolução centrada, polifásico, detrend | a fazer |
| `src/preproc.c` — a cadeia completa | a fazer |
| `src/edf.c` — EDF+ → janelas centradas na transição | a fazer |
| `src/rede_grad.c` + `adam.c` — backprop e otimizador | a fazer |
| `src/cyton.c` — serial, Win32 + POSIX | a fazer |
| `alvos/treinar.c`, `inferir.c`, `tempo_real.c` | a fazer |

## Resultado da paridade do forward

Com os pesos do `modelos/cnn2d_transicao.pt` carregados dos dois lados, 96
janelas (incluindo quatro casos de borda deliberados):

```
max |dif| .......: 4.172e-07   (tolerância 1e-04)
média |dif| .....: 5.610e-08
argmax igual ....: 96/96
```

O épsilon do float32 é 1,19e-07, então 4,17e-07 é ruído de arredondamento: a
ordem das somas nas convoluções difere entre o C e o BLAS do torch e muda os
últimos bits. Contra o `LIMIAR` de 0,65, uma discrepância dessa ordem nunca
inverte um acionamento.

**Não se exige igualdade bit a bit** — ela não existe e não seria informativa.
Exige-se que a diferença seja irrelevante para a decisão.

## Latência medida

| etapa | PC (x86_64, gcc 16.2, -O2) | Pi Zero v1.3 |
|---|---|---|
| forward da rede | **6,1 ms/janela** (164/s) | a medir |

O número do Pi ainda **não foi medido** — o ARM1176 não tem NEON e o `-O2` aqui
auto-vetoriza, então a razão não é estimável com confiança. A faixa que eu
esperaria para o forward sozinho é 0,12 a 0,25 s, o que já consome o orçamento
inteiro de um `PASSO_S` de 0,1 s e empurra o passo viável para 0,25–0,5 s.
Medir isso é o primeiro item quando houver um Pi na mão.

## Como rodar

```bash
# 1) compilar  (no Windows, DENTRO do shell do MSYS2 — ver nota abaixo)
make

# 2) exportar os pesos do .pt commitado
python ferramentas/exportar_pesos.py

# 3) paridade C vs PyTorch
python teste/comparar.py

# 4) latência do forward
./artefatos/bancada tempo artefatos/modelo.bin 200
```

### Nota sobre o build no Windows

O gcc do mingw-w64 chamado direto do Git Bash falha com
`Cannot create temporary file in C:\Windows\`: ele não herda um `TMP` utilizável.
Compile pelo shell de login do MSYS2, que monta o ambiente certo:

```bash
MSYSTEM=MINGW64 CHERE_INVOKING=1 /c/msys64/usr/bin/bash.exe -lc \
  'cd "/d/caminho/para/bci_c" && make'
```

### No Raspberry Pi Zero v1.3

```bash
make CFLAGS="-std=c99 -O2 -Wall -mcpu=arm1176jzf-s -mfpu=vfp"
```

O `-mfpu=neon` **não** compila no Zero v1.3: o BCM2835 é ARMv6 e não tem NEON.

## Decisões de projeto que valem saber

**Por que não é um port bit a bit do Python.** Como o treino também é em C,
treino e inferência passam pela mesma cadeia, então ela não precisa imitar o MNE
— precisa ser uma cadeia DSP correta e consistente consigo mesma. Isso libera
escolhas melhores para o alvo embarcado (reamostragem polifásica racional em vez
do resampler FFT do MNE) e move o critério de aceitação de "diferença numérica
por etapa" para **acurácia de validação equivalente**.

**Por que os BatchNorm não são dobrados nas convoluções.** Em eval cada BN é uma
afim de 2 floats por canal — 432 canais, ~227 mil operações contra 6,4 milhões
das convoluções. Dobrar economizaria ~3% e exigiria álgebra entre camadas (o
bias de um BN atravessando a conv seguinte), que é exatamente onde esse tipo de
port ganha um bug silencioso. O exportador colapsa os 4 tensores de cada BN em
dois (`s = γ/√(var+ε)`, `b = β − s·μ`), que é o único ganho que valia.

**Dois detalhes que quebram tudo em silêncio se errados**, comentados onde
acontecem no `rede.c`:

- `Conv2d` do PyTorch é **correlação cruzada**, não convolução — o kernel não se
  inverte.
- `AdaptiveAvgPool2d` de 131 para 32 **não é divisível**: as janelas têm
  tamanhos desiguais, pela regra `floor`/`ceil` do PyTorch. Pooling uniforme dá
  outro número.

## Ressalva sobre o baseline de acurácia

O `cnn2d_transicao.pt` é o baseline escolhido para comparação, mas só existem 10
sujeitos do PhysioNet em disco (`dados/physionet/S001..S010`) e o
`SUJEITOS_RUINS = [88, 89, 92, 100]` do `config.py` sugere que ele pode ter sido
treinado com mais dado do que há aqui. Se for o caso, um modelo C treinado só
nesses 10 vai parecer pior sem estar errado.

Mitigação já prevista: o `treinar.c` avaliará **os dois** modelos no mesmo
conjunto de validação gerado pelo C. Assim a comparação é maçã com maçã,
independente de com o que cada um foi treinado.
