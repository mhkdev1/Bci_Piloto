# 02 — C versus Python: diferença peça por peça

O projeto Python (a pasta acima desta) **não foi alterado em nada**. Ele segue
sendo a referência executável e o gerador dos vetores de teste. Este documento
descreve o que o projeto C faz diferente, e o que cada diferença custa e compra.

## O que NÃO mudou

Começar por aqui, porque é a maior parte e é o que dá segurança:

- **A arquitetura da rede.** Mesmos 5 ramos temporais, mesmos kernels
  `{5,11,15,33,65}`, mesmo `depth=3`, mesmo `Fs=300`, mesma cadeia
  `square → avgpool → log`, mesmo `AdaptiveAvgPool` para 32 bins, mesmo
  `Linear(1024,3)`. 21.519 parâmetros nos dois.
- **A geometria da janela.** 4,1 s, 526 amostras a 128 Hz, centrada na
  transição, metade de cada segmento vizinho.
- **O método.** As duas janelas (descanso→tarefa = movimento, tarefa→descanso =
  descanso) ladrilham a linha do tempo sem sobreposição e sem buraco.
- **As classes e o significado da normalização.** `{0: rest, 1: left, 2: right}`,
  z-score por canal com a estatística **do treino**, carregada do checkpoint.
- **Os parâmetros.** O `CONFIG.h` é um espelho 1:1 do `config.py`, com os mesmos
  nomes, para que divergência seja visível num diff lado a lado.

O que mudou é **como** essas coisas são calculadas, e por quais dependências.

## Tabela de diferenças

| peça | Python | C | estado |
|---|---|---|---|
| dependências | 10 pacotes¹ | C99 + `libm` | feito |
| forward da rede | `torch.nn` (`nucleo/modelo.py`) | laços explícitos (`src/rede.c`) | **verificado** |
| BatchNorm em inferência | camada do torch, 4 tensores | 2 floats/canal, pré-colapsados | **verificado** |
| checkpoint | `.pt` (pickle, exige torch) | `modelo.bin` (86.172 B, plano) | **verificado** |
| normalização | `preproc.normalizar_com_modelo` | `rede_normalizar` | feito |
| filtro notch + passa-banda | `mne.filter` (firwin do MNE) | firwin próprio | a fazer |
| reamostragem | FFT do MNE (`raw.resample`) | **polifásico racional** | a fazer |
| ordem filtro/reamostragem | filtra a 250 Hz, depois reamostra | **reamostra, depois filtra a 128 Hz** | a fazer |
| detrend | `scipy.signal.detrend` | mínimos quadrados próprio | a fazer |
| leitura de EDF | `mne.io.read_raw_edf` | parser próprio / `edflib` | a fazer |
| épocas | `mne.Epochs` | extração própria | a fazer |
| gradientes | autograd do torch | backprop à mão | a fazer |
| otimizador | `torch.optim.Adam` | Adam próprio | a fazer |
| aquisição da Cyton | BrainFlow | serial próprio, Win32 + POSIX | a fazer |
| tela de dicas | OpenCV | **fora de escopo** | — |
| órtese / servo | `nucleo/servo.py` | **fora de escopo** | — |

¹ `numpy, scipy, pandas, torch, mne, matplotlib, brainflow, pyserial,
opencv-python, pynput`

### Custo em linhas

**Medido** com `wc -l`. Para a única peça já portada:

| | Python | C | razão |
|---|---:|---:|---:|
| forward da rede | 154 (`nucleo/modelo.py`) | 338 (`src/rede.c`) | 2,2× |

2,2× é o preço de não ter framework, e é o preço esperado. Vale notar que boa
parte das 338 linhas é comentário de mecanismo e carregamento do binário — o
forward em si são ~90 linhas de laço.

## As diferenças que têm consequência real

As outras são detalhe de implementação. Estas quatro mudam o que o sistema pode
e não pode fazer.

### 1. Dependências: de 10 pacotes para zero

**Mecanismo.** O lado Python precisa de torch para o modelo, MNE para o DSP e a
leitura de EDF, BrainFlow para a placa. O lado C implementa os três.

**Consequência — e é a consequência central do projeto inteiro.** Os três
bloqueios que impediam embarcar no Pi Zero v1.3 eram todos do **ecossistema
Python**, não do processador:

| bloqueio | por que | em C |
|---|---|---|
| torch não tem wheel para ARMv6 | wheels oficiais são x86_64 e aarch64 | não existe |
| BrainFlow distribui binário nativo pré-compilado | não cobre `armv6l` | não existe |
| 512 MB de RAM com torch + MNE residentes | estouro e swap no cartão SD | 1,04 MB de residência (ver 01) |

Em C os três desaparecem juntos. **Isso reverte o veredito**: o Zero v1.3 deixa
de ser "inviável" e passa a ser "a medir". Essa é a diferença que justifica o
porte; as demais são consequência dela.

### 2. Reamostragem: FFT do MNE → polifásico racional

**Mecanismo no Python.** `raw.resample(128)` dentro de `preproc.cadeia_mne` usa
o caminho por FFT do MNE: transforma, corta ou completa o espectro, transforma de
volta, reescala. Os detalhes internos (`npad='auto'`, o parâmetro `window`, o
fator de escala) não estão especificados na documentação de forma que permita
reproduzi-los por leitura.

**Mecanismo no C.** As duas razões envolvidas são racionais e limpas — EDF do
PhysioNet a 160 Hz dá 128/160 = **4/5**, e a Cyton a 250 Hz dá 128/250 =
**64/125**. Um filtro polifásico faz isso diretamente: o passa-baixas embutido na
estrutura *é* o anti-aliasing, e cada amostra de saída custa
`n_taps / L` MACs (**derivado**: ≈39 para 64/125, ≈25 para 4/5).

**Consequência.** Esta é a diferença que mais mexe no projeto, por um motivo não
óbvio: ela **dissolve o maior risco técnico** do porte.

Se só a inferência fosse portada, o C teria que reproduzir o resampler do MNE
com fidelidade numérica, porque o modelo treinado espera exatamente a
distribuição que aquela cadeia produz. Isso era o risco número um — a peça que
decidia se o porte levava um fim de semana ou duas semanas.

Como **o treino também vai para C**, treino e inferência passam pela mesma cadeia.
Ela não precisa imitar o MNE: precisa ser uma cadeia DSP correta e consistente
consigo mesma. Isso libera escolher o que é bom para o alvo embarcado — e o
polifásico é melhor que o FFT em três frentes: mais barato, *streaming-friendly*
(importante no tempo real), e muito mais simples de acertar.

**O que isso custa.** O critério de aceitação muda de lugar:

| | porte só da inferência | projeto C completo |
|---|---|---|
| critério | `max│C − Python│ < 1e-4` por etapa | **mesma acurácia de validação** |
| risco dominante | internals do `mne.filter.resample` | backprop correto em C |
| liberdade de DSP | nenhuma | escolher o que embarca bem |

E é por isso que o `treinar.c` é obrigatório, não opcional: sem ele, a liberdade
de DSP não existiria.

### 3. Ordem das operações: filtrar antes ou depois de reamostrar

**Mecanismo no Python.** `cadeia_mne` faz notch → passa-banda → reamostragem,
nessa ordem. O passa-banda roda na **taxa original**.

**Derivado** da regra de comprimento de FIR do MNE (6,6 / menor largura de
transição, em segundos, janela de Hamming) com as larguras do `config.py` (0,5 Hz
na borda baixa, 8,75 Hz na alta): o FIR tem ≈**3301 taps a 250 Hz**, aplicado
sobre as 3028 amostras do sinal já espelhado. Ou seja, **o filtro é mais longo que
o sinal** — regime em que o MNE emite aviso. *Isto é derivado por leitura, não
confirmado por execução*, e é um dos itens a conferir.

**Mecanismo no C.** Reamostra primeiro (o passa-baixas do polifásico cobre o
anti-aliasing até 64 Hz, bem acima dos 35 Hz de interesse), e só então aplica o
passa-banda, **a 128 Hz**: ≈1690 taps sobre ≈1550 amostras.

**Consequência.** ≈3,8× menos trabalho de filtragem (**derivado**:
3301×3028 contra 1690×1550), sem perda para a banda de interesse. É um desvio
deliberado da ordem do Python, matematicamente sólido, e vai ser quantificado
pelo arnês e julgado pela acurácia.

### 4. O forward é o mesmo arquivo nos três alvos

**Mecanismo no Python.** `nucleo/modelo.py` define `CNN2D`, usada pelo
`treinar.py` e pelo `tempo_real.py`. Já é compartilhado — mas o *caminho de
execução* difere: treino roda em `train()` com autograd e dropout ativo;
inferência roda em `eval()` sob `no_grad()`.

**Mecanismo no C.** `src/rede.c` é literalmente o mesmo arquivo compilado nos
três alvos (treino no PC, inferência offline, tempo real no Pi). O treino
acrescenta `rede_grad.c` por fora; não existe um segundo forward.

**Consequência.** Não existe caminho de código que só o PC exercita. É isso que
faz "pronto para embarcar" ser uma propriedade verificável e não uma intenção: se
o forward está certo no PC, é o mesmo forward que roda no Pi — e o
`bancada comparar` mede exatamente essa afirmação, em C, sem precisar de Python
no Pi.

## Diferenças menores, mas que mordem

**O BatchNorm.** Em `eval()`, um BN é a afim `y = γ(x−μ)/√(var+ε) + β`. O
exportador colapsa os 4 tensores em 2 — `s = γ/√(var+ε)` e `b = β − s·μ` — e o C
aplica `s·x + b`. **Isto não é dobrar o BN na convolução**: cada BN continua uma
camada própria no `rede.c`. Dobrar economizaria ~3% do tempo (432 canais, ≈227 mil
operações contra 6,4 M das convoluções) e exigiria álgebra entre camadas, com o
bias de um BN atravessando a convolução seguinte. Mecanismo em
`03_mecanismos.md`; a decisão em `04_decisoes_riscos_pendencias.md`.

**A amostra que falta.** No Python, `mne.Epochs` com `tmin=-2.05, tmax=+2.05` a
128 Hz devolve **525** amostras, porque 4,1 × 128 = 524,8 não é inteiro, e
`preproc.ajustar_lote` completa para as 526 do `config.py`. O lado C tem que
reproduzir essa mesma regra de ajuste (mantém o fim, completa o começo por
espelho), senão EDF e gravação chegam à rede com comprimentos diferentes.

**A escala da Cyton.** O `tempo_real.py` multiplica por `1e-6` porque o BrainFlow
entrega microvolts. Lendo a serial direto, o C tem que chegar ao mesmo valor em
Volts a partir das contagens de 24 bits — **derivado**: 4,5 V / ganho 24 /
(2²³−1) ≈ 2,235e-8 V por contagem. **Não verificado** contra o que o BrainFlow
realmente aplica, e é o item que mais me preocupa na aquisição: escala errada faz
o z-score do checkpoint deixar de servir, exatamente o modo de falha que o
docstring de `nucleo/modelo.carregar` adverte.

**`Conv2d` é correlação cruzada.** O PyTorch não inverte o kernel:
`y[t] = Σⱼ w[j]·x[t − pad + j]`. Implementar convolução de verdade dá outro
resultado, e com kernels aprendidos a diferença não é obviamente errada ao olhar
— só erra. Está comentado no ponto exato do `rede.c`.

**`AdaptiveAvgPool` de 131 para 32 não é divisível.** As janelas têm tamanhos
**desiguais**, pela regra `início = ⌊i·131/32⌋`, `fim = ⌈(i+1)·131/32⌉`. Pooling
uniforme dá outro número. Também comentado no ponto exato.

Estes dois últimos são a razão pela qual o teste de paridade existe: são erros que
produzem saída plausível.
