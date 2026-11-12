# 01 — Estado e verificação

Estado em 2026-10-15. Tudo abaixo marcado como **medido** veio de uma execução
nesta máquina, com o comando citado.

## O que existe

| arquivo | linhas | papel | estado |
|---|---:|---|---|
| `src/rede.c` | 338 | passada direta da CNN2D | **verificado** |
| `src/rede.h` | 38 | interface compartilhada pelos três alvos | pronto |
| `CONFIG.h` | 85 | espelho 1:1 do `config.py` | pronto |
| `alvos/bancada.c` | 229 | arnês de comparação e medição | pronto |
| `ferramentas/exportar_pesos.py` | 177 | `.pt` → `modelo.bin` | pronto |
| `teste/comparar.py` | 131 | paridade C ↔ PyTorch | **passando** |
| `Makefile` | 57 | build, 3 alvos, sem dependências | pronto |
| | **1.055** | total |

Contagens **medidas** com `wc -l`.

Não existe ainda: `dsp.c`, `preproc.c`, `edf.c`, `rede_grad.c`, `adam.c`,
`cyton.c`, `alvos/treinar.c`, `alvos/inferir.c`, `alvos/tempo_real.c`.

## Dependências

**Zero externas.** C99 e `libm`. Sem torch, sem MNE, sem BrainFlow, sem BLAS.

O único Python do projeto são duas ferramentas que rodam no PC, uma vez cada:
o exportador de pesos e o arnês de paridade. O Raspberry Pi nunca vê um `.pt`
nem um interpretador Python.

## Toolchain

**Medido.** MSYS2 com `mingw-w64-x86_64-gcc` **16.2.0** e GNU Make **4.4.1**,
instalados nesta máquina durante o trabalho (não havia nenhum compilador C
antes — nem gcc, nem clang, nem MSVC, nem MinGW).

Compila com `-std=c99 -O2 -Wall -Wextra -Wpedantic` e **zero avisos**.

### Peculiaridade do build no Windows

O gcc do mingw-w64 chamado direto do Git Bash falha com:

```
Cannot create temporary file in C:\Windows\: Permission denied
```

Causa: não herda um `TMP` utilizável — o `TMP` do ambiente aponta para o caminho
curto 8.3 `C:\Users\KOVALS~1\...`, que o gcc não resolve, e a API do Windows cai
no default `C:\Windows\`. Exportar `TMP`/`TEMP`/`TMPDIR` à mão **não** resolve, e
`-pipe` também não.

O que resolve: compilar pelo shell de login do MSYS2, que monta `TMP=/tmp`:

```bash
MSYSTEM=MINGW64 CHERE_INVOKING=1 /c/msys64/usr/bin/bash.exe -lc \
  'cd "/d/Inovaçao/Bci_orthosis - teste piloto/bci_c" && make'
```

No Linux e no Raspberry Pi, `make` direto funciona.

## Verificação do forward

### O teste

`teste/comparar.py` carrega **os mesmos pesos** nos dois lados — o
`modelos/cnn2d_transicao.pt` commitado, via `nucleo.modelo.carregar` no lado
PyTorch e via `artefatos/modelo.bin` no lado C — passa 96 janelas idênticas pelos
dois, e compara as probabilidades do softmax.

O desenho importa: como os pesos são os mesmos e a entrada é a mesma, **qualquer
divergência acima do arredondamento está no `rede.c` e em nenhum outro lugar**.
Não há pré-processamento, treino ou aquisição no caminho para confundir a
atribuição de culpa.

Quatro das 96 janelas são casos de borda deliberados:

| # | janela | o que exercita |
|---|---|---|
| 0 | tudo zero | potência nula bate no piso de 1e-6 do `SafeLog` |
| 1 | tudo um | sinal constante: resposta só dos termos de bias |
| 2 | escala 100× | confere que nada estoura em float32 |
| 3 | escala 1e-4 | força o `clamp` do `SafeLog` nos dois lados |

As outras 92 são ruído gaussiano com semente fixa (`default_rng(0)`).

### O resultado

**Medido** — `python bci_c/teste/comparar.py`:

```
janelas .........: 96  (C=3, T=526, K=3)
max |dif| .......: 4.172e-07   (tolerância 1e-04)
média |dif| .....: 5.610e-08
argmax igual ....: 96/96
softmax soma 1 ..: desvio máx 1.192e-07

[0] tudo zero      max|dif| 7.451e-08   argmax py=0 c=0
[1] tudo um        max|dif| 2.384e-07   argmax py=0 c=0
[2] escala 100x    max|dif| 5.960e-08   argmax py=0 c=0
[3] escala 1e-4    max|dif| 4.172e-07   argmax py=0 c=0

PARIDADE OK
```

### Como ler isso

O épsilon do float32 é **1,19e-07**. Um `max |dif|` de 4,17e-07 é portanto da
ordem de três épsilons: **ruído de arredondamento**, não erro de implementação. A
origem é conhecida e não tem remédio — a ordem das somas nas convoluções difere
entre os laços do C e o BLAS do torch, e soma de ponto flutuante não é
associativa.

**Igualdade bit a bit não foi exigida, e não deveria ser.** Ela não existe entre
duas implementações independentes e não seria informativa. O que se exige é que a
diferença seja irrelevante para a decisão: contra o `LIMIAR = 0.65` do
`config.py`, uma discrepância de 1e-7 nunca inverte um acionamento da órtese. A
tolerância de 1e-4 do teste tem três ordens de grandeza de margem sobre o
observado.

O `argmax` igual em 96/96 é o critério operacional de verdade: a classe decidida
é a mesma em toda janela testada, inclusive nas quatro de borda.

## Geometria conferida

O exportador **lê** a geometria dos módulos do modelo carregado, não a chuta. O
que ele encontrou no `.pt` commitado (**medido**):

```
esquema ...: centrado-transicao-v2
geometria .: C=3 T=526 K=3 | 5 escalas x 20 filtros, kernels [5, 11, 15, 33, 65]
           : depth=3 Fs=300 n_saida=32 ksep=15 pool=4 ad=32 n_flat=1024
normaliz. .: mu=[-1.103e-09 -1.304e-09 -1.184e-10]
             sd=[ 4.340e-05  4.509e-05  4.319e-05]
parametros : 21,519 no .pt | 21,525 floats no .bin (+6 = norm_mu e norm_sd)
saida .....: artefatos/modelo.bin  (86,172 bytes)
```

Isso casou exatamente com o que havia sido derivado por leitura do
`nucleo/modelo.py` antes de qualquer execução, o que é uma confirmação
independente de que o `rede.c` foi escrito contra a arquitetura certa.

Os 6 floats de diferença entre `.pt` e `.bin` são `norm_mu` e `norm_sd` (3 canais
cada), que são estatística de normalização e não parâmetros treináveis.

## Custo computacional

**Derivado** por contagem de laços do `rede.c`, em multiplicações-acumulações
por janela:

| etapa | MACs | fração |
|---|---:|---:|
| banco temporal (5 escalas) | 4.071.240 | **64%** |
| conv espacial depthwise | 473.400 | 7% |
| separável depthwise (1,15) | 589.500 | 9% |
| pointwise 300→32 | 1.257.600 | **20%** |
| linear 1024→3 | 3.072 | 0,05% |
| **total** | **≈ 6,40 M** | |

Dois fatos úteis para otimizar depois:

- O **banco temporal é dois terços do custo**. E dentro dele, o ramo de kernel 65
  sozinho é 2.051.400 MACs — **32% do total da rede**. Qualquer trabalho de
  aceleração começa ali (convolução por FFT, ou fatorar o kernel longo).
- O `pointwise` é 20% e tem o pior padrão de acesso do arquivo: percorre
  `dp[i*Tp + p]` com passo `Tp`. É o candidato óbvio a uma transposição de buffer.

Fora dos MACs: 39.300 chamadas a `logf` por janela (uma por amostra pós-pooling,
300 × 131), que em ARMv6 sem unidade transcendental dedicada podem pesar mais que
a aritmética. **Não medido.**

## Residência de memória

**Derivado** dos `malloc` do `rede_carregar`:

| buffer | bytes |
|---|---:|
| `buf_a` — (100 × 3 × 526) | 631.200 |
| `buf_c` — (300 × 131) | 157.200 |
| `buf_d` — (300 × 131) | 157.200 |
| `buf_e` + `buf_t` | 4.620 |
| pesos (21.525 floats) | 86.100 |
| **total** | **≈ 1,04 MB** |

Nos 512 MB do Pi Zero isso é irrelevante — e é o contraste mais direto com o
lado Python, onde torch mais MNE residentes eram um dos três bloqueios para
embarcar.

## Latência

**Medido** — `./artefatos/bancada tempo artefatos/modelo.bin 200`:

| plataforma | forward por janela | taxa |
|---|---:|---:|
| PC (x86_64, gcc 16.2, `-O2`) | **6,095 ms** | 164,1 /s |
| Raspberry Pi Zero v1.3 | **não medido** | — |

6,4 M MACs em 6,095 ms são ≈ 2,1 GFLOP/s efetivos, o que é coerente com `-O2`
auto-vetorizando num núcleo moderno.

### Sobre o número do Pi

**Não foi medido e não há estimativa confiável.** O ARM1176 do Zero v1.3 é ARMv6:
não tem NEON, então a auto-vetorização que o `-O2` usa aqui não existe lá, e a
razão entre as duas máquinas não é um fator estável.

A faixa que eu esperaria para o forward sozinho é **0,12 a 0,25 s** —
**estimado**, e explicitamente mais pessimista que uma análise anterior deste
mesmo trabalho, que havia projetado 0,1–0,3 s para a *cadeia inteira*. Se a
estimativa se confirmar, o forward sozinho consome o orçamento que havia sido
atribuído a tudo, e o `PASSO_S = 0.1` do `config.py` fica inalcançável, com
0,25–0,5 s como faixa realista.

Medir isso é o item de maior valor por minuto de trabalho hoje, e **já é
possível**: ver `04_decisoes_riscos_pendencias.md`, seção "O que já dá para
medir no Pi".

## Como reproduzir tudo

```bash
# 1) compilar  (no Windows, pelo shell do MSYS2 — ver acima)
make

# 2) exportar os pesos do .pt commitado
python ferramentas/exportar_pesos.py

# 3) paridade C vs PyTorch
python teste/comparar.py

# 4) latência do forward
./artefatos/bancada tempo artefatos/modelo.bin 200
```
