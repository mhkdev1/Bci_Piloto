# 03 — Mecanismos: como cada peça funciona por dentro

Este documento é sobre **como**, não sobre por quê. As justificativas das
escolhas estão em `04_decisoes_riscos_pendencias.md`.

## 1. O forward, etapa por etapa

`src/rede.c`, função `rede_avaliar`. Entrada: `(C=3, T=526)` já em z-score,
contígua por canal. Saída: `(K=3)` probabilidades do softmax.

```
  entrada (3, 526)
    │
    ├─ 1  banco temporal: 5 ramos Conv2d(1,20,(1,k)), k ∈ {5,11,15,33,65}
    │     cada ramo vê todos os 3 canais, independentemente     → (100, 3, 526)
    ├─ 2  BatchNorm2d(100)
    │
    ├─ 3  Conv2d(100,300,(3,1), groups=100)  filtro espacial    → (300, 526)
    ├─ 4  BatchNorm2d(300)                                   ┐
    ├─ 5  x²                        potência                 │  FUNDIDAS
    ├─ 6  AvgPool2d((1,4))          média móvel              │  num laço
    ├─ 7  log(max(x, 1e-6))                                  ┘  → (300, 131)
    ├─ 8  Dropout(0.25)             identidade em eval
    │
    ├─ 9  Conv2d(300,300,(1,15), groups=300)  separável, tempo → (300, 131)
    ├─ 10 Conv2d(300,32,(1,1))                separável, mistura → (32, 131)
    ├─ 11 BatchNorm2d(32)  +  12 ReLU                        ┐  FUNDIDAS
    ├─ 13 AdaptiveAvgPool2d((1,32))                          ┘  → (32, 32)
    │
    └─ 15 Linear(1024,3) + softmax                            → (3)
```

### A fusão das etapas 3 a 7

É a decisão de implementação com mais efeito prático no arquivo. Escritas
ingenuamente, as etapas 3–7 exigiriam materializar o tensor pós-convolução
espacial — `(300 × 526)` floats = **631 KB** — só para depois reduzi-lo por
pooling a `(300 × 131)` = 157 KB.

O laço do `rede.c` nunca materializa o intermediário. Para cada canal de saída
`o` e cada posição de pooling `p`, ele calcula as 4 amostras do grupo, aplica BN,
eleva ao quadrado, acumula, e só grava o logaritmo da média:

```c
for (p = 0; p < Tp; ++p) {
    float acc = 0.0f;
    for (q = 0; q < P; ++q) {              /* P = 4 */
        const int tt = p * P + q;
        float v = 0.0f;
        for (c = 0; c < C; ++c)            /* conv espacial, 3 taps */
            v += w[c] * ag[c * T + tt];
        v = s * v + b;                     /* BN2 */
        acc += v * v;                      /* potência */
    }
    co[p] = logf(acc / P > 1e-6f ? acc / P : 1e-6f);   /* média + SafeLog */
}
```

Economia **derivada**: 474 KB de residência e uma passada completa de
leitura/escrita sobre 157.800 floats.

### Por que `groups` funciona com aritmética de índice

A convolução espacial é `Conv2d(100, 300, (3,1), groups=100)`: 100 grupos, 1
canal de entrada e 3 de saída por grupo. O mapeamento é
**saída `o` lê entrada `g = o / 3`**, e o `rede.c` percorre na ordem inversa — laço
externo em `g`, interno em `d ∈ {0,1,2}`, com `o = g*depth + d` — para que o
buffer de entrada `ag` seja carregado uma vez por grupo e reutilizado três vezes.

A separável da etapa 9 é `groups=300`: um grupo por canal, saída `o` lê entrada
`o`. Aritmética trivial, mas é onde um `groups` mal interpretado produziria
mistura de canais e saída plausível porém errada.

### Os dois detalhes que produzem saída plausível e errada

**`Conv2d` do PyTorch é correlação cruzada.** O kernel **não** se inverte:

```
y[t] = Σⱼ w[j] · x[t − pad + j]       com pad = k/2
```

Implementar convolução de verdade (`w[k−1−j]`) dá outro resultado. Com kernels
aprendidos, a saída errada não parece errada — ela só é.

**`AdaptiveAvgPool2d` de 131 para 32 não é divisível.** A regra do PyTorch é:

```
início(i) = ⌊ i · 131 / 32 ⌋
fim(i)    = ⌈ (i+1) · 131 / 32 ⌉
```

As janelas têm tamanhos **desiguais** (algumas 4, algumas 5 amostras) e as
vizinhas se **sobrepõem** em uma amostra. Pooling uniforme de 131/32 ≈ 4 dá outro
número. No C:

```c
const int ini = (i * Tp) / A;
const int fim = ((i + 1) * Tp + A - 1) / A;   /* ⌈a/b⌉ com inteiros */
```

**O achatamento.** `ea` é contíguo `[n_saida][A]`, que é exatamente a ordem de
`torch.flatten` sobre `(1, 32, 1, 32)`. A matriz do `Linear` é indexada
`w_lin[classe][o·A + i]`. Trocar a ordem aqui daria uma permutação dos 1024
pesos — e, de novo, saída plausível.

### Softmax

Subtrai o máximo antes de exponenciar, pelo motivo usual (evitar `exp` de
argumento grande). **Medido**: o desvio máximo de `Σp = 1` nas 96 janelas de teste
foi 1,192e-07, um épsilon de float32.

## 2. O colapso dos BatchNorm

### A álgebra

Em `eval()`, um `BatchNorm2d` não é normalização nenhuma — é uma função afim
fixa, por canal, com as estatísticas acumuladas durante o treino:

```
y = γ · (x − μ) / √(var + ε) + β
```

Reagrupando:

```
s = γ / √(var + ε)
b = β − s · μ
y = s · x + b
```

O exportador grava `s` e `b`; o C aplica `s*x + b`. Quatro tensores viram dois.

`ε = 1e-5` (default do PyTorch), lido do módulo e não fixado no código. O cálculo
é feito em **float64** no exportador e só depois convertido para float32, porque
`√(var + ε)` com `var` pequeno é onde a precisão importa.

### Por que NÃO foi dobrado nas convoluções

Dobrar um BN na convolução anterior (multiplicar os pesos por `s` e carregar `b`
como bias) é o que normalmente se faz — e aqui foi recusado deliberadamente.

O ganho seria ~3%: **derivado** de 432 canais de BN × 526 ou 131 amostras
≈ 227 mil operações, contra 6,4 M MACs das convoluções.

O custo seria álgebra **entre** camadas. Concretamente, `bn1` fica entre o banco
temporal e a convolução espacial. Dobrá-lo exige propagar seu bias pela
convolução seguinte:

```
depois do bn1:   z[g] = s₁[g]·conv₁[g] + b₁[g]
conv espacial:   w[o] = Σⱼ W[o,j] · z[g,j]
               = s₁[g] · Σⱼ W[o,j]·conv₁[g,j]  +  b₁[g] · Σⱼ W[o,j]
                 └── escala entra em W[o] ──┘     └── bias por canal de saída ──┘
```

Ou seja: o `s₁` do grupo entra na escala da conv espacial, e o `b₁` vira um bias
por canal de saída ponderado pela soma dos taps — e depois o `bn2` ainda aplica
por cima. É correto, é verificável, e é exatamente o tipo de passagem onde um
índice trocado gera um modelo que funciona *quase*.

3% de ganho não paga essa superfície de erro. Fica anotado como otimização
disponível, com o teste de paridade já no lugar para validá-la caso vire
necessária.

## 3. O formato `modelo.bin`

### Por que um binário plano

Um `.pt` é um pickle: ler exige torch, e torch não existe no alvo. O
`modelo.bin` é um cabeçalho de inteiros seguido de floats em sequência — `fread`
e pronto. **Medido**: 86.172 bytes para 21.525 floats mais 76 bytes de cabeçalho.

Little-endian sem conversão, porque x86_64 e ARM (em configuração Linux normal)
são ambos LE. Anotado no código como premissa explícita.

### O layout

```
char[8]  "BCIC0001"
i32      n_canais, n_classes, amostras
i32      n_por_escala, n_escalas
i32[n_escalas]  kernels
i32      depth, n_saida, ksep, pool, ad_saida, n_flat
f32[n_canais]   norm_mu, norm_sd
f32      w_banco[n_escalas][n_por_escala][k]
f32      bn1_s[F], bn1_b[F]                  F  = n_por_escala · n_escalas
f32      w_esp[Fs][n_canais]                 Fs = F · depth
f32      bn2_s[Fs], bn2_b[Fs]
f32      w_dep[Fs][ksep]
f32      w_pw[n_saida][Fs]
f32      bn3_s[n_saida], bn3_b[n_saida]
f32      w_lin[n_classes][n_flat], b_lin[n_classes]
```

### A geometria vem do arquivo, não do CONFIG.h

Isto é deliberado e espelha o que o `.pt` faz no lado Python. O docstring de
`nucleo.modelo.carregar` nomeia o modo de falha que se evita assim: mudar um
parâmetro meses depois e passar a alimentar o modelo com uma janela diferente da
que ele aprendeu, **sem nenhum erro aparecer**. O `CONFIG.h` só manda nos
parâmetros de aquisição e de montagem de janela; a rede manda na própria
geometria.

### O carregador: um `malloc`, fatiado

`rede_carregar` soma o tamanho de todos os tensores, faz **um** `malloc`, e
distribui ponteiros dentro do bloco com uma macro:

```c
#define FATIA(campo, quantos) do {                          \
    r->campo = r->bloco + usado; usado += (size_t)(quantos); \
} while (0)
```

Depois lê o arquivo inteiro numa única `fread` sobre o bloco. Isso dá três
coisas: uma alocação e uma liberação em vez de catorze; localidade de cache entre
tensores vizinhos; e — o ponto que importa — **a ordem de leitura é, por
construção, a ordem das fatias**, então não existe a possibilidade de o
exportador e o carregador discordarem sobre a sequência.

### Validação na carga

O carregador rejeita, com mensagem específica: magia errada, cabeçalho truncado,
kernel par (quebraria o padding simétrico), dimensões não-positivas,
`n_flat ≠ n_saida · ad_saida`, janela menor que o fator de pooling, e arquivo de
pesos truncado. Um `.bin` sutilmente errado falha alto em vez de inferir errado.

### O exportador lê a geometria dos módulos

`ferramentas/exportar_pesos.py` não chuta nenhum número. Cada um sai do módulo
correspondente do modelo carregado:

| valor | de onde |
|---|---|
| `kernels`, `n_por_escala` | `bloco[0].ramos[i].weight.shape` |
| `pool` | `bloco[5].kernel_size` |
| `ksep` | `bloco[8].weight.shape[3]` |
| `n_canais`, `Fs` | `bloco[2].weight.shape` |
| `n_saida` | `bloco[9].weight.shape[0]` |
| `ad_saida` | `modelo.pool.output_size[1]` |
| `n_flat`, `n_classes` | `cabeca[1].in_features / out_features` |

E afirma o resto: tipos dos módulos, `Fs == F·depth`,
`n_flat == n_saida·ad_saida`, kernels ímpares, convoluções com `bias=None`,
presença de `norm_mu`/`norm_sd`, ausência de NaN/inf, e contagem final de floats.
Se alguém mudar o `nucleo/modelo.py`, o script **quebra alto** em vez de gravar um
binário sutilmente errado.

## 4. O arnês de verificação

### Por que carregar os mesmos pesos isola a rede

`teste/comparar.py` passa as **mesmas 96 janelas** pelo PyTorch (via
`nucleo.modelo.carregar`) e pelo C (via `modelo.bin`), com os pesos do mesmo
`.pt`. Como peso e entrada são idênticos, o único grau de liberdade restante é a
implementação do forward.

Isso é o que dá ao teste poder de atribuição: não existe pré-processamento,
treino ou aquisição no caminho para dividir a culpa. Divergência ⇒ `rede.c`.

A entrada é gerada com `np.random.default_rng(0)`, então o teste é
determinístico e dois executáveis diferentes podem ser comparados mês que vem.

### Os quatro casos de borda

| # | janela | o que exercita |
|---|---|---|
| 0 | tudo zero | potência nula ⇒ piso de 1e-6 do `SafeLog` nos dois lados |
| 1 | tudo um | sinal constante: só os termos de bias respondem |
| 2 | escala 100× | nada estoura em float32 |
| 3 | escala 1e-4 | o `clamp` do `SafeLog` ativa de verdade |

O caso 3 é o que deu o pior `max |dif|` (4,172e-07), o que é o esperado: perto do
piso do log, a derivada é grande e as diferenças de arredondamento se amplificam.

### `bancada comparar`, e por que está em C

`alvos/bancada.c` tem três subcomandos:

```
bancada rede     <modelo.bin> <entrada.bin> <saida.bin>   roda o forward em lote
bancada tempo    <modelo.bin> [repetições]                 ms por janela
bancada comparar <saida_a.bin> <saida_b.bin>               A contra B
```

O `comparar` é implementado em C, não em Python, por um motivo específico: é
assim que se compara **o C do PC com o C do Pi**, e no Pi não existe torch para
rodar o `teste/comparar.py`. Ele é despachado antes de qualquer carregamento de
modelo — não precisa de um.

O fluxo de verificação PC ↔ Pi é então inteiramente livre de Python:

```
PC:   bancada rede modelo.bin paridade_entrada.bin paridade_saida.bin
Pi:   bancada rede modelo.bin paridade_entrada.bin saida_pi.bin
Pi:   bancada comparar paridade_saida.bin saida_pi.bin
```

E o `paridade_saida.bin` do PC já provou bater com o PyTorch, então um "idênticos"
no Pi fecha a corrente **PyTorch → C-no-x86 → C-no-ARMv6** ponta a ponta.

**Medido**: a autocomparação (`comparar A A`) dá `max |dif| = 0.000e+00`, o que
confirma que o comparador não introduz erro próprio.

### Medição de tempo

`bancada tempo` roda uma inferência de aquecimento antes de cronometrar (carrega
páginas e aquece cache), preenche a entrada com ruído determinístico de um LCG
— não zeros, para não ativar atalhos de FPU com denormais — e divide o `clock()`
total pelas repetições.

## 5. O build

```makefile
CC      := gcc
CFLAGS  ?= -std=c99 -O2 -Wall -Wextra -Wpedantic
LDLIBS  ?= -lm
```

`CC := gcc`, não `?=`: o make **já** define `CC` como `cc` nas regras embutidas,
e `?=` só atribui a variáveis indefinidas, então `?=` seria silenciosamente
ignorado. Atribuição direta ainda permite `make CC=...` na linha de comando, que
tem precedência sobre tudo.

A detecção de `.exe` usa `uname -s`, não `$(OS)`:

```makefile
UNAME := $(shell uname -s 2>/dev/null)
ifneq (,$(findstring MINGW,$(UNAME))$(findstring MSYS,$(UNAME))$(findstring CYGWIN,$(UNAME)))
EXE := .exe
```

Porque dentro do shell do MSYS2 o `$(OS)` não vem como `Windows_NT`, e sem
detectar o sufixo o alvo nunca existe para o make — tudo era reconstruído a cada
chamada. **Medido**: com a correção, a segunda chamada de `make` responde
`Nada a ser feito para 'todos'`.

No Raspberry Pi Zero v1.3:

```bash
make CFLAGS="-std=c99 -O2 -Wall -mcpu=arm1176jzf-s -mfpu=vfp"
```

`-mfpu=neon` **não compila** no Zero v1.3: o BCM2835 é ARMv6 e não tem NEON.
