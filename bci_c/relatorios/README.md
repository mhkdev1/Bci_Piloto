# Relatórios do projeto C

Documentação do porte do sistema BCI/órtese para C, do estado verificado em
2026-10-15. Quatro documentos, em ordem de leitura:

| documento | o que responde |
|---|---|
| [01 — Estado e verificação](01_estado_e_verificacao.md) | O que existe hoje, como foi verificado, e todos os números medidos |
| [02 — C versus Python](02_c_versus_python.md) | Diferença peça por peça do projeto Python, e a consequência de cada uma |
| [03 — Mecanismos](03_mecanismos.md) | Como cada peça funciona por dentro: forward, colapso dos BN, formato binário, arnês |
| [04 — Decisões, riscos e pendências](04_decisoes_riscos_pendencias.md) | Por que cada escolha, o que ela custa, o que falta e o que pode dar errado |

## Convenção sobre números

Estes documentos distinguem três coisas, sempre explicitamente:

- **Medido** — saiu de uma execução, com o comando reproduzível citado.
- **Derivado** — segue por aritmética de algo medido ou do código lido.
- **Estimado** — é um palpite informado e pode estar errado por um fator.

É a mesma disciplina do `config.py` do projeto Python, que anota ao lado de cada
constante se o valor foi varrido ou escolhido. Um relatório que apaga essa
distinção vira propaganda.

## Resumo em uma linha

O forward da rede em C **reproduz o PyTorch** (`max |dif| = 4,172e-07`, com épsilon
de float32 em 1,19e-07; `argmax` igual em 96/96 janelas), roda em **6,1 ms por
janela** no PC, e não depende de nada além de C99 e `libm`. Pré-processamento,
treino e aquisição ainda **não** estão portados.
