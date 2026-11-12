/* CONFIG.h -- espelho de config.py. Mesmos nomes, mesmos valores.
 *
 * Regra deste arquivo: todo numero do projeto mora aqui, como mora no config.py do
 * lado Python. Se os dois discordarem, o treino e a inferencia passam a ver janelas
 * diferentes e nenhum erro aparece -- e o modo de falha classico deste projeto.
 *
 * Os valores que a rede realmente usa NAO vem daqui na hora de inferir: vem do
 * cabecalho do modelo.bin, exatamente como o checkpoint .pt faz no lado Python.
 * Daqui saem so os parametros de aquisicao e de montagem de janela.
 */
#ifndef CONFIG_H
#define CONFIG_H

/* ---- 1) HARDWARE ------------------------------------------------------ */
#define TAXA_PLACA        250.0     /* Cyton no dongle USB: 250 Hz fixo        */
#define N_CANAIS          3         /* C3, C4, Cz -- a ORDEM importa           */
#define CYTON_N_CANAIS    8         /* canais que o pacote da Cyton carrega    */
/* indice de cada canal do CANAIS dentro dos 8 da placa (= MAPA_CANAIS) */
#define MAPA_C3           0
#define MAPA_C4           1
#define MAPA_CZ           2
/* Volts por contagem: 4,5 V / ganho 24 / (2^23 - 1).  Confere com o 1e-6 que o
 * tempo_real.py aplica sobre os microvolts que o BrainFlow entrega. */
#define CYTON_ESCALA_V    (4.5 / 24.0 / 8388607.0)

/* ---- 2) PROTOCOLO ---------------------------------------------------- */
#define DUR_DESCANSO      4.1
#define DUR_TAREFA        4.1
#define PAUSA_ESTABILIZACAO 2

/* ---- 3) PRE-PROCESSAMENTO -------------------------------------------- */
#define TAXA_IA           128.0
#define NOTCH             60.0
#define PASSA_BANDA_BAIXA 0.5
#define PASSA_BANDA_ALTA  35.0
#define PAD_FILTRO_S      4.0       /* espelho nas bordas, recortado depois    */
/* Largura de transicao, regra do MNE: min(max(f*0.25, 2), f) na borda baixa e
 * min(max(f*0.25, 2), nyq-f) na borda alta. Com 0,5 e 35 Hz dao 0,5 e 8,75. O
 * comprimento do FIR sai de 6,6/min(transicao) segundos (janela de Hamming). */
#define TRANS_BAIXA       0.5
#define TRANS_ALTA        8.75
#define FATOR_HAMMING     6.6
#define NOTCH_LARGURA     0.3       /* = NOTCH/200, igual ao default do MNE    */
#define NOTCH_TRANS       1.0

/* ---- 4) JANELA ------------------------------------------------------- */
#define MEIA_JANELA       (DUR_TAREFA / 2.0)
#define JANELA_S          DUR_TAREFA
#define AMOSTRAS          526       /* round(4,1*128)+1, as duas pontas        */
#define N_CLASSES         3
/* 0 = rest, 1 = left, 2 = right */

/* ---- 5) TREINO ------------------------------------------------------- */
#define EPOCAS            35
#define LOTE              16
#define TAXA_APRENDIZADO  1e-3
#define FRACAO_VALIDACAO  0.2
#define PESO_POR_CLASSE   1
#define SEMENTE           40
#define AUGMENT           1
#define AUG_PROB          0.5
#define AUG_RUIDO         0.10
#define AUG_ESCALA_MIN    0.9
#define AUG_ESCALA_MAX    1.1
#define AUG_DESLOCAMENTO  16

/* ---- 6) DECISAO ------------------------------------------------------ */
#define CLASSE_ACIONA     2         /* "right"                                 */
#define LIMIAR            0.65
#define PASSO_S           0.1
#define SEGURA_S          2.0
#define REFRATARIO_S      2.0

/* ---- 8) ARQUITETURA (so para treinar do zero; inferir le do modelo.bin) -- */
#define CNN_N_ESCALAS     5
#define CNN_KERNELS       { 5, 11, 15, 33, 65 }
#define CNN_FILTROS       20
#define CNN_DEPTH         3
#define CNN_SAIDA         32
#define CNN_KSEP          15        /* conv separavel (1,15)                   */
#define CNN_POOL          4         /* AvgPool2d((1,4))                        */
#define CNN_LOG_EPS       1e-6f     /* SafeLog                                 */
#define CNN_BN_EPS        1e-5f     /* default do BatchNorm2d do PyTorch       */

#endif /* CONFIG_H */
