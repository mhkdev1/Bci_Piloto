/* rede.c -- passada direta da CNN2D.
 *
 * A arquitetura reproduz, em convolucoes, o que o FBCSP faz a mao: filtra em
 * varias bandas, projeta no espaco dos eletrodos, mede a potencia e classifica
 * pelo logaritmo dela.
 *
 *   entrada (C=3, T=526) ja em z-score
 *     1  banco temporal: 5 ramos Conv2d(1,20,(1,k)) k in {5,11,15,33,65}  -> (100,3,T)
 *     2  BatchNorm2d(100)
 *     3  Conv2d(100,300,(3,1), groups=100)  -- filtro espacial depthwise   -> (300,T)
 *     4  BatchNorm2d(300)
 *     5  x*x                                -- potencia
 *     6  AvgPool2d((1,4))                   -- media movel                -> (300,131)
 *     7  log(max(x,1e-6))
 *     8  Dropout(0.25)                      -- identidade em eval
 *     9  Conv2d(300,300,(1,15), groups=300) -- separavel, parte temporal
 *    10  Conv2d(300,32,(1,1))               -- separavel, mistura os mapas -> (32,131)
 *    11  BatchNorm2d(32)  +  12 ReLU
 *    13  AdaptiveAvgPool2d((1,32))                                        -> (32,32)
 *    15  Linear(1024,3)  +  softmax
 *
 * As etapas 3..7 estao FUNDIDAS num unico laco: a potencia e consumida pela media
 * movel na mesma passada, o que dispensa um buffer de (300 x 526) floats.
 *
 * DOIS PONTOS QUE QUEBRAM TUDO SILENCIOSAMENTE SE ERRADOS, comentados onde acontecem:
 *   - Conv2d do PyTorch e CORRELACAO cruzada, nao convolucao: o kernel nao se inverte.
 *       y[t] = soma_j w[j] * x[t - pad + j]
 *   - AdaptiveAvgPool de 131 para 32 nao e divisivel. As janelas tem tamanhos
 *     DESIGUAIS, pela regra floor/ceil do PyTorch. Pooling uniforme da outro numero.
 *
 * Os BatchNorm nao sao dobrados nas convolucoes. Em eval cada um e so uma afim de
 * 2 floats por canal (432 canais no total, ~227 mil operacoes contra 6,4 milhoes
 * das convolucoes): dobrar economizaria ~3% e abriria uma classe inteira de bugs de
 * algebra entre camadas. O exportador ja colapsa os 4 tensores de cada BN em dois
 * -- s = gama/sqrt(var+eps) e b = beta - s*mu -- que e o unico ganho que valia.
 *
 * Formato do modelo.bin (little-endian; x86 e ARM sao ambos LE):
 *   char[8]  "BCIC0001"
 *   i32      n_canais, n_classes, amostras
 *   i32      n_por_escala, n_escalas
 *   i32[n_escalas]  kernels
 *   i32      depth, n_saida, ksep, pool, ad_saida, n_flat
 *   f32[n_canais]   norm_mu, norm_sd
 *   f32      w_banco[n_escalas][n_por_escala][k]
 *   f32      bn1_s[F], bn1_b[F]                        F  = n_por_escala*n_escalas
 *   f32      w_esp[Fs][n_canais]                       Fs = F*depth
 *   f32      bn2_s[Fs], bn2_b[Fs]
 *   f32      w_dep[Fs][ksep]
 *   f32      w_pw[n_saida][Fs]
 *   f32      bn3_s[n_saida], bn3_b[n_saida]
 *   f32      w_lin[n_classes][n_flat], b_lin[n_classes]
 */
#include "rede.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct Rede {
    int n_canais, n_classes, amostras;
    int n_por_escala, n_escalas, kernels[REDE_MAX_ESCALAS];
    int depth, n_saida, ksep, pool, ad_saida, n_flat;
    int F, Fs, T_pool;

    float *norm_mu, *norm_sd;
    float *w_banco[REDE_MAX_ESCALAS];
    float *bn1_s, *bn1_b;
    float *w_esp, *bn2_s, *bn2_b;
    float *w_dep, *w_pw, *bn3_s, *bn3_b;
    float *w_lin, *b_lin;

    float *bloco;                   /* um unico malloc para todos os pesos */
    float *buf_a, *buf_c, *buf_d, *buf_e, *buf_t;
};

static const char *NOMES[3] = { "rest", "left", "right" };

const char *rede_nome_classe(int i)
{
    return (i >= 0 && i < 3) ? NOMES[i] : "?";
}

int rede_canais(const Rede *r)   { return r->n_canais; }
int rede_amostras(const Rede *r) { return r->amostras; }
int rede_classes(const Rede *r)  { return r->n_classes; }

/* ------------------------------------------------------------------ leitura */

static int ler_i32(FILE *f, int *v)
{
    unsigned char b[4];
    if (fread(b, 1, 4, f) != 4) return 0;
    *v = (int)((unsigned)b[0] | ((unsigned)b[1] << 8) |
               ((unsigned)b[2] << 16) | ((unsigned)b[3] << 24));
    return 1;
}

#define FALHA(msg) do {                                                   \
    if (erro && n_erro > 0) snprintf(erro, (size_t)n_erro, "%s", (msg));  \
    goto falhou;                                                          \
} while (0)

Rede *rede_carregar(const char *caminho, char *erro, int n_erro)
{
    FILE *f = NULL;
    Rede *r = NULL;
    char magia[8];
    size_t n_pesos, usado = 0;
    int e;

    if (erro && n_erro > 0) erro[0] = '\0';

    f = fopen(caminho, "rb");
    if (!f) FALHA("nao consegui abrir o modelo");

    r = (Rede *)calloc(1, sizeof(*r));
    if (!r) FALHA("sem memoria");

    if (fread(magia, 1, 8, f) != 8 || memcmp(magia, "BCIC0001", 8) != 0)
        FALHA("nao e um modelo.bin BCIC0001 -- rode ferramentas/exportar_pesos.py");

    if (!ler_i32(f, &r->n_canais) || !ler_i32(f, &r->n_classes) ||
        !ler_i32(f, &r->amostras) || !ler_i32(f, &r->n_por_escala) ||
        !ler_i32(f, &r->n_escalas))
        FALHA("cabecalho truncado");

    if (r->n_escalas < 1 || r->n_escalas > REDE_MAX_ESCALAS)
        FALHA("n_escalas fora de faixa");
    for (e = 0; e < r->n_escalas; ++e)
        if (!ler_i32(f, &r->kernels[e]) || r->kernels[e] < 1 || (r->kernels[e] % 2) == 0)
            FALHA("kernel invalido: tem que ser impar, para o padding simetrico");

    if (!ler_i32(f, &r->depth) || !ler_i32(f, &r->n_saida) || !ler_i32(f, &r->ksep) ||
        !ler_i32(f, &r->pool) || !ler_i32(f, &r->ad_saida) || !ler_i32(f, &r->n_flat))
        FALHA("cabecalho truncado");

    if (r->n_canais < 1 || r->n_classes < 1 || r->amostras < 1 ||
        r->pool < 1 || r->ad_saida < 1 || r->ksep < 1 || r->depth < 1 ||
        r->n_saida < 1 || r->n_por_escala < 1)
        FALHA("dimensoes invalidas no cabecalho");

    r->F      = r->n_por_escala * r->n_escalas;
    r->Fs     = r->F * r->depth;
    r->T_pool = r->amostras / r->pool;      /* AvgPool2d (ceil_mode=False) descarta o resto */

    if (r->T_pool < 1) FALHA("janela menor que o fator de pooling");
    if (r->n_flat != r->n_saida * r->ad_saida)
        FALHA("n_flat nao casa com n_saida*ad_saida");

    /* Todos os pesos num unico bloco: um malloc, uma liberacao, boa localidade. */
    n_pesos = (size_t)2 * r->n_canais;                      /* norm_mu, norm_sd */
    for (e = 0; e < r->n_escalas; ++e)
        n_pesos += (size_t)r->n_por_escala * r->kernels[e];
    n_pesos += (size_t)2 * r->F;
    n_pesos += (size_t)r->Fs * r->n_canais;
    n_pesos += (size_t)2 * r->Fs;
    n_pesos += (size_t)r->Fs * r->ksep;
    n_pesos += (size_t)r->n_saida * r->Fs;
    n_pesos += (size_t)2 * r->n_saida;
    n_pesos += (size_t)r->n_classes * r->n_flat + (size_t)r->n_classes;

    r->bloco = (float *)malloc(n_pesos * sizeof(float));
    if (!r->bloco) FALHA("sem memoria para os pesos");

#define FATIA(campo, quantos) do {                          \
    r->campo = r->bloco + usado; usado += (size_t)(quantos); \
} while (0)

    FATIA(norm_mu, r->n_canais);
    FATIA(norm_sd, r->n_canais);
    for (e = 0; e < r->n_escalas; ++e)
        FATIA(w_banco[e], (size_t)r->n_por_escala * r->kernels[e]);
    FATIA(bn1_s, r->F);           FATIA(bn1_b, r->F);
    FATIA(w_esp, (size_t)r->Fs * r->n_canais);
    FATIA(bn2_s, r->Fs);          FATIA(bn2_b, r->Fs);
    FATIA(w_dep, (size_t)r->Fs * r->ksep);
    FATIA(w_pw,  (size_t)r->n_saida * r->Fs);
    FATIA(bn3_s, r->n_saida);     FATIA(bn3_b, r->n_saida);
    FATIA(w_lin, (size_t)r->n_classes * r->n_flat);
    FATIA(b_lin, r->n_classes);
#undef FATIA

    /* A ordem de leitura e exatamente a ordem das fatias acima. */
    if (fread(r->bloco, sizeof(float), n_pesos, f) != n_pesos)
        FALHA("arquivo de pesos truncado");

    /* Rascunhos, dimensionados uma vez. Na geometria padrao: 616 KB no buf_a,
     * 154 KB em cada um de buf_c e buf_d, o resto desprezivel -- com os 84 KB de
     * pesos, ~1,0 MB de residencia total. */
    r->buf_a = (float *)malloc((size_t)r->F * r->n_canais * r->amostras * sizeof(float));
    r->buf_c = (float *)malloc((size_t)r->Fs * r->T_pool * sizeof(float));
    r->buf_d = (float *)malloc((size_t)r->Fs * r->T_pool * sizeof(float));
    r->buf_e = (float *)malloc((size_t)r->n_saida * r->ad_saida * sizeof(float));
    r->buf_t = (float *)malloc((size_t)r->T_pool * sizeof(float));
    if (!r->buf_a || !r->buf_c || !r->buf_d || !r->buf_e || !r->buf_t)
        FALHA("sem memoria para os rascunhos");

    fclose(f);
    return r;

falhou:
    if (f) fclose(f);
    if (r) rede_liberar(r);
    return NULL;
}
#undef FALHA

void rede_liberar(Rede *r)
{
    if (!r) return;
    free(r->bloco);
    free(r->buf_a); free(r->buf_c); free(r->buf_d); free(r->buf_e); free(r->buf_t);
    free(r);
}

void rede_normalizar(const Rede *r, float *x)
{
    int c, t;
    for (c = 0; c < r->n_canais; ++c) {
        const float mu = r->norm_mu[c], sd = r->norm_sd[c];
        float *xc = x + (size_t)c * r->amostras;
        for (t = 0; t < r->amostras; ++t) xc[t] = (xc[t] - mu) / sd;
    }
}

/* ------------------------------------------------------------------ forward */

void rede_avaliar(Rede *r, const float *x, float *prob)
{
    const int C = r->n_canais, T = r->amostras, K = r->n_classes;
    const int F = r->F, Fs = r->Fs, Tp = r->T_pool, A = r->ad_saida;
    const int P = r->pool, ks = r->ksep, km = r->ksep / 2;
    float *a = r->buf_a, *cb = r->buf_c, *dp = r->buf_d;
    float *ea = r->buf_e, *pw = r->buf_t;
    int e, fi, c, t, g, d, o, p, q, i, j, k;
    int ch = 0;

    /* (1) banco temporal + (2) BN1  ->  a[F][C][T] */
    for (e = 0; e < r->n_escalas; ++e) {
        const int kk = r->kernels[e], meio = kk / 2;
        for (fi = 0; fi < r->n_por_escala; ++fi, ++ch) {
            const float *w = r->w_banco[e] + (size_t)fi * kk;
            const float s = r->bn1_s[ch], b = r->bn1_b[ch];
            for (c = 0; c < C; ++c) {
                const float *xi = x + (size_t)c * T;
                float *ao = a + ((size_t)ch * C + c) * T;
                for (t = 0; t < T; ++t) {
                    /* correlacao cruzada (nao convolucao), padding simetrico kk/2 */
                    const int j0 = (t - meio < 0) ? meio - t : 0;
                    const int j1 = (t - meio + kk > T) ? T - t + meio : kk;
                    float acc = 0.0f;
                    for (j = j0; j < j1; ++j) acc += w[j] * xi[t - meio + j];
                    ao[t] = s * acc + b;
                }
            }
        }
    }

    /* (3) espacial depthwise + (4) BN2 + (5) x^2 + (6) avgpool + (7) log,
     * tudo numa passada  ->  cb[Fs][Tp] */
    for (g = 0; g < F; ++g) {
        const float *ag = a + (size_t)g * C * T;
        for (d = 0; d < r->depth; ++d) {
            const float *w;
            float s, b, *co;
            o = g * r->depth + d;       /* groups=F: a saida o le a entrada g */
            w  = r->w_esp + (size_t)o * C;
            s  = r->bn2_s[o];
            b  = r->bn2_b[o];
            co = cb + (size_t)o * Tp;
            for (p = 0; p < Tp; ++p) {
                float acc = 0.0f;
                for (q = 0; q < P; ++q) {
                    const int tt = p * P + q;
                    float v = 0.0f;
                    for (c = 0; c < C; ++c) v += w[c] * ag[(size_t)c * T + tt];
                    v = s * v + b;
                    acc += v * v;                               /* potencia */
                }
                acc /= (float)P;
                co[p] = logf(acc > 1e-6f ? acc : 1e-6f);        /* SafeLog, eps=1e-6 */
            }
        }
    }

    /* (9) separavel depthwise (1,ksep), padding ksep/2  ->  dp[Fs][Tp] */
    for (o = 0; o < Fs; ++o) {
        const float *w = r->w_dep + (size_t)o * ks;
        const float *ci = cb + (size_t)o * Tp;
        float *to = dp + (size_t)o * Tp;
        for (p = 0; p < Tp; ++p) {
            const int j0 = (p - km < 0) ? km - p : 0;
            const int j1 = (p - km + ks > Tp) ? Tp - p + km : ks;
            float acc = 0.0f;
            for (j = j0; j < j1; ++j) acc += w[j] * ci[p - km + j];
            to[p] = acc;
        }
    }

    /* (10) pointwise + (11) BN3 + (12) ReLU + (13) adaptive avgpool
     *   ->  ea[n_saida][A] */
    for (o = 0; o < r->n_saida; ++o) {
        const float *w = r->w_pw + (size_t)o * Fs;
        const float s = r->bn3_s[o], b = r->bn3_b[o];
        float *eo = ea + (size_t)o * A;
        for (p = 0; p < Tp; ++p) {
            float acc = 0.0f;
            for (i = 0; i < Fs; ++i) acc += w[i] * dp[(size_t)i * Tp + p];
            acc = s * acc + b;                      /* pointwise e bias=False */
            pw[p] = acc > 0.0f ? acc : 0.0f;        /* ReLU */
        }
        /* AdaptiveAvgPool: inicio = floor(i*Tp/A), fim = ceil((i+1)*Tp/A).
         * De 131 para 32 as janelas NAO tem o mesmo tamanho. */
        for (i = 0; i < A; ++i) {
            const int ini = (i * Tp) / A;
            const int fim = ((i + 1) * Tp + A - 1) / A;
            float acc = 0.0f;
            for (p = ini; p < fim; ++p) acc += pw[p];
            eo[i] = acc / (float)(fim - ini);
        }
    }

    /* (15) linear + softmax. ea e contiguo [n_saida][A], que e a ordem do
     * torch.flatten sobre (1, n_saida, 1, A). */
    {
        float mx = -1e30f, soma = 0.0f;
        for (k = 0; k < K; ++k) {
            const float *w = r->w_lin + (size_t)k * r->n_flat;
            float acc = r->b_lin[k];
            for (i = 0; i < r->n_flat; ++i) acc += w[i] * ea[i];
            prob[k] = acc;
            if (acc > mx) mx = acc;
        }
        for (k = 0; k < K; ++k) { prob[k] = expf(prob[k] - mx); soma += prob[k]; }
        for (k = 0; k < K; ++k) prob[k] /= soma;
    }
}
