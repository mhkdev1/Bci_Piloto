/* bancada.c -- arnes de comparacao e de medicao.
 *
 *   bancada rede  <modelo.bin> <entrada.bin> <saida.bin>
 *       Le N janelas JA normalizadas, roda o forward, grava as probabilidades.
 *       E o alvo do teste de paridade contra o PyTorch: mesmos pesos, mesma
 *       entrada, as probabilidades tem que bater. Se nao batem, o erro esta no
 *       rede.c e em nenhum outro lugar -- nao ha o que interpretar.
 *
 *   bancada tempo  <modelo.bin> [repeticoes]
 *       Mede ms por janela. Rodar no PC e no Pi da a comparacao de latencia,
 *       e e o numero que decide qual PASSO_S e viavel no embarcado.
 *
 * Formato de entrada.bin:  i32 N, i32 C, i32 T, f32[N][C][T]
 * Formato de saida.bin:    i32 N, i32 K,        f32[N][K]
 */
#include "../src/rede.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static int ler_i32(FILE *f, int *v)
{
    unsigned char b[4];
    if (fread(b, 1, 4, f) != 4) return 0;
    *v = (int)((unsigned)b[0] | ((unsigned)b[1] << 8) |
               ((unsigned)b[2] << 16) | ((unsigned)b[3] << 24));
    return 1;
}

static int gravar_i32(FILE *f, int v)
{
    unsigned char b[4];
    b[0] = (unsigned char)(v & 0xff);
    b[1] = (unsigned char)((v >> 8) & 0xff);
    b[2] = (unsigned char)((v >> 16) & 0xff);
    b[3] = (unsigned char)((v >> 24) & 0xff);
    return fwrite(b, 1, 4, f) == 4;
}

static int cmd_rede(Rede *r, const char *f_ent, const char *f_sai)
{
    FILE *fe = NULL, *fs = NULL;
    float *x = NULL, *prob = NULL;
    int N, C, T, K, i, rc = 1;

    fe = fopen(f_ent, "rb");
    if (!fe) { fprintf(stderr, "nao abri %s\n", f_ent); goto fim; }
    if (!ler_i32(fe, &N) || !ler_i32(fe, &C) || !ler_i32(fe, &T)) {
        fprintf(stderr, "cabecalho de %s truncado\n", f_ent); goto fim;
    }
    if (C != rede_canais(r) || T != rede_amostras(r)) {
        fprintf(stderr, "entrada e (%d,%d) mas o modelo espera (%d,%d)\n",
                C, T, rede_canais(r), rede_amostras(r));
        goto fim;
    }
    K = rede_classes(r);

    x    = (float *)malloc((size_t)C * T * sizeof(float));
    prob = (float *)malloc((size_t)K * sizeof(float));
    if (!x || !prob) { fprintf(stderr, "sem memoria\n"); goto fim; }

    fs = fopen(f_sai, "wb");
    if (!fs) { fprintf(stderr, "nao abri %s para escrita\n", f_sai); goto fim; }
    if (!gravar_i32(fs, N) || !gravar_i32(fs, K)) goto fim;

    for (i = 0; i < N; ++i) {
        if (fread(x, sizeof(float), (size_t)C * T, fe) != (size_t)C * T) {
            fprintf(stderr, "entrada acabou na janela %d de %d\n", i, N); goto fim;
        }
        rede_avaliar(r, x, prob);
        if (fwrite(prob, sizeof(float), (size_t)K, fs) != (size_t)K) goto fim;
    }
    printf("%d janelas avaliadas -> %s\n", N, f_sai);
    rc = 0;

fim:
    if (fe) fclose(fe);
    if (fs) fclose(fs);
    free(x); free(prob);
    return rc;
}

static int cmd_tempo(Rede *r, int reps)
{
    const int C = rede_canais(r), T = rede_amostras(r), K = rede_classes(r);
    float *x = (float *)malloc((size_t)C * T * sizeof(float));
    float *prob = (float *)malloc((size_t)K * sizeof(float));
    clock_t t0;
    double ms;
    int i, j;
    unsigned int semente = 12345u;

    if (!x || !prob) { free(x); free(prob); return 1; }

    /* Ruido deterministico: o tempo da rede nao depende do dado, so da geometria,
     * mas um valor fixo em todas as posicoes poderia ativar atalhos de FPU. */
    for (i = 0; i < C * T; ++i) {
        semente = semente * 1103515245u + 12345u;
        x[i] = (float)((int)((semente >> 16) & 0x7fffu) - 16384) / 16384.0f;
    }

    rede_avaliar(r, x, prob);                   /* aquece cache e paginas */

    t0 = clock();
    for (j = 0; j < reps; ++j) rede_avaliar(r, x, prob);
    ms = 1000.0 * (double)(clock() - t0) / (double)CLOCKS_PER_SEC / (double)reps;

    printf("forward da rede : %.3f ms/janela  (%d repeticoes)\n", ms, reps);
    printf("  geometria     : C=%d T=%d K=%d\n", C, T, K);
    printf("  taxa maxima   : %.1f inferencias/s\n", ms > 0.0 ? 1000.0 / ms : 0.0);
    printf("  prob da ultima: ");
    for (i = 0; i < K; ++i) printf("%s=%.4f ", rede_nome_classe(i), prob[i]);
    printf("\n");

    free(x); free(prob);
    return 0;
}

/* Compara dois saida.bin. Nao carrega modelo e nao precisa de Python nem de
 * torch -- de proposito: e assim que se compara o C do PC com o C do Pi, onde
 * nao existe torch para rodar o teste/comparar.py. */
static int cmd_comparar(const char *fa, const char *fb)
{
    FILE *a = NULL, *b = NULL;
    float *pa = NULL, *pb = NULL;
    int na, ka, nb, kb, i, j, rc = 1;
    double pior = 0.0, soma = 0.0;
    int i_pior = 0, concorda = 0;

    a = fopen(fa, "rb");
    b = fopen(fb, "rb");
    if (!a || !b) { fprintf(stderr, "nao abri os dois arquivos\n"); goto fim; }
    if (!ler_i32(a, &na) || !ler_i32(a, &ka) ||
        !ler_i32(b, &nb) || !ler_i32(b, &kb)) {
        fprintf(stderr, "cabecalho truncado\n"); goto fim;
    }
    if (na != nb || ka != kb) {
        fprintf(stderr, "formas diferentes: (%d,%d) e (%d,%d)\n", na, ka, nb, kb);
        goto fim;
    }

    pa = (float *)malloc((size_t)na * ka * sizeof(float));
    pb = (float *)malloc((size_t)na * ka * sizeof(float));
    if (!pa || !pb) { fprintf(stderr, "sem memoria\n"); goto fim; }
    if (fread(pa, sizeof(float), (size_t)na * ka, a) != (size_t)na * ka ||
        fread(pb, sizeof(float), (size_t)na * ka, b) != (size_t)na * ka) {
        fprintf(stderr, "dados truncados\n"); goto fim;
    }

    for (i = 0; i < na; ++i) {
        int am_a = 0, am_b = 0;
        for (j = 0; j < ka; ++j) {
            const double d = fabs((double)pa[i * ka + j] - (double)pb[i * ka + j]);
            soma += d;
            if (d > pior) { pior = d; i_pior = i; }
            if (pa[i * ka + j] > pa[i * ka + am_a]) am_a = j;
            if (pb[i * ka + j] > pb[i * ka + am_b]) am_b = j;
        }
        if (am_a == am_b) ++concorda;
    }

    printf("========================================================\n");
    printf("  A ...............: %s\n", fa);
    printf("  B ...............: %s\n", fb);
    printf("  janelas .........: %d  (K=%d)\n", na, ka);
    printf("  max |dif| .......: %.3e\n", pior);
    printf("  media |dif| .....: %.3e\n", soma / ((double)na * ka));
    printf("  argmax igual ....: %d/%d\n", concorda, na);
    printf("  pior janela (#%d):\n", i_pior);
    printf("    A : ");
    for (j = 0; j < ka; ++j) printf("%.6f ", pa[i_pior * ka + j]);
    printf("\n    B : ");
    for (j = 0; j < ka; ++j) printf("%.6f ", pb[i_pior * ka + j]);
    printf("\n========================================================\n");
    /* 1e-4 e a mesma tolerancia do teste/comparar.py: contra um LIMIAR de 0,65,
     * uma discrepancia dessa ordem nunca inverte um acionamento. */
    rc = (pior <= 1e-4 && concorda == na) ? 0 : 1;
    printf("%s\n", rc == 0 ? "IDENTICOS (dentro do arredondamento)" : "DIVERGEM");

fim:
    if (a) fclose(a);
    if (b) fclose(b);
    free(pa); free(pb);
    return rc;
}

static void uso(void)
{
    fprintf(stderr,
        "uso:\n"
        "  bancada rede     <modelo.bin> <entrada.bin> <saida.bin>\n"
        "  bancada tempo    <modelo.bin> [repeticoes]\n"
        "  bancada comparar <saida_a.bin> <saida_b.bin>\n");
}

int main(int argc, char **argv)
{
    char erro[256];
    Rede *r;
    int rc;

    if (argc < 3) { uso(); return 2; }

    /* `comparar` nao toca no modelo: despachado antes de carregar qualquer coisa. */
    if (strcmp(argv[1], "comparar") == 0) {
        if (argc != 4) { uso(); return 2; }
        return cmd_comparar(argv[2], argv[3]);
    }

    r = rede_carregar(argv[2], erro, (int)sizeof(erro));
    if (!r) { fprintf(stderr, "modelo: %s\n", erro); return 1; }

    if (strcmp(argv[1], "rede") == 0) {
        if (argc != 5) { uso(); rc = 2; }
        else rc = cmd_rede(r, argv[3], argv[4]);
    } else if (strcmp(argv[1], "tempo") == 0) {
        int reps = (argc >= 4) ? atoi(argv[3]) : 50;
        if (reps < 1) reps = 1;
        rc = cmd_tempo(r, reps);
    } else {
        uso(); rc = 2;
    }

    rede_liberar(r);
    return rc;
}
