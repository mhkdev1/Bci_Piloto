/* rede.h -- a CNN2D, passada direta.
 *
 * ESTE ARQUIVO E COMPARTILHADO pelos tres alvos: treinar (PC), inferir (PC) e
 * tempo_real (PC e Raspberry Pi). Nao existe caminho de codigo que so o PC
 * exercite -- e isso que faz "pronto para embarcar" ser verdade e nao intencao.
 *
 * A geometria NAO vem do CONFIG.h: vem do cabecalho do modelo.bin, como o .pt faz
 * no lado Python. Assim um modelo treinado com outra configuracao nunca e
 * alimentado com uma janela diferente da que aprendeu.
 */
#ifndef REDE_H
#define REDE_H

#define REDE_MAX_ESCALAS 8

typedef struct Rede Rede;

/* Carrega modelo.bin. Devolve NULL e preenche `erro` (se nao-NULL) na falha. */
Rede *rede_carregar(const char *caminho, char *erro, int n_erro);
void  rede_liberar(Rede *r);

int rede_canais(const Rede *r);
int rede_amostras(const Rede *r);
int rede_classes(const Rede *r);

/* Aplica o z-score gravado no checkpoint sobre uma janela (canais x amostras),
 * no lugar. E a mesma estatistica do treino -- sem isso o modelo responde a
 * escala em vez de responder a padrao. */
void rede_normalizar(const Rede *r, float *x);

/* x: (canais x amostras) ja normalizado, contiguo por canal.
 * prob: (classes), softmax. */
void rede_avaliar(Rede *r, const float *x, float *prob);

/* Nome legivel da classe de maior probabilidade. */
const char *rede_nome_classe(int i);

#endif /* REDE_H */
