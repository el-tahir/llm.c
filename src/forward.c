#include "tinyllm.h"

// the residual connection: x += y
static void add(float *x, const float *y, int n) {
    for (int i = 0; i < n; i++) x[i] += y[i];
}

float *forward(Transformer *m, int token, int pos) {
    Config *p = &m->config;
    TransformerWeights *w = &m->weights;
    RunState *s = &m->state;

    embed_token(s->x, w, token, p->dim);

    for (int l = 0; l < p->n_layers; l++) {
        rmsnorm(s->xb, s->x, w->rms_att_weight + (l * p->dim), p->dim);
        attention(s->xb2, s->xb, m, l, pos);

        add(s->x, s->xb2, p->dim);

        rmsnorm(s->xb, s->x, w->rms_ffn_weight + (l * p->dim), p->dim);
        ffn(s->xb2, s->xb, m, l);

        add(s->x, s->xb2, p->dim);
    }

    rmsnorm(s->x, s->x, w->rms_final_weight, p->dim);

    matmul(s->logits, w->wcls, s->x, p->vocab_size, p->dim);

    return s->logits;
}
