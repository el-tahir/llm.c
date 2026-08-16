#include <stdio.h>
#include <stdlib.h>

#include "tinyllm.h"

float *forward(RunState *s, TransformerWeights *w, Config *p, int token, int pos) {

    embed_token(s->x, w, token, p->dim);

    for (int l = 0; l < p->n_layers; l++) {
        rmsnorm(s->xb, s->x, w->rms_att_weight + (l * p->dim), p->dim);
        attention(s->xb2, s->xb, s, w, p, l, pos);

        for (int i = 0; i < p->dim; i++) { s->x[i] += s->xb2[i]; }

        rmsnorm(s->xb, s->x, w->rms_ffn_weight + (l * p->dim), p->dim);
        ffn(s->xb2, s->xb, s, w, p, l);

        for (int i = 0; i < p->dim; i++) { s->x[i] += s->xb2[i]; }

    }

    rmsnorm(s->x, s->x, w->rms_final_weight, p->dim);

    matmul(s->logits, w->wcls, s->x, p->vocab_size, p->dim);

    return s->logits;
}
