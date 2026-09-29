#include <math.h>
#include "tinyllm.h"

static float silu(float v) {
    return v / (1 + expf(-v));
}

void ffn(float *out, float *xin, Transformer *m, int layer) {
    Config *p = &m->config;
    TransformerWeights *w = &m->weights;
    RunState *s = &m->state;
    long long layer_offset = (long long)layer * p->hidden_dim * p->dim;

    matmul(s->hb,  w->w1 + layer_offset, xin, p->hidden_dim, p->dim);
    matmul(s->hb2, w->w3 + layer_offset, xin, p->hidden_dim, p->dim);

    // apply silu on s->hb, and element-wise multiply hb and hb2
    for (int i = 0; i < p->hidden_dim; i++) {
        s->hb[i] = silu(s->hb[i]);
        s->hb[i] = s->hb[i] * s->hb2[i];
    }

    matmul(out, w->w2 + layer_offset, s->hb, p->dim, p->hidden_dim);
}
