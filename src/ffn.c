#include <math.h>
#include "tinyllm.h"

static float silu(float v) {
    return v / (1 + expf(-v));
}

void ffn(float *out, float *xin, Transformer *m, int layer) {
    Config *p = &m->config;
    TransformerWeights *w = &m->weights;
    RunState *s = &m->state;
    int dim = p->dim, hidden_dim = p->hidden_dim;

    matmul(s->hb,  layer_slice(w->w1, layer, hidden_dim, dim), xin, hidden_dim, dim);
    matmul(s->hb2, layer_slice(w->w3, layer, hidden_dim, dim), xin, hidden_dim, dim);

    // gate: silu(w1 x) * (w3 x), element-wise
    for (int i = 0; i < hidden_dim; i++) {
        s->hb[i] = silu(s->hb[i]) * s->hb2[i];
    }

    matmul(out, layer_slice(w->w2, layer, dim, hidden_dim), s->hb, dim, hidden_dim);
}
