#include <math.h>
#include "tinyllm.h"

static float silu(float v) {
    return v / (1 + expf(-v));
}

void ffn(float *out, float *xin, RunState *s, TransformerWeights *w, Config *p, int layer) {

    long long layer_offset = (long long)layer * p->hidden_dim * p->dim;

    matmul(s->hb,  xin, w->w1 + layer_offset, p->dim, p->hidden_dim);
    matmul(s->hb2, xin, w->w3 + layer_offset, p->dim, p->hidden_dim);

    // apply silu on s->hb, and element-wise multiply hb and hb2
    for (int i = 0; i < p->hidden_dim; i++) {
        s->hb[i] = silu(s->hb[i]);
        s->hb[i] = s->hb[i] * s->hb2[i];
    }

    matmul(out, s->hb, w->w2 + layer_offset, p->hidden_dim, p->dim);
}
