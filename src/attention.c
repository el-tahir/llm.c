#include <math.h>
#include "tinyllm.h"

// multi-head causal attention with a kv cache with GQA.
// 'xin' is the (already normalized) attention input for this position: (dim, )
// 'out' receives the block's output, after wo: (dim, )

void attention(float *out, float *xin, Transformer *m, int layer, int pos) {
    Config *p = &m->config;
    TransformerWeights *w = &m->weights;
    RunState *s = &m->state;
    int dim = p->dim;
    int head_size = dim / p->n_heads;
    int kv_dim = p->n_kv_heads * head_size;
    int kv_mul = p->n_heads / p->n_kv_heads; // query heads sharing one kv head

    // this layer's slice of the caches, then this position's row inside it
    float *key_cache   = layer_slice(s->key_cache,   layer, p->seq_len, kv_dim);
    float *value_cache = layer_slice(s->value_cache, layer, p->seq_len, kv_dim);
    float *k = key_cache   + (long long)pos * kv_dim;
    float *v = value_cache + (long long)pos * kv_dim;

    // project. k and v land directly in their cache slots - no copy
    matmul(s->q, layer_slice(w->wq, layer, dim,    dim), xin, dim,    dim);
    matmul(k,    layer_slice(w->wk, layer, kv_dim, dim), xin, kv_dim, dim);
    matmul(v,    layer_slice(w->wv, layer, kv_dim, dim), xin, kv_dim, dim);

    // position goes into q and k only. v is never dotted with anything
    rope(s->q, dim,    head_size, pos);
    rope(k,    kv_dim, head_size, pos);

    for (int h = 0; h < p->n_heads; h++) {
        float *q = s->q     + h * head_size; // this head's query
        float *att = s->att + h * p->seq_len; // this head's score row
        float *xb = s->xb   + h * head_size; // this head's output slot

        int kv_offset = (h / kv_mul) * head_size; // this head's kv head, inside a row

        // score q against every key from 0 to pos. loop bound -> causal mask
        for (int t = 0; t <= pos; t++) {
            float *kt = key_cache + (long long)t * kv_dim + kv_offset;
            float score = 0.0f;
            for (int i = 0; i < head_size; i++) {
                score += q[i] * kt[i];
            }
            att[t] = score / sqrtf((float)head_size);
        }

        softmax(att, pos + 1);

        // this head's output = weighted sum of its kv head's cached values
        for (int i = 0; i < head_size; i++) { xb[i] = 0.0f; }
        for (int t = 0; t <= pos; t++) {
            float *vt = value_cache + (long long)t * kv_dim + kv_offset;
            float a = att[t];
            for (int i = 0; i < head_size; i++) {
                xb[i] += a * vt[i];
            }
        }
    }

    // let the heads combine, and land back in the residual stream's basis
    matmul(out, layer_slice(w->wo, layer, dim, dim), s->xb, dim, dim);
}
