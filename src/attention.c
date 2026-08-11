#include <math.h>
#include "tinyllm.h"

// single-head causal attention with a kv-cache
// 'xin' is the (already normalized) attention input for this position: (dim, )
// 'out' receives head 0's output: (head_size, )

void attention(float *out, float *xin, RunState *s, TransformerWeights *w, Config *p,
    int layer, int pos) {
        int dim = p->dim;
        int head_size = dim / p->n_heads;
        int kv_dim = p->n_kv_heads * head_size;

        // this layer's slice of the caches, then this position's row inside it
        long long layer_offset = (long long)layer * p->seq_len * kv_dim;
        float *k = s->key_cache   + layer_offset + (long long)pos * kv_dim;
        float *v = s->value_cache + layer_offset + (long long)pos * kv_dim;

        // project. k and v land directly in their cache slots - no copy
        matmul(s->q, xin, w->wq + (long long)layer * dim * dim,    dim, dim);
        matmul(k   , xin, w->wk + (long long)layer * dim * kv_dim, dim, kv_dim);
        matmul(v   , xin, w->wv + (long long)layer * dim * kv_dim, dim, kv_dim);

        // position goes into q and k only. v is never dotted with anything
        rope(s->q, dim,    head_size, pos);
        rope(k,    kv_dim, head_size, pos);

        // score q against every key from 0 to pos. loop bound -> causal mask
        for (int t = 0; t <= pos; t++) {
            float *kt = s->key_cache + layer_offset + (long long)t * kv_dim;
            float score = 0.0f;
            for (int i = 0; i < head_size; i++) {
                score += s->q[i] * kt[i];
            }
            s->att[t] = score / sqrtf((float)head_size);
        }

        softmax(s->att, pos + 1);

        // out = weighted sum of the cached values
        for (int i = 0; i < head_size; i++) { out[i] = 0.0f; }
        for (int t = 0; t <= pos; t++) {
            float *vt = s->value_cache + layer_offset + (long long)t * kv_dim;
            float a = s->att[t];
            for (int i = 0; i < head_size; i++) {
                out[i] += a * vt[i];
            }
        }
    }
