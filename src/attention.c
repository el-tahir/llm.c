#include <math.h>
#include "tinyllm.h"

// multi-head causal attention with a kv cache with GQA.
// 'xin' is the (already normalized) attention input for this position: (dim, )
// 'out' receives the block's output, after wo: (dim, )

void attention(float *out, float *xin, RunState *s, TransformerWeights *w, Config *p,
    int layer, int pos) {
        int dim = p->dim;
        int head_size = dim / p->n_heads;
        int kv_dim = p->n_kv_heads * head_size;
        int kv_mul = p->n_heads / p->n_kv_heads; // query heads sharing one kv head

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

        for (int h = 0; h < p->n_heads; h++) {
            float *q = s->q     + h * head_size; // this head's query
            float *att = s->att + h * p->seq_len; // this head's score row
            float *xb = s->xb   + h * head_size; // this head's output slot

            long long kv_offset = layer_offset + (long long)(h / kv_mul) * head_size;

            // score q againt every key from 0 to pos. loop bound -> causal mask
            for (int t = 0; t <= pos; t++) {
                float *kt = s->key_cache + kv_offset + (long long)t * kv_dim;
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
                float *vt = s->value_cache + kv_offset + (long long)t * kv_dim;
                float a = att[t];
                for (int i = 0; i < head_size; i++) {
                    xb[i] += a * vt[i];
                }
            }
        }

        // let the heads combine, and land back in the residual stream's basis
        matmul(out, s->xb, w->wo + (long long)layer * dim * dim, dim, dim);
    }
