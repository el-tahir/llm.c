#ifndef TINYLLM_H
#define TINYLLM_H

/* Transformer model */

typedef struct {
    int dim; // width of the residual stream
    int hidden_dim; // inner width of the FFN
    int n_layers; // number of transformer blocks
    int n_heads; // number of attention heads
    int n_kv_heads; // number of key/value heads
    int vocab_size; // number of distinct tokens
    int seq_len; // max sequence length the model was trained for
} Config;

typedef struct {
    float *token_embedding_table; // (vocab_size, dim)

    //weights for rmsnorms
    float *rms_att_weight; // (n_layers, dim)
    float *rms_ffn_weight; // (n_layers, dim)

    //weights for the attention block
    float *wq; // (n_layers, n_heads * head_size, dim)
    float *wk; // (n_layers, n_kv_heads * head_size, dim)
    float *wv; // (n_layers, n_kv_heads * head_size, dim)
    float *wo; // (n_layers, dim, n_heads * head_size)

    //weights for the ffn
    float *w1; // (n_layers, hidden_dim, dim)
    float *w2; // (n_layers, dim, hidden_dim)
    float *w3; // (n_layers, hidden_dim, dim)

    // final rmsnorm
    float *rms_final_weight; // (dim, )

    // classifier weights for the logits, on the last layer
    float *wcls; // (vocab_size, dim) or aliased to the embedding

} TransformerWeights;

//allocated once and reused for every token
typedef struct {
    float *x; // residual stream : (dim, )
    float *xb; // the head's outputs, concatenated : (dim, )
    float *xb2; // a sublayer's output : (dim, )
    float *q; // query, all heads : (dim, )
    float *att; // attention scores, one row per head : (n_heads, seq_len )

    float *hb; // ffn hidden buffer : (hidden_dim, )
    float *hb2; // ffn gate buffer  : (hidden_dim, )

    // every key and value ever computed, kept for the rest of the sequence.
    // kv_dim = n_kv_heads * head_size
    float *key_cache;   // (n_layers, seq_len, kv_dim)
    float *value_cache; // (n_layers, seq_len, kv_dim)

    float *logits; // (vocab_size, )

} RunState;

/* model.c — loading the checkpoint and allocating scratch space */

float *memory_map_weights(TransformerWeights *w, Config *p, float *ptr, int shared_weights);
void read_checkpoint(const char *path, Config *config, TransformerWeights *weights,
                     float **data, long *file_size);
void malloc_run_state(RunState *s, Config *p);
void free_run_state(RunState *s);

/* primitives.c — the three building blocks */

void rmsnorm(float *o, float *x, float *weight, int size);
void softmax(float *x, int size); // IN PLACE
void matmul(float *xout, float *w, float *x, int d, int n);

/* embedding.c */

void embed_token(float *x, TransformerWeights *w, int token, int dim);

/* rope.c */

void rope(float *v, int n, int head_size, int pos); // IN PLACE

/* attention.c */
// 'out' is (dim, ) - the whole attention block's output, after wo
void attention(float *out, float *xin, RunState *s, TransformerWeights *w,
    Config *p, int layer, int pos);

/* ffn.c */
void ffn(float *out, float *xin, RunState *s, TransformerWeights *w, Config *p, int layer);

/* forward.c */
float *forward(RunState *s, TransformerWeights *w, Config *p, int token, int pos);

#endif
