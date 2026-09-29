#include <stdio.h>
#include <stdlib.h>

#include "tinyllm.h"

static void malloc_run_state(RunState *s, Config *p) {
    int head_size = p->dim / p->n_heads;
    int kv_dim = p->n_kv_heads * head_size;

    // 80 layers x 128k context would overflow an int here, so widen before multiplying
    size_t cache = (size_t)p->n_layers * p->seq_len * kv_dim;

    s->x = xcalloc(p->dim, sizeof(float));
    s->xb = xcalloc(p->dim, sizeof(float));
    s->xb2 = xcalloc(p->dim, sizeof(float));
    s->q = xcalloc(p->dim, sizeof(float));
    s->att = xcalloc(p->n_heads * p->seq_len, sizeof(float));
    s->hb = xcalloc(p->hidden_dim, sizeof(float));
    s->hb2 = xcalloc(p->hidden_dim, sizeof(float));
    s->key_cache = xcalloc(cache, sizeof(float));
    s->value_cache = xcalloc(cache, sizeof(float));
    s->logits = xcalloc(p->vocab_size, sizeof(float));
}

static void free_run_state(RunState *s) {
    free(s->x);
    free(s->xb);
    free(s->xb2);
    free(s->q);
    free(s->att);
    free(s->hb);
    free(s->hb2);
    free(s->key_cache);
    free(s->value_cache);
    free(s->logits);
}
// walk 'ptr' through the weight blob, recording where each tensor starts
// returns the address one past the end of the last tensor, for the size check
static float *memory_map_weights(TransformerWeights *w, Config *p, float *ptr, int shared_weights) {
    int head_size = p->dim / p->n_heads;
    // the products overflow a 32-bit int on larger models, so widen once here
    unsigned long long n_layers = p->n_layers;

    w->token_embedding_table = ptr;
    ptr += (unsigned long long)p->vocab_size * p->dim;

    w->rms_att_weight = ptr;
    ptr += n_layers * p->dim;
    w->wq = ptr;
    ptr += n_layers * p->dim * (p->n_heads * head_size);
    w->wk = ptr;
    ptr += n_layers * p->dim * (p->n_kv_heads * head_size);
    w->wv = ptr;
    ptr += n_layers * p->dim * (p->n_kv_heads * head_size);
    w->wo = ptr;
    ptr += n_layers * (p->n_heads * head_size) * p->dim;

    w->rms_ffn_weight = ptr;
    ptr += n_layers * p->dim;

    w->w1 = ptr;
    ptr += n_layers * p->dim * p->hidden_dim;
    w->w2 = ptr;
    ptr += n_layers * p->hidden_dim * p->dim;
    w->w3 = ptr;
    ptr += n_layers * p->dim * p->hidden_dim;

    w->rms_final_weight = ptr;
    ptr += p->dim;

    ptr += p->seq_len * head_size / 2; // skip legacy freq_cos
    ptr += p->seq_len * head_size / 2; // skip legacy freq_sin

    if (shared_weights) {
        w->wcls = w->token_embedding_table;
    } else {
        w->wcls = ptr;
        ptr += (unsigned long long)p->vocab_size * p->dim;
    }

    return ptr;
}

// load a checkpoint: header into m->config, weight pointers into m->weights
// m->data receives the malloc'd blob (must outlive every pointer in m->weights)
static void read_checkpoint(Transformer *m, const char *path) {
    Config *config = &m->config;
    FILE *file = xfopen(path, "rb");
    // the 7-int header maps exactly onto Config
    xfread(config, sizeof(Config), 1, file, path);

    // a negative vocab_size == "classifier is NOT shared" flag
    int shared_weights = config->vocab_size > 0;
    config->vocab_size = abs(config->vocab_size);

    fseek(file, 0, SEEK_END);
    long file_size = ftell(file);
    rewind(file);

    m->data = xmalloc(file_size);
    xfread(m->data, 1, file_size, file, path);
    fclose(file);

    // weights begin immediately after the header
    float *weights_ptr = m->data + sizeof(Config) / sizeof(float);
    float *end = memory_map_weights(&m->weights, config, weights_ptr, shared_weights);

    // the check: the walk must land on the last byte of the file exactly
    long consumed = (long)((char*)end - (char*)m->data);
    if (consumed != file_size) {
        die("read_checkpoint: layout mismatch in %s:\n"
            "walked to byte %ld\n"
            "file is  %ld bytes\n"
            "off by %ld bytes (%ld floats)",
            path, consumed, file_size,
            consumed - file_size, (consumed - file_size) / 4);
    }
    fprintf(stderr, "loaded %s: %ld bytes, %d layers, dim=%d, vocab=%d, %s classifier\n",
        path, file_size, config->n_layers, config->dim, config->vocab_size,
        shared_weights ? "shared" : "separate");
}

void load_transformer(Transformer *m, const char *path) {
    read_checkpoint(m, path);
    malloc_run_state(&m->state, &m->config);
}

void free_transformer(Transformer *m) {
    free_run_state(&m->state);
    free(m->data);
}
