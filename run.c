#include <stdio.h>
#include <stdlib.h>
#include <math.h>

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
    float *wq; // (n_layers, dim, n_heads * head_size)
    float *wk; // (n_layers, dim, n_kv_heads * head_size)
    float *wv; // (n_layers, dim, n_kv_heads * head_size)
    float *wo; // (n_layers, n_heads * head_size, dim)

    //weights for the ffn
    float *w1; // (n_layers, hidden_dim, dim)
    float *w2; // (n_layers, dim, hidden_dim)
    float *w3; // (n_layers, hidden_dim, dim)

    // final rmsnorm
    float *rms_final_weight; // (dim, )

    // classifier weights for the logits, on the last layer
    float *wcls; // (vocab_size, dim) or aliased to the embedding

} TransformerWeights;

// walk 'ptr' through the weight blob, recording where each tensor starts
// returns the address one past the end of the last tensor, for the size check
float *memory_map_weights(TransformerWeights *w, Config *p, float *ptr, int shared_weights) {
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

// load a checkpoint: header into 'config', weight pointers into 'weights'
// 'data' receieves the malloc'd blob (must outlive every pointer in 'weights')
void read_checkpoint(const char *path, Config *config, TransformerWeights *weights,
    float **data, long *file_size) {
        FILE *file = fopen(path, "rb");
        if (!file) {
            fprintf(stderr, "couldn't open %s\n", path);
            exit(EXIT_FAILURE);
        }
        // the 7-int header maps exactly onto Config
        if (fread(config, sizeof(Config), 1, file) != 1) {
            fprintf(stderr, "failed to read header from %s\n", path);
            exit(EXIT_FAILURE);
        }

        // a negative vocab_size == "classifier is NOT shared" flag
        int shared_weights = config->vocab_size > 0;
        config->vocab_size = abs(config->vocab_size);

        fseek(file, 0, SEEK_END);
        *file_size = ftell(file);
        rewind(file);

        *data = malloc(*file_size);
        if (! *data) {
            fprintf(stderr, "malloc of %ld bytes failed", *file_size);
            exit(EXIT_FAILURE);
        }
        if (fread(*data, 1, *file_size, file) != (size_t)*file_size) {
            fprintf(stderr, "failed to read %s\n", path);
            exit(EXIT_FAILURE);
        }
        fclose(file);

        //weights begin immediately after the header
        float *weights_ptr = *data + sizeof(Config) / sizeof(float);
        float *end = memory_map_weights(weights, config, weights_ptr, shared_weights);

        // the check: the walk must maldn on the last byte of the file exactly
        long consumed = (long)((char*)end - (char*)*data);
        if (consumed != *file_size) {
            fprintf(stderr,
                "layout mismatch in %s:\n"
                "walked to byte %ld\n"
                "file is  %ld bytes\n"
                "off by %ld bytes (%ld floats)\n",
                path, consumed, *file_size,
                consumed - *file_size, (consumed - *file_size) / 4);
            exit(EXIT_FAILURE);
        }
        fprintf(stderr, "loaded %s: %ld bytes, %d layers, dim=%d, vocab=%d, %s classifier\n",
            path, *file_size, config->n_layers, config->dim, config->vocab_size,
            shared_weights ? "shared" : "separate");
    }

    // ------------------------------------------------------------------------------
    // neural net blocks: the three primitives
    void rmsnorm(float *o, float *x, float *weight, int size) {
        // sum of squares
        float ss = 0.0f;
        for (int j = 0; j < size; j++) {
            ss += x[j] * x[j];
        }

        ss /= size; // mean of squares
        ss += 1e-5f; // eps, inside the sqrt
        ss = 1.0f / sqrtf(ss); // now ss is the reciprocal RMS
        // sqrt() promotes to fp64 silently, we want fp32, same for exp()

        // normalize and scale
        for (int j = 0; j < size; j++) {
            o[j] = weight[j] * (x[j] * ss);
        }
    }

    void softmax(float *x, int size) {
        // find the max
        float max_val = x[0];
        for (int i = 1; i < size; i++) {
            if (x[i] > max_val) max_val = x[i];
        }

        // exponentiate the shifted values, also accumulate the sum
        float sum = 0.0f;
        for (int i = 0; i < size; i++) {
            x[i] = expf(x[i] - max_val);
            sum += x[i];
        }

        // normalize
        for (int i = 0; i < size; i++) {
            x[i] /= sum;
        }
    }

    void matmul(float *xout, float *x, float *w, int n, int d) {
        // W (d, n) @ x (n, ) -> xout (d, )
        for (int i = 0; i < d; i++) {
            float val = 0.0f;
            for (int j = 0; j < n; j++) {
                val += w[i * n + j] * x[j];
            }
            xout[i] = val;
        }
    }

    #ifndef TESTING

    int main(int argc, char* argv[]) {
        const char *checkpoint_path = argc > 1 ? argv[1] : "stories15M.bin";

        Config config;
        TransformerWeights weights;
        float *data = NULL;
        long file_size = 0;

        read_checkpoint(checkpoint_path, &config, &weights, &data, &file_size);

        printf("loaded %s (%ld bytes)\n", checkpoint_path, file_size);
            printf("  dim        = %d\n", config.dim);
            printf("  hidden_dim = %d\n", config.hidden_dim);
            printf("  n_layers   = %d\n", config.n_layers);
            printf("  n_heads    = %d\n", config.n_heads);
            printf("  n_kv_heads = %d\n", config.n_kv_heads);
            printf("  vocab_size = %d\n", config.vocab_size);
            printf("  seq_len    = %d\n", config.seq_len);
            printf("  head_size  = %d  (derived)\n", config.dim / config.n_heads);
            printf("  classifier = %s\n",
                   weights.wcls == weights.token_embedding_table ? "shared" : "separate");

            printf("\ntoken_embedding_table[0..3] = %g %g %g %g\n",
                   weights.token_embedding_table[0], weights.token_embedding_table[1],
                   weights.token_embedding_table[2], weights.token_embedding_table[3]);
            printf("rms_final_weight[0..3]      = %g %g %g %g\n",
                   weights.rms_final_weight[0], weights.rms_final_weight[1],
                   weights.rms_final_weight[2], weights.rms_final_weight[3]);

            free(data);
            return 0;
    }
    #endif
