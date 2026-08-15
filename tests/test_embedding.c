#include "tinyllm.h"
#include "test_common.h"

int main(void) {
    int fails = 0;
    const int tokens[] = {0, 1, 2534, 31999};
    const int n_tokens = sizeof(tokens) / sizeof(tokens[0]);

    Config config;
    TransformerWeights weights;
    float *data = NULL;
    long file_size = 0;
    read_checkpoint("stories15M.bin", &config, &weights, &data, &file_size);

    RunState s;
    malloc_run_state(&s, &config);

    for (int i = 0; i < n_tokens; i++) {
        int t = tokens[i];
        char path[64], label[32];
        snprintf(path, sizeof(path), "ref/s3_embed_%d.bin", t);
        snprintf(label, sizeof(label), "embed token %d", t);

        float *expected = load_bin(path, config.dim);
        embed_token(s.x, &weights, t, config.dim);
        fails += compare(label, s.x, expected, config.dim, 0.0f);
        free(expected);

    }

    //reference-free, the one-hot claim
    // embedding is one-hot vector @ table
    float onehot[8] = {0};
    const int V = (int)(sizeof(onehot) / sizeof(onehot[0]));
    const int t = 3;
    onehot[t] = 1.0f;

    float *transposed = malloc((size_t)config.dim * V * sizeof(float)); // table^T, (dim, V)
    float *via_matmul = malloc(config.dim * sizeof(float));
    for (int v = 0; v < V; v++) {
        for (int j = 0; j < config.dim; j++) {
            transposed[j * V + v] = weights.token_embedding_table[v * config.dim + j];
        }
    }

    matmul(via_matmul, transposed, onehot, config.dim, V);
    embed_token(s.x, &weights, t, config.dim);
    fails += compare("one-hot == lookup", via_matmul, s.x, config.dim, 0.0f);
    free(transposed);
    free(via_matmul);

    printf("\nSTAGE 3: %s\n", fails ? "FAIL" : "PASS");
    free_run_state(&s);
    free(data);
    return fails;

}
