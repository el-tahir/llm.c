#include "tinyllm.h"
#include "test_common.h"

int main(void) {
    int fails = 0;
    const int tokens[] = {0, 1, 2534, 31999};
    const int n_tokens = sizeof(tokens) / sizeof(tokens[0]);

    Transformer m;
    load_transformer(&m, "stories15M.bin");

    for (int i = 0; i < n_tokens; i++) {
        int t = tokens[i];
        char path[64], label[32];
        snprintf(path, sizeof(path), "ref/s3_embed_%d.bin", t);
        snprintf(label, sizeof(label), "embed token %d", t);

        float *expected = load_bin(path, m.config.dim);
        embed_token(m.state.x, &m.weights, t, m.config.dim);
        fails += compare(label, m.state.x, expected, m.config.dim, 0.0f);
        free(expected);

    }

    //reference-free, the one-hot claim
    // embedding is one-hot vector @ table
    float onehot[8] = {0};
    const int V = (int)(sizeof(onehot) / sizeof(onehot[0]));
    const int t = 3;
    onehot[t] = 1.0f;

    float *transposed = malloc((size_t)m.config.dim * V * sizeof(float)); // table^T, (dim, V)
    float *via_matmul = malloc(m.config.dim * sizeof(float));
    for (int v = 0; v < V; v++) {
        for (int j = 0; j < m.config.dim; j++) {
            transposed[j * V + v] = m.weights.token_embedding_table[v * m.config.dim + j];
        }
    }

    matmul(via_matmul, transposed, onehot, m.config.dim, V);
    embed_token(m.state.x, &m.weights, t, m.config.dim);
    fails += compare("one-hot == lookup", via_matmul, m.state.x, m.config.dim, 0.0f);
    free(transposed);
    free(via_matmul);

    printf("\nSTAGE 3: %s\n", fails ? "FAIL" : "PASS");
    free_transformer(&m);
    return fails;

}
