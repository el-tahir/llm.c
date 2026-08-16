#include "tinyllm.h"
#include "test_common.h"

#include <string.h>

int main(void) {

    int fails = 0;

    const int tokens[] = {1, 306, 3186, 29889, 0, 31999, 450, 6635, 13, 2};
    const int T = sizeof(tokens) / sizeof(tokens[0]);

    Config config;
    TransformerWeights weights;
    float *data = NULL;
    long file_size = 0;
    read_checkpoint("stories15M.bin", &config, &weights, &data, &file_size);

    RunState s;
    malloc_run_state(&s, &config);

    const int V = config.vocab_size;

    for (int pos = 0; pos < T; pos++) {
        char ref[64], label[40];
        snprintf(ref, sizeof(ref), "ref/s8_logits_%d.bin", pos);
        snprintf(label, sizeof(label), "logits pos=%d", pos);

        float *logits = forward(&s, &weights, &config, tokens[pos], pos);
        float *expected = load_bin(ref, V);
        fails += compare(label, logits, expected, V, 1e-4f);
        free(expected);
    }

    // non-vacuity: pos = 9 logits must actually depend on the 8 before it

    const int head_size = config.dim / config.n_heads;
    const int kv_dim = config.n_kv_heads * head_size;
    const size_t cache = (size_t)config.n_layers * config.seq_len * kv_dim;

    float *expected = load_bin("ref/s8_logits_9.bin", V);
    memset(s.key_cache,   0, cache * sizeof(float));
    memset(s.value_cache, 0, cache * sizeof(float));
    float *logits = forward(&s, &weights, &config, tokens[T - 1], T - 1);

    float worst = 0.0f;
    for (int i = 0; i < V; i++) {
        float d = fabsf(logits[i] - expected[i]);
        if (d > worst) { worst = d; }
    }
    free(expected);

    if (worst < 1e-3f) {
        printf(" FAIL %-22s history wiped, logits barely moved (%.3g)\n",
            "logits need history", worst);
        fails++;
    } else {
        printf(" ok %-22s wiping the cache moved them by %.3g\n",
            "logits need history", worst);
    }

    // the logit for token t is final residual stream dot product token t embedding
    // wcls == embedding_table
    const int probe[] = {0, 1, 2, 306, 15000, 31999};
    float tie = 0.0f;

    for (int j = 0; j < (int)(sizeof(probe) / sizeof(probe[0])); j++) {
        float *row = weights.token_embedding_table + (long long)probe[j] * config.dim;
        float dot = 0.0f;
        for (int i = 0; i < config.dim; i++) { dot += s.x[i] * row[i]; }
        float d = fabsf(dot - logits[probe[j]]);
        if (d > tie) { tie = d; }
    }
    fails += (tie > 1e-4f);
    printf(" %s %-22s logit == dot(x, embedding row), max %.3g\n",
        tie > 1e-4f ? "FAIL" : "ok", "weight tying", tie);

    free_run_state(&s);
    free(data);

    printf("\n STAGE 8: %s\n", fails ? "FAILS" : "PASS");
    return fails;


}
