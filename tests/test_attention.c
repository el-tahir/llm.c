#include <string.h>

#include "tinyllm.h"
#include "test_common.h"

int main(void) {
    int fails = 0;
    const int T = 5; // seqence length from dump
    Config config;
    TransformerWeights weights;
    float *data = NULL;
    long file_size = 0;
    read_checkpoint("stories15M.bin", &config, &weights, &data, &file_size);

    RunState s;
    malloc_run_state(&s, &config);

    const int dim = config.dim;
    const int head_size = dim / config.n_heads;
    const int kv_dim = config.n_kv_heads * head_size;
    const int layers[] = {0, 3};
    const int n_layers = sizeof(layers) / sizeof(layers[0]);

    float *xin = load_bin("ref/s5_xin.bin", T * dim);
    float *out = malloc(head_size * sizeof(float));

    // feed the sequence one position at a time, and match with reference
    for (int pos = 0; pos < T; pos++) {
        for (int l = 0; l < n_layers; l++) {
            char ref[64], label[40];
            snprintf(ref, sizeof(ref), "ref/s5_out_l%d_%d.bin", layers[l], pos);
            snprintf(label, sizeof(label), "layer %d pos=%d", layers[l], pos);

            attention(out, xin + pos * dim, &s, &weights, &config, layers[l], pos);
            float *expected = load_bin(ref, head_size);
            fails += compare(label, out, expected, head_size, 1e-4f);
            free(expected);

        }
    }

    // property 1: at position 0, there is nothing to blend,
    //             softmax of a single element is exactly 1.0,
    //             so the output must be v0 bit for bit

    attention(out, xin, &s, &weights, &config, 0, 0);
    fails += compare("pos 0 == v0", out, s.value_cache, head_size, 0.0f);

    // property 2: the output is a convex combination of the cached values, so every
    //             component must land inside the range of those value spans
    for (int pos = 0; pos < T; pos++) {
        attention(out, xin + pos * dim, &s, &weights, &config, 0, pos);
    }
    int escaped = -1;
    for (int i = 0; i < head_size; i++) {
        float lo = s.value_cache[i], hi = s.value_cache[i];
        for (int t = 1; t < T; t++) {
            float u = s.value_cache[(size_t)t * kv_dim + i];
            if (u < lo) { lo = u; }
            if (u > hi) { hi = u; }
        }
        if (out[i] < lo - 1e-6f || out[i] > hi + 1e-6f) { escaped = i; }

    }
    if (escaped >= 0) {
        printf(" FAIL %-22s component %d left the hull of the cached values\n",
            "convex combination", escaped);
        fails++;
    } else {
        printf(" ok %-22s all %d components inside the value hull\n",
            "convex combination", head_size);
    }

    // history matters, wipe the cache and redo the last position: it must not match
    float *ref4 = load_bin("ref/s5_out_l0_4.bin", head_size);
    size_t cache = (size_t)config.n_layers * config.seq_len * kv_dim;
    memset(s.key_cache, 0, cache * sizeof(float));
    memset(s.value_cache, 0, cache * sizeof(float));

    attention(out, xin + 4 * dim, &s, &weights, &config, 0, 4);
    float worst = 0.0f;
    for (int i = 0; i < head_size; i++) {
        float d = fabsf(out[i] - ref4[i]);
        if (d > worst) { worst = d; }
    }
    if (worst < 1e-3f) {
        printf(" FAIL %-22s pos 4 matched with an empty cache (max|diff| = %.3g)\n",
            "history matters", worst);
        fails++;
    } else {
        printf(" ok %-22s empty cache shifts pos 4 by %.3g\n", "history matters", worst);
    }
    free(ref4);

    free(xin);
    free(out);
    free_run_state(&s);
    free(data);
    printf("\nSTAGE 5: %s\n", fails ? "FAIL" : "PASS");
    return fails;

}
