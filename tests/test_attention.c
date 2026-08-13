#include <string.h>

#include "tinyllm.h"
#include "test_common.h"

int main(void) {

    int fails = 0;

    // --- part 1: stories15M. n_kv_heads == n_heads, so kv_mul == 1 ---
    const int T = 5; // sequence length

    Config config;
    TransformerWeights weights;
    float *data = NULL;
    long file_size = 0;
    read_checkpoint("stories15M.bin", &config, &weights, &data, &file_size);

    RunState s;
    malloc_run_state(&s, &config);

    const int dim = config.dim;
    const int head_size = dim / config.n_heads;
    const int layers[] = {0, 3};
    const int n_sel = sizeof(layers) / sizeof(layers[0]);

    float *xin = load_bin("ref/s5_xin.bin", T * dim);
    float *out = malloc(dim * sizeof(float)); // out is dim wide now, not head_size

    for (int pos = 0; pos < T; pos++) {
        for (int l = 0; l < n_sel; l++) {
            char ref[64], label[40];
            snprintf(ref, sizeof(ref), "ref/s6_out_l%d_%d.bin", layers[l], pos);
            snprintf(label, sizeof(label), "mha layer %d pos=%d", layers[l], pos);

            attention(out, xin + pos * dim, &s, &weights, &config, layers[l], pos);
            float *expected = load_bin(ref, dim);
            fails += compare(label, out, expected, dim, 1e-4f);
            free(expected);

            // head 0's slice of the concatenation is what stage 5 computed
            if (layers[l] == 0) {
                snprintf(ref, sizeof(ref), "ref/s5_out_l0_%d.bin", pos);
                snprintf(label, sizeof(label), "head 0 == stage5 pos=%d", pos);
                expected = load_bin(ref, head_size);
                fails += compare(label, s.xb, expected, head_size, 1e-4f);
                free(expected);
            }
        }
    }

    free(xin);
    free(out);
    free_run_state(&s);
    free(data);

    // ---- part 2: the synthetic model. n_heads = 6, n_kv_heads = 2, kv_mul = 3 ---

    const int GT = 7; // sequence length in the gqa dumps

    Config gc;
    TransformerWeights gw;
    float *gdata = NULL;

    long gsize = 0;
    read_checkpoint("ref/gqa.bin", &gc, &gw, &gdata, &gsize);

    RunState g;
    malloc_run_state(&g, &gc);

    const int gdim = gc.dim;
    const int ghs = gdim / gc.n_heads;
    const int kv_mul = gc.n_heads / gc.n_kv_heads;

    float *gx = load_bin("ref/s6_gqa_x.bin", GT * gdim);
    float *gout = malloc(gdim * sizeof(float));

    for (int pos = 0; pos < GT; pos++) {
        for (int l = 0; l < gc.n_layers; l++) {
            char ref[64], label[40];
            snprintf(ref, sizeof(ref), "ref/s6_gqa_out_l%d_%d.bin", l, pos);
            snprintf(label, sizeof(label), "gqa layer %d pos=%d", l, pos);

            attention(gout, gx + pos * gdim, &g, &gw, &gc, l, pos);
            float *expected = load_bin(ref, gdim);
            fails += compare(label, gout, expected, gdim, 1e-4f);
            free(expected);
        }
    }

    // property 1: at pos 0 thre is nothing to blend, so each query head's output
    // is its kv head's value row verbatim. with kv_mul = 3, heads 0,1,2 must all
    // hold identical copies of kv head 0. exact equality, not a tolerance

    attention(gout, gx, &g, &gw, &gc, 0, 0);
    int bad = -1;
    for (int h = 0; h < gc.n_heads && bad < 0; h++) {
        float *v0 = g.value_cache + (h / kv_mul) * ghs;
        for (int i = 0; i < ghs; i++) {
            if (g.xb[h * ghs + i] != v0[i]) { bad = h; break; }
        }
    }

    if (bad >= 0) {
        printf(" FAIL %-22s head %d is not a copy of kv head %d\n",
            "pos 0 == kv values", bad, bad / kv_mul);
        fails++;
    } else {
        printf("ok %-22s all %d heads copied their kv head exactly\n",
            "pos 0 == kv values", gc.n_heads);
    }

    // property 2: sharing k and v must not collapse a group. heads 0, 1, 2, see the
    // same keys and values but different queries, so they must still differ
    for (int pos = 0; pos < GT; pos++) {
        attention(gout, gx + pos * gdim, &g, &gw, &gc, 0, pos);
    }

    float closest = INFINITY;
    for (int h = 0; h < gc.n_heads; h++) {
        for (int h2 = h + 1; h2 < gc.n_heads; h2++) {
            if (h / kv_mul != h2 / kv_mul) { continue; } // same group only
            float d = 0.0f;
            for (int i = 0; i < ghs; i++) {
                float u = fabsf(g.xb[h * ghs + i] - g.xb[h2 * ghs + i]);
                if (u > d) { d = u; }
            }
            if (d < closest) { closest = d; }
        }
    }

    if (closest < 1e-3f) {
        printf(" FAIL %-22s two heads sharing a kv head agree to %.3g\n",
            "heads stay distinct", closest);
        fails++;
    } else {
        printf(" ok %-22s closest pair within a group differs by %.3g\n",
            "heads stay distinct", closest);
    }

    free(gx);
    free(gout);
    free_run_state(&g);
    free(gdata);

    printf("\n STAGE 6: %s\n", fails ? "FAIL" : "PASS");

    return fails;


}
