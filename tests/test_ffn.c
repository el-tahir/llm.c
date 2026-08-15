#include "tinyllm.h"
#include "test_common.h"

int main(void) {

    int fails = 0;
    const int T = 5;

    Config config;
    TransformerWeights weights;
    float *data = NULL;
    long file_size = 0;
    read_checkpoint("stories15M.bin", &config, &weights, &data, &file_size);

    RunState s;
    malloc_run_state(&s, &config);

    const int dim = config.dim;
    const int layers[] = {0, 3};
    const int n_sel = sizeof(layers) / sizeof(layers[0]);

    float *xin = load_bin("ref/s5_xin.bin", T * dim);
    float *out = malloc(dim * sizeof(float));

    for (int row = 0; row < T; row++) {
        for (int l = 0; l < n_sel; l++) {
            char ref[64], label[40];
            snprintf(ref, sizeof(ref), "ref/s7_out_l%d_%d.bin", layers[l], row);
            snprintf(label, sizeof(label), "ffn layer %d row==%d", layers[l], row);

            ffn(out, xin + row * dim, &s, &weights, &config, layers[l]);
            float *expected = load_bin(ref, dim);
            fails += compare(label, out, expected, dim, 1e-4f);
            free(expected);
        }
    }

    // the w1/w3 swap
    ffn(out, xin, &s, &weights, &config, 0);
    float *swapped = load_bin("ref/s7_swapped_l0_0.bin", dim);

    float worst = 0.0f;
    for (int i = 0; i < dim; i++) {
        float d = fabsf(out[i] - swapped[i]);
        if (d > worst) { worst = d; }
    }
    free(swapped);

    if (worst < 1e-3f) {
        printf(" FAIL %-22s agrees with the swapped model to %.3g\n",
            "w1/w3 not swapped", worst);
        fails++;
    } else {
        printf(" ok %-22s differs from the swapped model by %.3g\n",
            "w1/w3 not swapped", worst);
    }

    //theres no bias, so a zero input should give zero out, everything zero

    float *zero = calloc(dim, sizeof(float));
    ffn(out, zero, &s, &weights, &config, 0);
    fails += compare("zero in, zero out", out, zero, dim, 0.0f);
    free(zero);

    free(xin);
    free(out);
    free_run_state(&s);
    free(data);

    printf("\n STAGE 7: %s\n", fails ? "FAIL" : "PASS");
    return fails;
}
