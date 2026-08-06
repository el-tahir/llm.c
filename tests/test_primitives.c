#define TESTING
#include "../run.c"
#include "test_common.h"

int main(void) {
    int fails = 0;
    const int n = 288, d = 768, m = 256;

    // --- rmsnorm ---
    float *rx = load_bin("ref/s2_rms_x.bin", n);
    float *rw = load_bin("ref/s2_rms_weight.bin", n);
    float *rexp = load_bin("ref/s2_rms_out.bin", n);
    float *rgot = malloc(n * sizeof(float));
    rmsnorm(rgot, rx, rw, n);
    fails += compare("rmsnorm", rgot, rexp, n, 1e-4f);
    free(rx);
    free(rw);
    free(rexp);
    free(rgot);

    // -- softmax
    float *sexp = load_bin("ref/s2_soft_out.bin", m);
    float *sgot = load_bin("ref/s2_soft_x.bin", m);
    softmax(sgot, m);
    fails += compare("softmax", sgot, sexp, m, 1e-4f);

    // -- property 1: the output is a probability distribution ---
    float sum = 0.0f;
    for (int i = 0; i < m; i++) sum += sgot[i];
    float one = 1.0f;
    fails += compare("softmax sums to 1", &sum, &one, 1, 1e-6f);

    // -- property 2 : shfit invariance --
    float * shifted = load_bin("ref/s2_soft_x.bin", m);
    for (int i = 0; i < m; i++) shifted[i] += 1000.0f;
    softmax(shifted, m);
    fails += compare("softmax shift-invar", shifted, sgot, m, 1e-4f);
    free(sexp);
    free(sgot);
    free(shifted);

    // -- matmaul
    float *mx = load_bin("ref/s2_matmul_x.bin", n);
    float *mw = load_bin("ref/s2_matmul_w.bin", d * n);
    float *mexp = load_bin("ref/s2_matmul_out.bin", d);
    float *mgot = malloc(d * sizeof(float));
    matmul(mgot, mx, mw, n, d);
    fails += compare("matmul", mgot, mexp, d, 1e-4f);
    free(mx);
    free(mw);
    free(mexp);
    free(mgot);

    printf("\nSTAGE 2: %s\n", fails ? "FAIL" : "PASS");
    return fails;
}
