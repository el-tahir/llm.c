#include <string.h>

#include "tinyllm.h"
#include "test_common.h"

// rotate copies of q0 and k0 to positions m and n, return their dot product
// accumuate in double
static float rope_dot(const float *q0, const float *k0, int dim,
    int head_size, int m, int n) {
        float *q = malloc(dim * sizeof(float));
        float *k = malloc(dim * sizeof(float));

        memcpy(q, q0, dim * sizeof(float));
        memcpy(k, k0, dim * sizeof(float));
        rope(q, dim, head_size, m);
        rope(k, dim, head_size, n);

        double acc = 0.0;
        for (int i = 0; i < dim; i++) { acc += (double)q[i] * k[i];}
        free(q);
        free(k);
        return (float)acc;
    }


int main(void) {
    int fails = 0;
    const int dim = 288, head_size = 48;
    const int positions[] = {0, 1, 5, 42, 255};
    const int n_pos = sizeof(positions) / sizeof(positions[0]);
    const char *names[] = {"q", "k"};

    for (int p = 0; p < n_pos; p++) {
        for (int v = 0; v < 2; v++) {
            char in[64], ref[64], label[40];
            snprintf(in, sizeof(in), "ref/s4_%s.bin", names[v]);
            snprintf(ref, sizeof(ref), "ref/s4_%s_rope_%d.bin", names[v], positions[p]);
            snprintf(label, sizeof(label), "rope %s pos=%d", names[v], positions[p]);

            float *got = load_bin(in, dim); // fresh copy everytime, rope is in-place
            float *expected = load_bin(ref, dim);
            rope(got, dim, head_size, positions[p]);
            fails += compare(label, got, expected, dim, 1e-4f);
            free(got);
            free(expected);
        }
    }

    /// reference-free property 1: a rotation cannot change a vector's length
    // accumulate in double: fp32 summation over 288 terms costs several ulps,
    // error drifts otherwise
    float *q0 = load_bin("ref/s4_q.bin", dim);
    double acc = 0.0;
    for (int i = 0; i < dim; i++) {acc += (double)q0[i] * q0[i];}
    float n0 = (float)sqrt(acc);
    free(q0);

    for (int p = 0; p < n_pos; p++) {
        float *r = load_bin("ref/s4_q.bin", dim);
        rope(r, dim, head_size, positions[p]);
        acc = 0.0;
        for (int i = 0; i < dim; i++) { acc += (double)r[i] * r[i];}
        float n1 = (float)sqrt(acc);

        char label[40];
        snprintf(label, sizeof(label), "norm kept pos=%d", positions[p]);
        fails += compare(label, &n1, &n0, 1, 1e-5f);
        free(r);
    }

    // reference-free propery 2: the score depends only on the gap between positions
    float *qq = load_bin("ref/s4_q.bin", dim);
    float *kk = load_bin("ref/s4_k.bin", dim);
    float base = rope_dot(qq, kk, dim, head_size, 5, 3); // gap == 2

    const int gap2[][2] = {{2, 0}, {100, 98}, {255, 253}, {254, 252}};
    for (int i = 0; i < (int)(sizeof(gap2) / sizeof(gap2[0])); i++) {
        float d = rope_dot(qq, kk, dim, head_size, gap2[i][0], gap2[i][1]);
        char label[40];
        snprintf(label, sizeof(label), "dot(%d, %d)==dot(5, 3)", gap2[i][0], gap2[i][1]);
        fails += compare(label, &d, &base, 1, 1e-4f);
    }

    // the score must vary with gap, otherwise everything above is vacuous
    float gap3 = rope_dot(qq, kk, dim, head_size, 5, 2);
    if (fabsf(gap3 - base) < 1e-3f) {
        printf(" FAIL %-22s gap 3 scored the same as gap 2 (%.9g vs %.9g)\n",
            "score varies with gap", gap3, base);
        fails++;
    } else {
        printf(" ok %-22s gap3 = %.6g, gap2 = %.6g\n",
            "score varies with gap",gap3, base);
    }
    free(qq);
    free(kk);

    printf("\nSTAGE 4: %s\n", fails ? "FAIL" : "PASS");
    return fails;


}
