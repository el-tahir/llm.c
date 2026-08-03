#ifndef TEST_COMMON_H
#define TEST_COMMON_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// read exactly n fp32 values from path into a fresh buffer.
// dies loudly on any problem
static float *load_bin(const char *path, int n) {
    FILE *f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "load_bin: cannot open %s \n", path);
        exit(1);
    }
    fseek(f, 0, SEEK_END);
    long bytes = ftell(f);
    rewind(f);
    long expected = (long)n * sizeof(float);
    if (bytes != expected) {
        fprintf(stderr, "load_bin: %s is %ld bytes, expected %ld (n=%d)\n",
            path, bytes, expected, n);
        exit(1);
    }
    float *buf = malloc(bytes);
    if (!buf) {
        fprintf(stderr, "load_bin: malloc of %ld bytes failed\n", bytes);
        exit(1);
    }
    if (fread(buf, sizeof(float), n, f) != (size_t)n) {
        fprintf(stderr, "load_bin: short read on %s\n", path);
        exit(1);
    }
    fclose(f);
    return buf;
}

// compare n floats elementwise. prints verdict line and the worst offender
// returns 0 on pass, 1 on fail, a test main can sum results and return the total
// label is a name for this particular comparison
static int compare(const char *label, const float *actual, const float *expected, int n, float tol) {
    float max_diff = 0.0f;
    int worst = 0;
    for (int i = 0; i < n; i++) {
        float diff = fabsf(actual[i] - expected[i]);
        if (isnan(diff)) { // have to check for NaN first
            printf(" FAIL %-22s NaN at index %d: got %g, want %g\n",
                label, i, actual[i], expected[i]);
            return 1;
        }
        if (diff > max_diff) {max_diff = diff; worst = i;}
    }
    if (max_diff <= tol) {
        printf(" ok %-22s max|diff| = %.3g (tol %g)\n", label, max_diff, tol);
        return 0;
    }
    printf(" FAIL %-22s max|diff| = %.3g at index %d: got %.9g, want %.9g (tol %g)\n",
        label, max_diff, worst, actual[worst], expected[worst], tol);
    return 1;
}



#endif
