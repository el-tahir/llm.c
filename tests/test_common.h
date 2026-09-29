#ifndef TEST_COMMON_H
#define TEST_COMMON_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "tinyllm.h"

// the files are raw int32
_Static_assert(sizeof(int) == 4, "test_common.h: int must be 32 bits");

// read a whole file into a fresh buffer, its size in *n_bytes.
// dies loudly on any problem
static inline void *read_file(const char *path, long *n_bytes) {
    FILE *f = xfopen(path, "rb");
    fseek(f, 0, SEEK_END);
    long bytes = ftell(f);
    rewind(f);

    void *buf = xmalloc(bytes > 0 ? bytes : 1); // malloc(0) may return NULL
    xfread(buf, 1, bytes, f, path);
    fclose(f);

    *n_bytes = bytes;
    return buf;
}

// read exactly n fp32 values
static inline float *load_bin(const char *path, int n) {
    long bytes;
    float *buf = read_file(path, &bytes);
    long expected = (long)n * sizeof(float);
    if (bytes != expected) {
        die("load_bin: %s is %ld bytes, expected %ld (n=%d)", path, bytes, expected, n);
    }
    return buf;
}

// read raw bytes, their count in *n
static inline unsigned char *load_bytes(const char *path, long *n) {
    return read_file(path, n);
}

// read a raw int32 file, the count in *n
static inline int *load_ints(const char *path, int *n) {
    long bytes;
    int *buf = read_file(path, &bytes);
    if (bytes % sizeof(int) != 0) {
        die("load_ints: %s is %ld bytes, not a whole number of int32s", path, bytes);
    }
    *n = bytes / sizeof(int);
    return buf;
}

// compare n floats elementwise. prints verdict line and the worst offender
// returns 0 on pass, 1 on fail, a test main can sum results and return the total
// label is a name for this particular comparison
static inline int compare(const char *label, const float *actual, const float *expected, int n,
                          float tol) {
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

static inline int compare_bytes(const char *label, const unsigned char *actual,
                                const unsigned char *expected, long n) {
    for (long i = 0; i < n; i++) {
        if (actual[i] != expected[i]) {
            printf(" FAIL %-22s mismatch at index %ld: got 0x%02X, want 0x%02X\n",
                label, i, actual[i], expected[i]);
            return 1;
        }
    }
    printf(" ok %-22s %ld bytes match\n", label, n);
    return 0;
}

// ids are exact, no tolerance
static inline int compare_ints(const char *label, const int *actual, int actual_n,
                               const int *expected, int expected_n) {
    if (actual_n != expected_n) {
        printf(" FAIL %-22s produced %d ids, reference has %d\n", label, actual_n, expected_n);
        return 1;
    }
    for (int i = 0; i < actual_n; i++) {
        if (actual[i] != expected[i]) {
            printf(" FAIL %-22s mismatch at index %d: got %d, want %d\n",
                label, i, actual[i], expected[i]);
            return 1;
        }
    }
    printf(" ok %-22s %d ids match\n", label, actual_n);
    return 0;
}

#endif
