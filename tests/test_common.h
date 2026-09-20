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

static inline unsigned char *load_bytes(const char *path, long *n) {
    FILE *f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "load_bytes: cannot open %s\n", path);
        exit(1);
    }

    fseek(f, 0, SEEK_END);
    long bytes = ftell(f);
    rewind(f);

    unsigned char *buf = malloc(bytes);
    if (!buf) {
        fprintf(stderr, "load_bytes: malloc of %ld bytes failed\n", bytes);
        exit(1);
    }

    if (fread(buf, sizeof(char), bytes, f) != (size_t)bytes) {
        fprintf(stderr, "load_bytes: short read on %s\n", path);
        exit(1);
    }

    fclose(f);

    *n = bytes;
    return buf;
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

// the files are raw int32
_Static_assert(sizeof(int) == 4, "test_common.h: int must be 32 bits");

// read a raw int32 file
static inline int *load_ints(const char *path, int *n) {
    FILE *f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "load_ints: cannot open %s\n", path);
        exit(1);
    }

    fseek(f, 0, SEEK_END);
    long bytes = ftell(f);
    rewind(f);

    if (bytes % sizeof(int) != 0) {
        fprintf(stderr, "load_ints: %s is %ld bytes, not a whole number of int32s\n", path, bytes);
        exit(1);
    }

    int count = bytes / sizeof(int);

    int *buf = malloc(bytes);
    if (!buf) {
        fprintf(stderr, "load_ints: malloc of %ld bytes failed\n", bytes);
        exit(1);
    }

    if (fread(buf, sizeof(int), count, f) != (size_t)count) {
        fprintf(stderr, "load_ints: short read on %s\n", path);
        exit(1);
    }
    fclose(f);

    *n = count;
    return buf;
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
