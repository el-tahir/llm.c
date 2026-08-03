#include "test_common.h"

int main(void) {
    const int n = 8;

    // the same eight values ref.py dumped, as fp32 here
    // rounding is the same, so this should match exactly

    float expected[8] = {
        -3.0f, -1.5f, 0.0f, 0.1f, 1.0f / 3.0f, 1.5f, 3.14159265f, 1e8f
    };

    float *got = load_bin("ref/stage0.bin", n);

    printf("read %d floats from ref/stage0.bin:\n", n);
    for (int i = 0; i < n; i++) {
        printf("  [%d]  %.9g\n", i, got[i]);
    }
    printf("\n");

    int fails = compare("stage0 round-trip", got, expected, n, 0.0f);

    printf("\nSTAGE 0: %s\n", fails ? "FAIL" : "PASS");
    return fails;

}
