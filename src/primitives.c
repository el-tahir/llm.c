#include <math.h>

#include "tinyllm.h"

// ------------------------------------------------------------------------------
// neural net blocks: the three primitives

void rmsnorm(float *o, float *x, float *weight, int size) {
    // sum of squares
    float ss = 0.0f;
    for (int j = 0; j < size; j++) {
        ss += x[j] * x[j];
    }

    ss /= size; // mean of squares
    ss += 1e-5f; // eps, inside the sqrt
    ss = 1.0f / sqrtf(ss); // now ss is the reciprocal RMS
    // sqrt() promotes to fp64 silently, we want fp32, same for exp()

    // normalize and scale
    for (int j = 0; j < size; j++) {
        o[j] = weight[j] * (x[j] * ss);
    }
}

void softmax(float *x, int size) {
    // find the max
    float max_val = x[0];
    for (int i = 1; i < size; i++) {
        if (x[i] > max_val) max_val = x[i];
    }

    // exponentiate the shifted values, also accumulate the sum
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }

    // normalize
    for (int i = 0; i < size; i++) {
        x[i] /= sum;
    }
}

void matmul(float *xout, float *x, float *w, int n, int d) {
    // W (d, n) @ x (n, ) -> xout (d, )
    for (int i = 0; i < d; i++) {
        float val = 0.0f;
        for (int j = 0; j < n; j++) {
            val += w[i * n + j] * x[j];
        }
        xout[i] = val;
    }
}
