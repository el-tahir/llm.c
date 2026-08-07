#include <math.h>

#include "tinyllm.h"

// rotary position embedding, in place
// v is n floats -> n / head_size consecutive heads; each head's headsize/2
// adjacent pairs are rotated by pos * theta_i, theta_i = 10000^(-2i/head_size)
void rope(float *v, int n, int head_size, int pos) {
    for (int i = 0; i < n; i += 2) {
        int pair = (i % head_size) / 2; // pair index within head
        float theta = powf(10000.0f, -2.0f * pair / (float)head_size);
        float a = pos * theta;
        float cos = cosf(a), sin = sinf(a);

        float x0 = v[i],  x1 = v[i + 1]; // read before writing
        v[i] =     (x0 * cos) - (x1 * sin);
        v[i + 1] = (x0 * sin) + (x1 * cos);
    }
}
