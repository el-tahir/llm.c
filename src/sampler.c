#include <stdlib.h>
#include <stdio.h>

#include "tinyllm.h"


void malloc_sampler(Sampler *s, int vocab_size, float temperature, float top_p,
    unsigned long long rng_seed) {
        s->vocab_size = vocab_size;
        s->temperature = temperature;
        s->top_p = top_p;
        s->rng_state = rng_seed;
        // scratch for top_p, allocated once
        s->prob_index = malloc(vocab_size * sizeof(ProbIndex));

        if (!s->prob_index) {
            fprintf(stderr, "sampler allocation failed\n");
            exit(EXIT_FAILURE);
        }
    }

void free_sampler(Sampler *s) {
    free(s->prob_index);
}

unsigned int random_u32(unsigned long long *state) {
    // xorshift64*
    *state ^= *state >> 12;
    *state ^= *state << 25;
    *state ^= *state >> 27;
    return (*state * 0x2545F4914F6CDD1D) >> 32;
}
float random_f32(unsigned long long *state){  // uniform in [0, 1)
    // float mantissa holds only 24 bit
    return (random_u32(state) >> 8) / 16777216.0f; // 2^24
}

int sample_argmax(float *probabilites, int n) {
    int index = 0;
    float maxi = probabilites[0];

    for (int i = 1; i < n; i++) {
        if (probabilites[i] > maxi) {
            maxi = probabilites[i];
            index = i;
        }
    }

    return index;
}
int sample_mult(float *probabilites, int n, float coin) {
    /* coin is uniform in [0, 1), walk the cumultative sum,
     * stop at the first index whose running total passes it */
    float cumultative = 0.0f;
    for (int i = 0; i < n; i++) {
        cumultative += probabilites[i];
        if (cumultative > coin) {
            return i;
        }
    }

    return n - 1;
}

static int compare_prob_index(const void *a, const void *b) {
    float pa = ((ProbIndex *)a)->prob;
    float pb = ((ProbIndex *)b)->prob;
    if (pa > pb) return -1;
    if (pb > pa) return 1;
    return 0;
}

int sample_top_p(float *probabilites, int n, float top_p, ProbIndex *prob_index, float coin) {
    for (int i = 0; i < n; i++) {
        prob_index[i].index = i;
        prob_index[i].prob = probabilites[i];
    }

    qsort(prob_index, n, sizeof(ProbIndex), compare_prob_index);

    // smallest prefix whose mass exceeds top_p
    float cumulative = 0.0f;
    int last_index = n - 1;

    for (int i = 0; i < n; i++) {
        cumulative += prob_index[i].prob;
        if (cumulative > top_p) {
            last_index = i;
            break;
        }
    }

    // scale the draw by the kept mass instead of renormalizing every probability

    float r = coin * cumulative;
    float cdf = 0.0f;
    for (int i = 0; i <= last_index; i++) {
        cdf += prob_index[i].prob;
        if (cdf > r) {
            return prob_index[i].index;
        }
    }

    return prob_index[last_index].index;
}

int sample(Sampler *s, float *logits) {
    int next;
    if (s->temperature == 0.0f) {
        next = sample_argmax(logits, s->vocab_size);
        return next;
    }

    for (int i = 0; i < s->vocab_size; i++) {
        logits[i] /= s->temperature;
    }

    softmax(logits, s->vocab_size);

    float coin = random_f32(&s->rng_state);
    if (s->top_p <= 0.0f || s->top_p >= 1.0f) {
        next = sample_mult(logits, s->vocab_size, coin);
    } else {
        next = sample_top_p(logits, s->vocab_size, s->top_p, s->prob_index, coin);
    }

    return next;
}
