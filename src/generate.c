#define _POSIX_C_SOURCE 199309L

#include <stdio.h>
#include <string.h>
#include <time.h>

#include "tinyllm.h"

int generate(RunState *s, TransformerWeights *w, Config *p, Tokenizer *t, Sampler *sampler,
             const char *prompt, int steps, int *out, int max_out, int *out_prompt_len) {
    if (prompt == NULL) prompt = "";

    if (steps > p->seq_len) {
        fprintf(stderr, "generate: steps clamped to seq_len (%d)\n", p->seq_len);
        steps = p->seq_len;
    }

    // worst case encode: BOS + dummy space + one id per byte + EOS
    if ((int)strlen(prompt) + 3 > max_out) {
        fprintf(stderr, "generate: prompt needs %zu ids, out holds %d\n",
            strlen(prompt) + 3, max_out);
        return 0;
    }

    int prompt_len;
    encode(t, prompt, 1, 0, out, &prompt_len);
    int n = prompt_len;

    if (out_prompt_len != NULL) *out_prompt_len = prompt_len;

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    int ran = 0;

    for (int pos = 0; pos < steps; pos++) {
        float *logits = forward(s, w, p, out[pos], pos);
        ran++;

        int next_id;
        if (pos < prompt_len - 1) {
            next_id = out[pos + 1];
        } else {
            next_id = sample(sampler, logits);
        }

        if (next_id == TOKEN_BOS) break;
        if (pos + 1 >= max_out) break;

        out[pos + 1] = next_id;
        if (pos + 1 >= prompt_len) n = pos + 2;

        safe_printf(decode(t, out[pos], next_id));
        fflush(stdout);
    }

    clock_gettime(CLOCK_MONOTONIC, &t1);
    double secs = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) / 1e9;
    if (secs > 0.0) {
        fprintf(stderr, "\n%d forward passes in %.2f s -> %.1f toks/s\n",
            ran, secs, ran / secs);
    }

    return n;
}
