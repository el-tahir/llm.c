#include <stdio.h>
#include <string.h>

#include "tinyllm.h"

int generate(RunState *s, TransformerWeights *w, Config *p, Tokenizer *t, Sampler *sampler,
             const char *prompt, int steps, int *out, int max_out, int *out_prompt_len,
             TokenCallback on_token, void *ctx) {
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

    for (int pos = 0; pos < steps; pos++) {
        float *logits = forward(s, w, p, out[pos], pos);

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

        if (on_token != NULL) on_token(decode(t, out[pos], next_id), ctx);
    }

    return n;
}
