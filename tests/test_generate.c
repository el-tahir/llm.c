#include <string.h>

#include "tinyllm.h"
#include "test_common.h"

#define MAXOUT 1024

int main(void) {
    int fails = 0;

    Config config;
    TransformerWeights weights;
    float *data = NULL;
    long file_size = 0;
    read_checkpoint("stories15M.bin", &config, &weights, &data, &file_size);

    RunState state;
    malloc_run_state(&state, &config);

    Tokenizer t;
    malloc_tokenizer(&t, "tokenizer.bin", config.vocab_size);

    Sampler sampler;
    malloc_sampler(&sampler, config.vocab_size, 0.0f, 0.9f, 42ULL);

    int out[MAXOUT];
    int prompt_len = -1;
    int n = generate(&state, &weights, &config, &t, &sampler,
                     "One day, Lily met a", 10000, out, MAXOUT, &prompt_len, NULL, NULL);

    int want_n;
    int *want = load_ints("ref/s12_greedy.bin", &want_n);
    fails += compare_ints("greedy ids", out, n, want, want_n);
    free(want);

    if (n == config.seq_len + 1) {
        printf(" ok %-22s 10000 steps became %d, n = seq_len + 1\n",
            "steps clamp", config.seq_len);
    } else {
        printf(" FAIL %-22s n = %d, want seq_len + 1 = %d\n",
            "steps clamp", n, config.seq_len + 1);
        fails++;
    }

// the out_prompt_len contract: out[0..prompt_len-1] must be the encoded prompt
    int fresh[64];            // "One day, Lily met a" is 19 bytes, so at most 22 ids
    int fresh_n;
    encode(&t, "One day, Lily met a", 1, 0, fresh, &fresh_n);
    int split_ok = (prompt_len == fresh_n) &&
                    (memcmp(out, fresh, (size_t)fresh_n * sizeof(int)) == 0);
    printf(split_ok ? " ok %-22s prompt_len = %d, prefix matches encode\n"
                    : " FAIL %-22s prompt_len = %d, prefix does not match encode\n",
        "prompt split", prompt_len);
    fails += !split_ok;

    // a second run on the same RunState, and an empty prompt: prefill of length 1
    int e_out[64];
    int e_n = generate(&state, &weights, &config, &t, &sampler, "", 12, e_out, 64, NULL,
                       NULL, NULL);
    want = load_ints("ref/s12_empty.bin", &want_n);
    fails += compare_ints("empty prompt", e_out, e_n, want, want_n);
    free(want);

    free_sampler(&sampler);
    free_tokenizer(&t);
    free_run_state(&state);
    free(data);

    printf("\n STAGE 12: %s\n", fails ? "FAILS" : "PASS");
    return fails;

}
