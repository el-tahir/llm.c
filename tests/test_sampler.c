#include "tinyllm.h"
#include "test_common.h"

#define N 64
#define N_COINS 12
#define DRAWS 100000
#define SEED 20240824ULL
#define TOP_P 0.9f

int main(void) {
    int fails = 0;

    const float coins[N_COINS] = {0.0f, 0.05f, 0.17f, 0.31f, 0.42f, 0.5f,
                                     0.63f, 0.755f, 0.86f, 0.93f, 0.977f, 0.995f};
    float *probs = load_bin("ref/s9_probs.bin", N);
    float *trunc = load_bin("ref/s9_trunc.bin", N);

    /* 1. the rng stream  */
    float stream[16];
    unsigned long long state = SEED;
    for (int i = 0; i < 16; i++) { stream[i] = random_f32(&state); }

    float *expected = load_bin("ref/s9_rng.bin", 16);
    fails += compare("rng stream", stream, expected, 16, 0.0f);
    free(expected);

    /* 2. the multinomial walk */
    float got[N_COINS];
    for (int i = 0; i < N_COINS; i++) {
        got[i] = (float) sample_mult(probs, N, coins[i]);
    }

    expected = load_bin("ref/s9_mult.bin", N_COINS);
    fails +=  compare("sample_mult ids", got, expected, N_COINS, 0.0f);
    free(expected);

    /* 3. the same twelve coins through the nuclues */
    ProbIndex *pi = malloc(N * sizeof(ProbIndex));
    if (!pi) {
        fprintf(stderr, "test allocation failed\n");
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < N_COINS; i++) {
        got[i] = (float)sample_top_p(probs, N, TOP_P, pi, coins[i]);
    }

    expected = load_bin("ref/s9_top_p.bin", N_COINS);
    fails += compare("sample_top_p ids", got, expected, N_COINS, 0.0f);
    free(expected);


    /* 4 and 5. one loop, two independent assertions:
     the empirical distribution matches the analytic truncated one,
     and no draw ever lands outside the nucleus */
    int counts[N] = {0};
    int escaped = 0;
    state = SEED;

    for (int d = 0; d < DRAWS; d++) {
        float coin = random_f32(&state);
        int id = sample_top_p(probs, N, TOP_P, pi, coin);
        counts[id]++;
        if (trunc[id] == 0.0f) { escaped++; }
    }

    float freq[N];
    for (int i = 0; i < N; i++) { freq[i] = (float)counts[i] / (float)DRAWS; }
    fails += compare("top-p histogram", freq, trunc, N, 0.01f);

    float esc = (float)escaped, none = 0.0f;
    fails += compare("nothing escapes", &esc, &none, 1, 0.0f);


    /* 6. temperature 0 on the real 32000-vocab model. deterministic,
     so this is an exact match against numpy's argmax */
    const int tokens[] = {1, 306, 3186, 29889, 0, 31999, 450, 6635, 13, 2};
    const int T = sizeof(tokens) / sizeof(tokens[0]);

    Config config;
    TransformerWeights weights;
    float *data = NULL;
    long file_size = 0;
    read_checkpoint("stories15M.bin", &config, &weights, &data, &file_size);

    RunState s;
    malloc_run_state(&s, &config);

    Sampler greedy;
    malloc_sampler(&greedy, config.vocab_size, 0.0f, TOP_P, SEED);

    float ids[10];
    for (int pos = 0; pos < T; pos++) {
        ids[pos] = (float)sample(&greedy, forward(&s, &weights, &config, tokens[pos], pos));
    }
    expected = load_bin("ref/s9_argmax.bin", T);
    fails += compare("greedy ids", ids, expected, T, 0.0f);
    free(expected);

    /* 7. same seed must give the same sequence, and a different seed must not.
    forward() is re-run every time because sample() divides logits by the
    temperature and softmaxes them in place - the buffer is destroyed by use */
    float a[10], b[10], c[10];
    Sampler sa, sb, sc;
    malloc_sampler(&sa, config.vocab_size, 1.0f, TOP_P, SEED);
    malloc_sampler(&sb, config.vocab_size, 1.0f, TOP_P, SEED);
    malloc_sampler(&sc, config.vocab_size, 1.0f, TOP_P, SEED + 1);

    for (int pos = 0; pos < T; pos++) {
        a[pos] = (float)sample(&sa, forward(&s, &weights, &config, tokens[pos], pos));
    }
    for (int pos = 0; pos < T; pos++) {
        b[pos] = (float)sample(&sb, forward(&s, &weights, &config, tokens[pos], pos));
    }
    for (int pos = 0; pos < T; pos++) {
        c[pos] = (float)sample(&sc, forward(&s, &weights, &config, tokens[pos], pos));
    }

    fails += compare("same seed same ids", a, b, T, 0.0f);

    int differs = 0;
    for (int i = 0; i < T; i++) { if (c[i] != a[i]) { differs = 1; } }
    printf(" %s %-22s a different seed gives a different sequence\n",
        differs ? "ok" : "FAIL", "seed matters");
    fails += !differs;

    free(pi);
    free(probs);
    free(trunc);
    free_sampler(&greedy);
    free_sampler(&sa);
    free_sampler(&sb);
    free_sampler(&sc);
    free_run_state(&s);
    free(data);

    printf("\n STAGE 9: %s\n", fails ? "FAILS" : "PASS");
    return fails;

}
