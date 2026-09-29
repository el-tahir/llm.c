#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "tinyllm.h"

static void usage(const char *argv0) {
    fprintf(stderr, "usage: %s [checkpoint] [options]\n", argv0);
    fprintf(stderr, "  -t <float>  temperature, 0 is greedy     (default 1.0)\n");
    fprintf(stderr, "  -p <float>  top-p, in [0, 1]             (default 0.9)\n");
    fprintf(stderr, "  -n <int>    steps, capped at seq_len     (default 256)\n");
    fprintf(stderr, "  -i <text>   prompt                       (default none)\n");
    fprintf(stderr, "  -s <int>    rng seed                     (default: the time)\n");
    exit(EXIT_FAILURE);
}

int main(int argc, char **argv) {
    const char *checkpoint = "stories15M.bin";
    float temperature = 1.0f;
    float top_p = 0.9f;
    int steps = 256;
    const char *prompt = "";
    unsigned long long seed = (unsigned long long)time(NULL);

    int i = 1;
    if (i < argc && argv[i][0] != '-') checkpoint = argv[i++];

    for (; i < argc; i += 2) {
        // every flag is exactly two characters and takes exactly one value
        if (i + 1 >= argc || argv[i][0] != '-' || strlen(argv[i]) != 2) usage(argv[0]);
        char flag = argv[i][1];
        const char *val = argv[i + 1];
        if      (flag == 't') temperature = strtof(val, NULL);
        else if (flag == 'p') top_p = strtof(val, NULL);
        else if (flag == 'n') steps = atoi(val);
        else if (flag == 'i') prompt = val;
        else if (flag == 's') seed = strtoull(val, NULL, 10);
        else usage(argv[0]);
    }

    if (temperature < 0.0f) temperature = 0.0f;
    if (top_p < 0.0f || top_p > 1.0f) top_p = 0.9f;
    if (steps <= 0) steps = 256;
    if (seed == 0) seed = 1;

    Config config;
    TransformerWeights weights;
    float *data = NULL;
    long file_size = 0;
    read_checkpoint(checkpoint, &config, &weights, &data, &file_size);

    RunState state;
    malloc_run_state(&state, &config);

    Tokenizer tokenizer;
    malloc_tokenizer(&tokenizer, "tokenizer.bin", config.vocab_size);

    Sampler sampler;
    malloc_sampler(&sampler, config.vocab_size, temperature, top_p, seed);

    int max_steps = steps < config.seq_len ? steps : config.seq_len;
    int need = (int)strlen(prompt) + 3;
    if (max_steps + 1 > need) need = max_steps + 1;

    int *out = malloc(need * sizeof(int));
    if (!out) {
        fprintf(stderr, "run: malloc of %d ids failed\n", need);
        exit(EXIT_FAILURE);
    }

    generate(&state, &weights, &config, &tokenizer, &sampler, prompt, steps, out, need, NULL);

    free(out);
    free_sampler(&sampler);
    free_tokenizer(&tokenizer);
    free_run_state(&state);
    free(data);
    return 0;
}
