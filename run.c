#include <stdio.h>
#include <stdlib.h>

#include "tinyllm.h"

int main(int argc, char* argv[]) {
    const char *checkpoint_path = argc > 1 ? argv[1] : "stories15M.bin";

    Config config;
    TransformerWeights weights;
    float *data = NULL;
    long file_size = 0;

    read_checkpoint(checkpoint_path, &config, &weights, &data, &file_size);

    printf("loaded %s (%ld bytes)\n", checkpoint_path, file_size);
        printf("  dim        = %d\n", config.dim);
        printf("  hidden_dim = %d\n", config.hidden_dim);
        printf("  n_layers   = %d\n", config.n_layers);
        printf("  n_heads    = %d\n", config.n_heads);
        printf("  n_kv_heads = %d\n", config.n_kv_heads);
        printf("  vocab_size = %d\n", config.vocab_size);
        printf("  seq_len    = %d\n", config.seq_len);
        printf("  head_size  = %d  (derived)\n", config.dim / config.n_heads);
        printf("  classifier = %s\n",
               weights.wcls == weights.token_embedding_table ? "shared" : "separate");

        printf("\ntoken_embedding_table[0..3] = %g %g %g %g\n",
               weights.token_embedding_table[0], weights.token_embedding_table[1],
               weights.token_embedding_table[2], weights.token_embedding_table[3]);
        printf("rms_final_weight[0..3]      = %g %g %g %g\n",
               weights.rms_final_weight[0], weights.rms_final_weight[1],
               weights.rms_final_weight[2], weights.rms_final_weight[3]);

        free(data);
        return 0;
}
