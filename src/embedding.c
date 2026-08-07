#include <string.h>

#include "tinyllm.h"

//copy row 'token' of the embedding table into the residual stream
void embed_token(float *x, TransformerWeights *w, int token, int dim) {
    float *row = w->token_embedding_table + (long long)token * dim;
    memcpy(x, row, dim * sizeof(float));
}
