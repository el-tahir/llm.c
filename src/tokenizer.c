#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>

#include "tinyllm.h"


static int hex_digit(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    fprintf(stderr, "hex_digit: invalid hex character '%c'\n", c);
    exit(1);
}

void malloc_tokenizer(Tokenizer *t, const char *path, int vocab_size) {
    t->vocab_size = vocab_size;

    FILE* f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "malloc_tokenizer: cannot open file %s\n", path);
        exit(1);
    }

    if (fread(&t->max_token_length, sizeof(unsigned int), 1, f) != 1) {
        fprintf(stderr, "malloc_tokenizer: failed to read header from %s\n", path);
        exit(1);
    }

    t->vocab = malloc(vocab_size * sizeof(*t->vocab));
    if (!t->vocab) {
        fprintf(stderr, "malloc_tokenizer: malloc of %ld bytes failed\n", vocab_size * sizeof(*t->vocab));
        exit(1);
    }
    t->vocab_scores = malloc(vocab_size * sizeof(*t->vocab_scores));
    if (!t->vocab_scores) {
        fprintf(stderr, "malloc_tokenizer: malloc of %ld bytes failed\n", vocab_size * sizeof(*t->vocab_scores));
        exit(1);
    }

    for (int i = 0; i < vocab_size; i++) {
        if (fread(&t->vocab_scores[i], sizeof(float), 1, f) != 1) {
            fprintf(stderr, "malloc_tokenizer: failed to read score for token %d in %s\n", i, path);
            exit(1);
        }
        int len;
        if (fread(&len, sizeof(int), 1, f) != 1) {
            fprintf(stderr, "malloc_tokenizer: failed to read length for token %d in %s\n", i, path);
            exit(1);
        }

        if (len < 0) {
            fprintf(stderr, "malloc_tokenizer: invalid negative length %d for token %d in %s\n", len, i, path);
            exit(1);
        }

        char *token = malloc(len + 1);
        if (!token) {
            fprintf(stderr, "malloc_tokenizer: malloc of %d bytes failed\n", len + 1);
            exit(1);
        }
        if (fread(token, 1, len, f) != (size_t)len) {
            fprintf(stderr, "malloc_tokenizer: failed to read token bytes for token %d in %s\n", i, path);
            exit(1);
        }
        token[len] = '\0';
        t->vocab[i] = token;
    }

    fclose(f);
    for (int i = 0; i < 256; i++) {
        t->byte_pieces[(2*i)] = i;
        t->byte_pieces[(2*i) + 1] = '\0';
    }
}


void free_tokenizer(Tokenizer *t) {
    for (int i = 0; i < t->vocab_size; i++) free(t->vocab[i]);
    free(t->vocab);
    free(t->vocab_scores);
}

char *decode(Tokenizer *t, int prev_token, int token) {
    char *piece = t->vocab[token];

    if (prev_token == 1 && piece[0] == ' ') {
        piece += 1;
    }

    //byte-fallback
    if (piece[0] == '<' && piece[1] == '0' && piece[2] == 'x' && piece[5] == '>') {
        int hi = hex_digit(piece[3]);
        int lo = hex_digit(piece[4]);
        unsigned char byte_val = (hi << 4) | lo;
        piece = ((char *)t->byte_pieces) + (2 * byte_val);
    }

    return piece;

}

void safe_printf(char *piece) {
    if (piece == NULL) return;
    if (piece[0] == '\0') return;

    // if this is a single byte, only print it if its printable or whitespace
    if (piece[1] == '\0') {
        unsigned char byte = (unsigned char)piece[0];
        if (!(isprint(byte) || isspace(byte))) {
            return;
        }
    }

    printf("%s", piece);
}
