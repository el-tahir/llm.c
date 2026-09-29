#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <string.h>

#include "tinyllm.h"

static int compare_tokens(const void *a, const void *b) {
    const char *s1 = ((const TokenIndex *)a)->str;
    const char *s2 = ((const TokenIndex *)b)->str;
    return strcmp(s1, s2);
}

static int hex_digit(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    die("hex_digit: invalid hex character '%c'", c);
}

void malloc_tokenizer(Tokenizer *t, const char *path, int vocab_size) {
    t->vocab_size = vocab_size;

    FILE *f = xfopen(path, "rb");
    xfread(&t->max_token_length, sizeof(unsigned int), 1, f, path);

    t->vocab = xmalloc(vocab_size * sizeof(*t->vocab));
    t->vocab_scores = xmalloc(vocab_size * sizeof(*t->vocab_scores));

    for (int i = 0; i < vocab_size; i++) {
        int len;
        xfread(&t->vocab_scores[i], sizeof(float), 1, f, path);
        xfread(&len, sizeof(int), 1, f, path);
        if (len < 0) {
            die("malloc_tokenizer: invalid negative length %d for token %d in %s", len, i, path);
        }

        char *token = xmalloc(len + 1);
        xfread(token, 1, len, f, path);
        token[len] = '\0';
        t->vocab[i] = token;
    }

    fclose(f);
    for (int i = 0; i < 256; i++) {
        t->byte_pieces[(2*i)] = i;
        t->byte_pieces[(2*i) + 1] = '\0';
    }

    t->sorted_vocab = xmalloc(t->vocab_size * sizeof(TokenIndex));
    for (int i = 0; i < t->vocab_size; i++) {
        t->sorted_vocab[i].str = t->vocab[i];
        t->sorted_vocab[i].id = i;
    }

    qsort(t->sorted_vocab, t->vocab_size, sizeof(TokenIndex), compare_tokens);
}

void free_tokenizer(Tokenizer *t) {
    for (int i = 0; i < t->vocab_size; i++) free(t->vocab[i]);
    free(t->vocab);
    free(t->vocab_scores);
    free(t->sorted_vocab);
}

char *decode(Tokenizer *t, int prev_token, int token) {
    char *piece = t->vocab[token];

    if (prev_token == TOKEN_BOS && piece[0] == ' ') {
        piece += 1;
    }

    // byte-fallback
    if (strlen(piece) == 6 && piece[0] == '<' && piece[1] == '0' && piece[2] == 'x' && piece[5] == '>') {
        int hi = hex_digit(piece[3]);
        int lo = hex_digit(piece[4]);
        unsigned char byte_val = (hi << 4) | lo;
        piece = ((char *)t->byte_pieces) + (2 * byte_val);
    }

    return piece;
}
int str_lookup(char *str, TokenIndex *sorted_vocab, int vocab_size) {
    TokenIndex key;
    key.str = str;

    TokenIndex *result = (TokenIndex *)bsearch(&key, sorted_vocab, vocab_size, sizeof(TokenIndex),
                          compare_tokens);

    return result ? result->id : -1;
}

void encode(Tokenizer *t, const char *text, int bos, int eos, int *tokens, int *n_tokens) {
    int write = 0;
    if (bos) tokens[write++] = TOKEN_BOS;

    // dummy space prefix
    if (text[0] != '\0') {
        tokens[write++] = str_lookup(" ", t->sorted_vocab, t->vocab_size);
    }

    unsigned int buf_size = t->max_token_length * 2 + 3;
    char *str_buffer = xmalloc(buf_size);
    size_t str_len = 0;

    for (const char *c = text; *c != '\0'; c++) {
        // any byte that is not a continuation byte begins a new codepoint
        if ((*c & 0xC0) != 0x80) str_len = 0;

        str_buffer[str_len++] = *c;
        str_buffer[str_len] = '\0';

        // if the next byte continues this codepoint, keep collecting
        if ((*(c + 1) & 0xC0) == 0x80 && str_len < 4) continue;

        int id = str_lookup(str_buffer, t->sorted_vocab, t->vocab_size);
        if (id != -1) {
            tokens[write++] = id;
        } else {
            for (size_t i = 0; i < str_len; i++) {
                tokens[write++] = (unsigned char)str_buffer[i] + TOKEN_BYTE_OFFSET;
            }
        }
        str_len = 0;
    }

    // merge the best adjacent pair over and over until none is left
    while (1) {
        float best_score = -1e10;
        int best_id = -1;
        int best_idx = -1;

        for (int i = 0; i < write - 1; i++) {
            snprintf(str_buffer, buf_size, "%s%s",
                t->vocab[tokens[i]], t->vocab[tokens[i + 1]]);
            int id = str_lookup(str_buffer, t->sorted_vocab, t->vocab_size);
            if (id != -1 && t->vocab_scores[id] > best_score) {
                best_score = t->vocab_scores[id];
                best_id = id;
                best_idx = i;
            }
        }

        if (best_idx == -1) break;

        // replace the pair with the merged token, then close the gap
        tokens[best_idx] = best_id;
        for (int i = best_idx + 1; i < write - 1; i++) {
            tokens[i] = tokens[i + 1];
        }
        write--;
    }

    if (eos) tokens[write++] = TOKEN_EOS;

    free(str_buffer);
    *n_tokens = write;
}

void safe_printf(const char *piece) {
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
