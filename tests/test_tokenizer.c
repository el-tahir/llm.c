#include <string.h>

#include "tinyllm.h"
#include "test_common.h"

static const int S8_TOKENS[] = {1, 306, 3186, 29889, 0, 31999, 450, 6635, 13, 2};
#define N_TOK   ((int)(sizeof(S8_TOKENS) / sizeof(S8_TOKENS[0])))

static const int S10_EMOJI[] = {243, 162, 155, 141};
#define N_EMOJI ((int)(sizeof(S10_EMOJI) / sizeof(S10_EMOJI[0])))

static int check_bytes(const char *label, const unsigned char *got, long got_n,
    const char *path) {
        long n;
        unsigned char *want = load_bytes(path, &n);
        int r;
        if (got_n != n) {
            printf(" FAIL %-22s produced %ld bytes, reference has %ld\n", label, got_n, n);
            r = 1;
        } else {
            r = compare_bytes(label, got, want, n);
        }
        free(want);
        return r;
    }


    int main(void) {
        int fails = 0;

        Config config;
        TransformerWeights weights;
        float *data = NULL;
        long file_size = 0;
        read_checkpoint("stories15M.bin", &config, &weights, &data, &file_size);

        Tokenizer t;
        malloc_tokenizer(&t, "tokenizer.bin", config.vocab_size);

        // the header
        float meta[2] = { (float)t.max_token_length, (float)t.vocab_size };
        float *expected  = load_bin("ref/s10_meta.bin", 2);
        fails += compare("header", meta, expected, 2, 0.0f);
        free(expected);

        // every token's length
        float *lens = malloc(t.vocab_size * sizeof(float));
        for (int i = 0; i < t.vocab_size; i++) { lens[i] = (float)strlen(t.vocab[i]); }
        expected = load_bin("ref/s10_lens.bin", t.vocab_size);
        fails += compare("vocab lengths", lens, expected, t.vocab_size, 0.0f);
        free(expected);
        free(lens);

        // the whole vocabulary
        long total = 0;
        for (int i = 0; i < t.vocab_size; i++) { total += (long)strlen(t.vocab[i]); }
        unsigned char *blob = malloc(total);
        long off = 0;
        for (int i = 0; i < t.vocab_size; i++) {
            size_t len = strlen(t.vocab[i]);
            memcpy(blob + off, t.vocab[i], len);
            off += (long)len;
        }
        fails += check_bytes("vocab bytes", blob, total, "ref/s10_vocab.bin");
        free(blob);

        // the pinned sequence
        unsigned char seq[512];
        float piece_lens[N_TOK];
        long m = 0;
        for (int i = 0; i < N_TOK; i++) {
            char *p = decode(&t, i ? S8_TOKENS[i - 1] : -1, S8_TOKENS[i]);
            size_t len = strlen(p);
            piece_lens[i] = (float)len;
            memcpy(seq + m, p, len);
            m += (long)len;
        }

        expected = load_bin("ref/s10_piece_lens.bin", N_TOK);
        fails += compare("piece lengths", piece_lens, expected, N_TOK, 0.0f);
        free(expected);

        fails += check_bytes("decode sequence", seq, m, "ref/s10_decode.bin");

        // BOS
        char *bos = decode(&t, 1, 9038);
        fails += check_bytes("bos strip", (unsigned char *)bos, (long)strlen(bos),
            "ref/s10_bos.bin");

        char *nobos = decode(&t, 29889, 9038);
        fails += check_bytes("bos control", (unsigned char *)nobos, (long)strlen(nobos),
            "ref/s10_nobos.bin");

        // emoji
        unsigned char emoji[512];
        long k = 0;
        for (int i = 0; i < N_EMOJI; i++) {
            char *p = decode(&t, -1, S10_EMOJI[i]);
            size_t len = strlen(p);
            if (len != 1) {
                printf(" FAIL %-22s piece %d is %zu bytes, want 1\n", "byte fallback", i, len);
                fails++;
            }
            memcpy(emoji + k, p, len);
            k += (long)len;
        }
        fails += check_bytes("byte fallback", emoji, k, "ref/s10_emoji.bin");

        // property: decode must not consume the table
        unsigned char again[512];
        long m2 = 0;
        for (int i = 0; i < N_TOK; i++) {
            char *p = decode(&t, i ? S8_TOKENS[i - 1] : -1, S8_TOKENS[i]);
            size_t len = strlen(p);
            memcpy(again + m2, p, len);
            m2 += (long)len;
        }
        if (m2 == m && memcmp(again, seq, (size_t)m) == 0) {
            printf(" ok %-22s second pass identical\n", "decode repeatable");
        } else {
            printf(" FAIL %-22s second pass is different\n", "decode repeatable");
            fails++;
        }

        // stage 9 ids as text
        float *g = load_bin("ref/s9_argmax.bin", N_TOK);
        unsigned char text[512];
        long q = 0;
        for (int i = 0; i < N_TOK; i++) {
            char *p = decode(&t, S8_TOKENS[i], (int)g[i]);
            size_t len = strlen(p);
            memcpy(text + q, p, len);
            q += (long)len;
        }
        fails += check_bytes("greedy text", text, q, "ref/s10_greedy.bin");

        printf("\n what the model said: ");
        for (int i = 0; i < N_TOK; i++) { safe_printf(decode(&t, S8_TOKENS[i], (int)g[i])); }
        printf("\n");
        free(g);

        free_tokenizer(&t);
        free(data);

        printf("\n STAGE 10: %s\n", fails ? "FAILS" : "PASS");
        return fails;

    }
