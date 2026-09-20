#include <string.h>

#include "tinyllm.h"
#include "test_common.h"

static const int S8_TOKENS[] = {1, 306, 3186, 29889, 0, 31999, 450, 6635, 13, 2};
#define N_TOK   ((int)(sizeof(S8_TOKENS) / sizeof(S8_TOKENS[0])))

static const int S10_EMOJI[] = {243, 162, 155, 141};
#define N_EMOJI ((int)(sizeof(S10_EMOJI) / sizeof(S10_EMOJI[0])))

// the same fourteen strings as ref/ref.py, in the same order
static const char *S11_CORPUS[] = {
    "Once",
    "One day, Lily met a",
    "Once upon a time, there was a little girl",
    "",
    "  ",
    "a  b",
    "café naïve",
    "😊",
    "I 😊 you",
    "日本語",
    "tab\there",
    "MiXeD CaSe 123",
    "   leading",
    "trailing   ",
};
#define N_CORPUS ((int)(sizeof(S11_CORPUS) / sizeof(S11_CORPUS[0])))

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

// encode one string and compare the ids against reference
static int check_ids(Tokenizer *t, const char *label, const char *text,
    int bos, int eos, const char *path) {
        int got[512];
        if (strlen(text) + 3 > sizeof(got) / sizeof(got[0])) {
            printf(" FAIL %-22s input too long for the test buffer\n", label);
            return 1;
        }
        int got_n;
        encode(t, text, bos, eos, got, &got_n);

        int want_n;
        int *want = load_ints(path, &want_n);
        int r = compare_ints(label, got, got_n, want, want_n);
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

    printf("\n");

    // the sorted view, ascending
    int sorted_ok = 1;
    for (int i = 0; i < t.vocab_size; i++) {
        TokenIndex *e = &t.sorted_vocab[i];
        if (e->id < 0 || e->id >= t.vocab_size || e->str != t.vocab[e->id]) sorted_ok = 0;
        if (i > 0 && strcmp(t.sorted_vocab[i - 1].str, e->str) >= 0) sorted_ok = 0;
    }
    printf(sorted_ok ? " ok %-22s strictly ascending, ids intact\n"
                     : " FAIL %-22s order or ids broken\n", "sorted view");
    fails += !sorted_ok;

    // str_lookup is the exact inverse of vocab[], for all 32,000
    int lookup_ok = 1;
    for (int i = 0; i < t.vocab_size; i++) {
        if (str_lookup(t.vocab[i], t.sorted_vocab, t.vocab_size) != i) lookup_ok = 0;
    }

    char *absent = "a string longer than twenty-seven bytes";
    if (str_lookup(absent, t.sorted_vocab, t.vocab_size) != -1) lookup_ok = 0;
    printf(lookup_ok ? " ok %-22s 32000 hits and one miss\n"
                     : " FAIL %-22s wrong id or a false hit\n", "str_lookup");
    fails += !lookup_ok;

    fails += check_ids(&t, "encode once",   "Once",                1, 0, "ref/s11_once.bin");
    fails += check_ids(&t, "encode prompt", "One day, Lily met a", 1, 0, "ref/s11_prompt.bin");
    fails += check_ids(&t, "encode spaces", "  ",                  1, 0, "ref/s11_spaces.bin");
    fails += check_ids(&t, "encode emoji",  "😊",                  1, 0, "ref/s11_emoji.bin");
    fails += check_ids(&t, "encode cjk",    "日本語",               1, 0, "ref/s11_cjk.bin");
    fails += check_ids(&t, "encode empty",  "",                    1, 0, "ref/s11_empty.bin");
    fails += check_ids(&t, "encode flags",  "Once",                0, 1, "ref/s11_flags.bin");

    // the whole corpus
    int corpus[512];
    int corpus_lens[N_CORPUS];
    int corpus_n = 0;
    for (int i = 0; i < N_CORPUS; i++) {
        int n;
        encode(&t, S11_CORPUS[i], 1, 0, corpus + corpus_n, &n);
        corpus_lens[i] = n;
        corpus_n += n;
    }

    int want_n;
    int *want = load_ints("ref/s11_corpus_lens.bin", &want_n);
    fails += compare_ints("compare lengths", corpus_lens, N_CORPUS, want, want_n);
    free(want);

    want = load_ints("ref/s11_corpus.bin", &want_n);
    fails += compare_ints("corpus ids", corpus, corpus_n, want, want_n);
    free(want);

    // property: decode(encode(s)) == s
    int rt_ok = 1;
    for (int i = 0, off = 0; i < N_CORPUS; off += corpus_lens[i], i++) {
        const int *ids = corpus + off;
        char back[1200];
        size_t len = 0;
        for (int j = 1; j < corpus_lens[i]; j++) {
            char *p = decode(&t, ids[j - 1], ids[j]);
            size_t pl = strlen(p);
            memcpy(back + len, p , pl);
            len += pl;
        }
        back[len] = '\0';
        if (len != strlen(S11_CORPUS[i]) || memcmp(back, S11_CORPUS[i], len) != 0) {
            printf( "FAIL %-22s [%d] got \"%s\", want \"%s\"\n" ,
                "round trip", i, back, S11_CORPUS[i]);
            rt_ok = 0;
        }
    }
    if (rt_ok) { printf(" ok %-22s %d strings, exact\n", "round trip", N_CORPUS);}
    fails += !rt_ok;

    printf("\n how it splits: ");
    {
        int ids[512], n;
        encode(&t, "One day, Lily met a", 1, 0, ids, &n);
        for (int i = 1; i < n; i++) {
            safe_printf(decode(&t, ids[i - 1], ids[i]));
            printf("|");
        }
    }
    printf("\n");

    free_tokenizer(&t);
    free(data);

    printf("\n STAGES 10-11: %s\n", fails ? "FAILS" : "PASS");
    return fails;

}
