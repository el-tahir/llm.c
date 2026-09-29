#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>

#include "tinyllm.h"

void die(const char *fmt, ...) {
    va_list args;
    va_start(args, fmt);
    vfprintf(stderr, fmt, args);
    va_end(args);
    fputc('\n', stderr);
    exit(EXIT_FAILURE);
}

void *xmalloc(size_t size) {
    void *p = malloc(size);
    if (!p) die("malloc of %zu bytes failed", size);
    return p;
}

void *xcalloc(size_t n, size_t size) {
    void *p = calloc(n, size);
    if (!p) die("calloc of %zu x %zu bytes failed", n, size);
    return p;
}

FILE *xfopen(const char *path, const char *mode) {
    FILE *f = fopen(path, mode);
    if (!f) die("cannot open file %s", path);
    return f;
}

void xfread(void *buf, size_t size, size_t n, FILE *f, const char *path) {
    if (fread(buf, size, n, f) != n) die("unexpected end of file %s", path);
}
