CC = gcc
CFLAGS = -Wall -Wextra -std=c11 -O2
LDLIBS = -lm

TESTS = test_stage0 test_primitives test_embedding

tests: $(TESTS)

test_%: tests/test_%.c tests/test_common.h run.c
	$(CC) $(CFLAGS) -Itests -o $@ $< $(LDLIBS)

run: run.c
	$(CC) $(CFLAGS) -o run run.c $(LDLIBS)

clean:
	rm -f run $(TESTS)

.PHONY: tests clean
