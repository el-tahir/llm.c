CC = gcc
CFLAGS = -Wall -Wextra -std=c11 -O2 -Isrc
LDLIBS = -lm

# every .c under src/ is part of the library. run.c lives at the root and holds main()
SRC = $(wildcard src/*.c)
OBJ = $(SRC:.c=.o)

TESTS = test_stage0 test_primitives test_embedding test_rope test_attention test_ffn

tests: $(TESTS)
	@for t in $(TESTS); do \
			echo "running $$t..."; \
			./$$t || exit 1; \
		done

src/%.o: src/%.c src/tinyllm.h
	$(CC) $(CFLAGS) -c -o $@ $<

test_%: tests/test_%.c tests/test_common.h $(OBJ)
	$(CC) $(CFLAGS) -Itests -o $@ $< $(OBJ) $(LDLIBS)

run: run.c $(OBJ)
	$(CC) $(CFLAGS) -o run run.c $(OBJ) $(LDLIBS)

clean:
	rm -f run $(TESTS) $(OBJ)

.PHONY: tests clean
