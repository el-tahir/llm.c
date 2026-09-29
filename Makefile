CC = gcc
WARN = -Wall -Wextra -std=c11 -Isrc
CFLAGS = $(WARN) -g -O0
LDLIBS = -lm

# every .c under src/ is part of the library. run.c lives at the root and holds main()
SRC = $(wildcard src/*.c)
OBJ = $(SRC:.c=.o)

# model files, downloaded by 'make setup' instead of living in git
MODEL = stories15M.bin tokenizer.bin
STORIES_URL = https://huggingface.co/karpathy/tinyllamas/resolve/main/stories15M.bin
TOKENIZER_URL = https://raw.githubusercontent.com/karpathy/llama2.c/master/tokenizer.bin

TESTS = test_stage0 test_primitives test_embedding test_rope test_attention test_ffn test_forward test_sampler test_tokenizer test_generate

tests: $(TESTS) | $(MODEL)
	@for t in $(TESTS); do \
			echo "running $$t..."; \
			./$$t || exit 1; \
		done

story: run | $(MODEL)
	@./run stories15M.bin -t 0 -i "One day, Lily met a" -n 256 > /tmp/tinyllm_story.txt
	@cmp /tmp/tinyllm_story.txt ref/s12_text.bin && echo " ok story                  935 bytes match"

setup: $(MODEL)

# download to a temp name and rename, so an interrupted download never leaves a
# truncated file that make would treat as up to date
stories15M.bin:
	curl -fL --retry 3 -o $@.part $(STORIES_URL) && mv $@.part $@

tokenizer.bin:
	curl -fL --retry 3 -o $@.part $(TOKENIZER_URL) && mv $@.part $@

src/%.o: src/%.c src/tinyllm.h
	$(CC) $(CFLAGS) -c -o $@ $<

test_%: tests/test_%.c tests/test_common.h $(OBJ)
	$(CC) $(CFLAGS) -Itests -o $@ $< $(OBJ) $(LDLIBS)

run: run.c $(OBJ)
	$(CC) $(CFLAGS) -o run run.c $(OBJ) $(LDLIBS)

# optimized build. cleans first so no -O0 objects get linked in; run 'make clean'
# before going back to a debug build
release:
	$(MAKE) clean
	$(MAKE) run CFLAGS="$(WARN) -O2"

clean:
	rm -f run $(TESTS) $(OBJ)

.PHONY: tests clean story release setup
