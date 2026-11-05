CC      ?= cc
CFLAGS  ?= -O3 -march=native -std=c11 -Wall -Wextra -Wpedantic -D_POSIX_C_SOURCE=200809L
LDLIBS  = -lpthread -lm
SRC     = src/arena.c src/threadpool.c src/kernels.c src/model.c

all: build/libtinyinfer.a build/test build/bench

build:
	mkdir -p build

build/libtinyinfer.a: $(SRC) include/tinyinfer.h | build
	$(foreach s,$(SRC),$(CC) $(CFLAGS) -Iinclude -c $(s) -o build/$(notdir $(s:.c=.o)) &&) true
	ar rcs $@ $(addprefix build/,$(notdir $(SRC:.c=.o)))

build/test: tests/test_main.c build/libtinyinfer.a
	$(CC) $(CFLAGS) -Iinclude $< build/libtinyinfer.a -o $@ $(LDLIBS)

build/bench: bench/bench.c build/libtinyinfer.a
	$(CC) $(CFLAGS) -Iinclude $< build/libtinyinfer.a -o $@ $(LDLIBS)

models/digits_f32.tinf: tools/train_export.py
	python3 tools/train_export.py --out models

test: build/test models/digits_f32.tinf
	./build/test models

# Same tests under AddressSanitizer + UndefinedBehaviorSanitizer, and ThreadSanitizer.
sanitize: models/digits_f32.tinf | build
	$(CC) -O1 -g -std=c11 -D_POSIX_C_SOURCE=200809L -fsanitize=address,undefined -fno-omit-frame-pointer \
	    -Iinclude tests/test_main.c $(SRC) -o build/test_asan $(LDLIBS) && ./build/test_asan models
	$(CC) -O1 -g -std=c11 -D_POSIX_C_SOURCE=200809L -fsanitize=thread \
	    -Iinclude tests/test_main.c $(SRC) -o build/test_tsan $(LDLIBS) && ./build/test_tsan models

bench: build/bench models/digits_f32.tinf
	./build/bench $(shell nproc)

clean:
	rm -rf build

.PHONY: all test sanitize bench clean
