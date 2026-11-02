MODEL ?= $(HOME)/.cache/kungfu-models/Llama-3.2-1B-Instruct-Q4_K_M.gguf
MODELS ?= $(MODEL) $(HOME)/.cache/kungfu-models/gemma-3-4b-it-Q4_K_M.gguf
PORT ?= 8080
PP ?= 512
TG ?= 128
REPS ?= 3

.PHONY: build clean dev test bench

build:
	cargo build --release

clean:
	cargo clean

dev:
	cargo run --release -p kungfu -- serve --model $(MODEL) --port $(PORT) $(ARGS)

test:
	cargo test --workspace

bench: build
	python3 -I tools/bench/compare.py --pp $(PP) --tg $(TG) --reps $(REPS) $(MODELS)
