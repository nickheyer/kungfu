# kungfu

Inference runtime based loosely on STRATA. Teaches your inference server Kung-Fu.

## Install

```sh
cargo install --path crates/kungfu
cargo install --path crates/kungfu --no-default-features
```

## Usage

```sh
kungfu inspect --json model.gguf
kungfu describe
kungfu serve --model model.gguf --host 127.0.0.1 --port 8080 --ctx auto --parallel 4
kungfu serve --model model.gguf --plan plan.json --cache-dir cache --profile cache/routing_profile.json
kungfu serve --model vl.gguf --mmproj mmproj.gguf --port 8080
kungfu serve --model model.gguf --role stage --layer-from 8 --layer-to 20 --port 9001 --admit 10.0.0.1
kungfu serve --model model.gguf --role head --peer stage=0@10.0.0.2:9001 --port 8080
kungfu serve --model model.gguf --role prefill --slot-dir slots --port 8081
kungfu serve --model model.gguf --role draft --draft-model small.gguf --port 9002
kungfu serve --model model.gguf --role head --peer draft=0@10.0.0.3:9002 --draft-max 4 --port 8080
```

## Novel Objective Requirement

Do not refer to existing implementations in competing applications for a solution to our "5-10x better performance for the same or less resource cost on the same device(s)". Do not attempt to re-interprete this requirement, it is immutable. You will always choose the larger scope that has the lowest amount of available research, and you will do the research yourself to achieve the objective at all costs.