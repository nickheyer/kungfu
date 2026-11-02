"""Runs llama-bench and kungfu bench on the same models with the same parameters and prints both with their ratios"""
import argparse
import math
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
LLAMA_BENCH = os.path.join(HERE, "llama-bench")
KUNGFU = os.path.join(ROOT, "target", "release", "kungfu")


def table(cmd):
    """Runs a bench command and returns test name to tokens a second from its markdown rows"""
    out = subprocess.run(cmd, capture_output=True, text=True)
    if out.returncode != 0:
        sys.exit(f"{' '.join(cmd)} failed\n{out.stderr[-2000:]}")
    rows = {}
    for line in out.stdout.splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) < 2 or not re.match(r"^\d", cells[-1]):
            continue
        rows[cells[-2]] = float(cells[-1].split()[0])
    if not rows:
        sys.exit(f"{' '.join(cmd)} printed no results\n{out.stdout[-2000:]}")
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("models", nargs="+")
    ap.add_argument("--pp", type=int, default=512)
    ap.add_argument("--tg", type=int, default=128)
    ap.add_argument("--reps", type=int, default=3)
    a = ap.parse_args()
    for path in (LLAMA_BENCH, KUNGFU):
        if not os.access(path, os.X_OK):
            sys.exit(f"{path} is missing")
    tests = [f"pp{a.pp}", f"tg{a.tg}"]
    ratios = {t: [] for t in tests}
    print(f"| {'model':<34} | {'test':>6} | {'llama.cpp t/s':>13} | {'kungfu t/s':>11} | {'kungfu/llama':>12} |")
    print(f"| {'-' * 34} | {'-' * 6} | {'-' * 13} | {'-' * 11} | {'-' * 12} |")
    for m in a.models:
        llama = table([LLAMA_BENCH, "-m", m, "-p", str(a.pp), "-n", str(a.tg), "-r", str(a.reps), "-fa", "on", "-ngl", "99"])
        ours = table([KUNGFU, "bench", "--model", m, "--pp", str(a.pp), "--tg", str(a.tg), "--reps", str(a.reps)])
        name = os.path.basename(m).removesuffix(".gguf")[:34]
        for t in tests:
            if t not in llama or t not in ours:
                sys.exit(f"{t} missing from a result, llama.cpp has {sorted(llama)}, kungfu has {sorted(ours)}")
            r = ours[t] / llama[t]
            ratios[t].append(r)
            print(f"| {name:<34} | {t:>6} | {llama[t]:>13.1f} | {ours[t]:>11.1f} | {r:>11.2f}x |")
    print()
    for t in tests:
        geo = math.exp(sum(math.log(r) for r in ratios[t]) / len(ratios[t]))
        print(f"{t}: kungfu is {geo:.2f}x llama.cpp on average over {len(ratios[t])} model(s), {100 * (geo - 1):+.1f}%")


if __name__ == "__main__":
    main()
