#!/usr/bin/env python3
"""Compare the machine code of two builds of the same binary, function by function.

A change that claims to leave a code path untouched can be checked rather than
timed: disassemble both builds, strip what moves when anything else in the
binary moves — addresses, branch targets, symbol hashes — and compare each
function's instruction stream. Identical instructions are identical cost, which
no benchmark on a loaded machine can establish to better than a few percent.

    cargo bench --bench search --no-run            # in each checkout
    python3 benchmarks/codegen_diff.py BEFORE AFTER [--all]

BEFORE and AFTER are the two `target/release/deps/<bench>-<hash>` executables.
Without `--all`, only functions whose names mention the crate or the bench are
compared. Exits non-zero if any compared function differs.

Needs an `objdump` that understands `--no-leading-addr` (LLVM's does).
"""

import collections
import re
import subprocess
import sys

HASH = re.compile(r"(::)?h[0-9a-f]{16}")
NUMBER = re.compile(r"0x[0-9a-f]+|\b[0-9a-f]{6,}\b")
TARGET = re.compile(r"<[^>]*>")


def functions(path):
    listing = subprocess.run(
        ["objdump", "-d", "--no-show-raw-insn", "--no-leading-addr", path],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    funcs = collections.OrderedDict()
    name = None
    for line in listing.splitlines():
        label = re.match(r"^(?:[0-9a-f]+ )?<(.+)>:$", line)
        if label:
            name = HASH.sub("", label.group(1))
            funcs.setdefault(name, [])
            continue
        if name and line.strip():
            instruction = NUMBER.sub("N", line.strip())
            instruction = TARGET.sub("<S>", instruction)
            funcs[name].append(HASH.sub("", instruction))
    return funcs


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    every = "--all" in sys.argv
    if len(args) != 2:
        sys.exit(__doc__)
    before, after = functions(args[0]), functions(args[1])
    wanted = (lambda n: True) if every else (lambda n: "mcts" in n or "closure" in n)

    shared = [n for n in before if wanted(n) and n in after]
    differ = [n for n in shared if before[n] != after[n]]
    print(
        f"{len(shared)} functions compared: {len(shared) - len(differ)} identical, "
        f"{len(differ)} differ"
    )
    for name in differ:
        print(f"  differs: {name} ({len(before[name])} -> {len(after[name])} instructions)")
    # Functions present in only one build are usually the compiler renumbering
    # local duplicates (`.1222` against `.1224`); listed so a reader can tell.
    only = [n for n in before if wanted(n) and n not in after]
    if only:
        print(f"  {len(only)} named only in BEFORE, e.g. {only[0][:100]}")
    sys.exit(1 if differ else 0)


if __name__ == "__main__":
    main()
