#!/usr/bin/env python3
"""
Thin wrapper around bin/medley that fixes a Python 3.9 glob compatibility issue.

In Python 3.9, glob.glob('path/**.smt2', recursive=True) only matches files
directly in 'path/', not in subdirectories. The correct pattern is '**/*.smt2'.

This wrapper patches glob.glob to fix this, then runs the real bin/medley
main() function completely unmodified.
"""
import glob
import sys

# Patch glob.glob for Python <3.11 compatibility
_orig_glob = glob.glob

def _patched_glob(pattern, *, recursive=False):
    if recursive and pattern.endswith("/**.smt2"):
        pattern = pattern[:-len("/**.smt2")] + "/**/*.smt2"
    return _orig_glob(pattern, recursive=recursive)

glob.glob = _patched_glob

# Run the real bin/medley
MEDLEY_BIN = "/dcs/large/u5573765/artifacts/medley-solver/bin/medley"
with open(MEDLEY_BIN) as f:
    code = f.read()
exec(compile(code, MEDLEY_BIN, "exec"))
