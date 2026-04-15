"""From-scratch S-expression tokenizer and SMT-LIB parser.

Provides lightweight parsing of SMT-LIB2 files sufficient for feature extraction.
No external parser dependency — hand-written tokenizer with iterative DFS.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union


@dataclass
class FormulaInfo:
    """Parsed information from an SMT-LIB2 file."""

    logic: Optional[str] = None
    status: Optional[str] = None  # sat / unsat / unknown
    file_size_bytes: int = 0
    n_assertions: int = 0
    n_declare_fun: int = 0
    n_define_fun: int = 0
    n_check_sat: int = 0
    declared_vars: List[str] = field(default_factory=list)
    declared_sorts: Dict[str, str] = field(default_factory=dict)  # name -> sort
    assertions: List[list] = field(default_factory=list)  # parsed s-exprs
    raw_tokens: List[str] = field(default_factory=list)  # all tokens
    sexprs: List[list] = field(default_factory=list)  # all top-level s-exprs


def tokenize(text: str) -> List[str]:
    """Tokenize SMT-LIB2 text into a list of tokens.

    Handles: parentheses, quoted symbols |...|, string literals "...",
    comments ;..., and standard atoms.
    """
    tokens = []
    i = 0
    n = len(text)
    while i < n:
        c = text[i]
        # Whitespace
        if c in (' ', '\t', '\n', '\r'):
            i += 1
        # Comment
        elif c == ';':
            while i < n and text[i] != '\n':
                i += 1
        # Parentheses
        elif c == '(':
            tokens.append('(')
            i += 1
        elif c == ')':
            tokens.append(')')
            i += 1
        # Quoted symbol |...|
        elif c == '|':
            j = i + 1
            while j < n and text[j] != '|':
                j += 1
            tokens.append(text[i:j + 1])
            i = j + 1
        # String literal "..."
        elif c == '"':
            j = i + 1
            while j < n:
                if text[j] == '"':
                    if j + 1 < n and text[j + 1] == '"':
                        j += 2  # escaped quote
                    else:
                        break
                else:
                    j += 1
            tokens.append(text[i:j + 1])
            i = j + 1
        # Atom (symbol, numeral, keyword, etc.)
        else:
            j = i
            while j < n and text[j] not in (' ', '\t', '\n', '\r', '(', ')', ';', '"', '|'):
                j += 1
            tokens.append(text[i:j])
            i = j
    return tokens


def parse_sexpr(tokens: List[str], pos: int) -> Tuple[Union[str, list], int]:
    """Parse one S-expression starting at tokens[pos].

    Iterative (stack-based) to handle deeply nested formulas without recursion.
    Returns (parsed, next_pos). Parsed is either a string atom or a list.
    """
    if pos >= len(tokens):
        return '', pos
    tok = tokens[pos]
    if tok != '(':
        return tok, pos + 1

    # Stack-based iterative parsing
    stack: List[list] = []
    current: list = []
    pos += 1  # skip opening '('

    while pos < len(tokens):
        tok = tokens[pos]
        if tok == '(':
            stack.append(current)
            current = []
            pos += 1
        elif tok == ')':
            pos += 1
            if stack:
                parent = stack.pop()
                parent.append(current)
                current = parent
            else:
                return current, pos
        else:
            current.append(tok)
            pos += 1

    # Unbalanced parens — return what we have
    return current, pos


def parse_all_sexprs(tokens: List[str]) -> List[Union[str, list]]:
    """Parse all top-level S-expressions from a token list."""
    sexprs = []
    pos = 0
    while pos < len(tokens):
        expr, pos = parse_sexpr(tokens, pos)
        if expr != '':
            sexprs.append(expr)
    return sexprs


def parse_file(path: Union[str, Path]) -> FormulaInfo:
    """Parse an SMT-LIB2 file and extract structural information.

    Returns a FormulaInfo with logic, assertions, declared variables, etc.
    """
    path = Path(path)
    file_size = path.stat().st_size

    text = path.read_text(errors='replace')
    tokens = tokenize(text)
    sexprs = parse_all_sexprs(tokens)

    info = FormulaInfo(
        file_size_bytes=file_size,
        raw_tokens=tokens,
        sexprs=sexprs,
    )

    for expr in sexprs:
        if not isinstance(expr, list) or len(expr) == 0:
            continue
        cmd = expr[0] if isinstance(expr[0], str) else ''

        if cmd == 'set-logic' and len(expr) >= 2:
            info.logic = expr[1] if isinstance(expr[1], str) else None

        elif cmd == 'set-info' and len(expr) >= 3:
            key = expr[1] if isinstance(expr[1], str) else ''
            if key == ':status' and isinstance(expr[2], str):
                info.status = expr[2]

        elif cmd == 'declare-fun' and len(expr) >= 2:
            info.n_declare_fun += 1
            var_name = expr[1] if isinstance(expr[1], str) else ''
            info.declared_vars.append(var_name)
            # Extract sort from (declare-fun name (args) sort)
            if len(expr) >= 4:
                sort = _sort_to_str(expr[3])
                info.declared_sorts[var_name] = sort

        elif cmd == 'declare-const' and len(expr) >= 2:
            info.n_declare_fun += 1
            var_name = expr[1] if isinstance(expr[1], str) else ''
            info.declared_vars.append(var_name)
            if len(expr) >= 3:
                sort = _sort_to_str(expr[2])
                info.declared_sorts[var_name] = sort

        elif cmd == 'define-fun':
            info.n_define_fun += 1

        elif cmd == 'assert' and len(expr) >= 2:
            info.n_assertions += 1
            info.assertions.append(expr[1])

        elif cmd == 'check-sat':
            info.n_check_sat += 1

    return info


def _sort_to_str(sort) -> str:
    """Convert a sort expression to a string representation."""
    if isinstance(sort, str):
        return sort
    if isinstance(sort, list):
        return '(' + ' '.join(_sort_to_str(s) for s in sort) + ')'
    return str(sort)


def extract_cnf(assertions: List) -> Tuple[List[List[int]], Dict[str, int]]:
    """Extract a pseudo-CNF representation from parsed assertions.

    Treats each assertion as a clause. Variables in assertions get integer IDs.
    Negated variables get negative IDs. Non-variable subexpressions are
    flattened — this is an approximation for graph construction, not exact CNF.

    Returns:
        clauses: list of lists of signed variable IDs
        var_to_id: mapping from variable name to positive integer ID
    """
    var_to_id: Dict[str, int] = {}
    next_id = 1
    clauses: List[List[int]] = []

    for assertion in assertions:
        clause_lits = []
        _collect_literals(assertion, clause_lits, var_to_id, False)
        if clause_lits:
            clauses.append(clause_lits)
        elif assertion:
            # If no literals found, treat the whole assertion as one unit
            _collect_atoms_into(assertion, clause_lits, var_to_id)
            if clause_lits:
                clauses.append(clause_lits)

    return clauses, var_to_id


def _get_or_assign_id(name: str, var_to_id: Dict[str, int]) -> int:
    """Get existing variable ID or assign a new one."""
    if name not in var_to_id:
        var_to_id[name] = len(var_to_id) + 1
    return var_to_id[name]


_LOGICAL_OPS = frozenset({'or', 'and', '=>', 'ite', 'not'})
_RELATIONAL_OPS = frozenset({
    '=', '<=', '>=', '<', '>', 'bvult', 'bvslt', 'bvugt', 'bvsgt',
    'bvule', 'bvsle', 'bvuge', 'bvsge', 'distinct',
})
_SKIP_ATOMS = frozenset({'true', 'false'})


def _collect_literals(
    expr, lits: List[int], var_to_id: Dict[str, int], negated: bool
) -> None:
    """Iteratively collect literals from an assertion (stack-based)."""
    # Stack items: (expression, negated)
    work: List[Tuple] = [(expr, negated)]

    while work:
        node, neg = work.pop()

        if isinstance(node, str):
            if node in _SKIP_ATOMS or node.startswith(':'):
                continue
            if _is_numeral(node) or node.startswith('"') or node.startswith('|'):
                continue
            vid = _get_or_assign_id(node, var_to_id)
            lits.append(-vid if neg else vid)
            continue

        if not isinstance(node, list) or len(node) == 0:
            continue

        op = node[0] if isinstance(node[0], str) else None

        if op == 'not' and len(node) == 2:
            work.append((node[1], not neg))
        elif op in ('or', 'and', '=>', 'ite'):
            for child in reversed(node[1:]):
                work.append((child, neg))
        elif op in _RELATIONAL_OPS:
            for child in reversed(node[1:]):
                work.append((child, False))  # atoms from relational, unsigned
        else:
            # General function application — collect all atoms unsigned
            _collect_atoms_into(node, lits, var_to_id)


def _collect_atoms_into(expr, lits: List[int], var_to_id: Dict[str, int]) -> None:
    """Iteratively collect all variable atoms from an expression (unsigned)."""
    work = [expr]
    while work:
        node = work.pop()
        if isinstance(node, str):
            if (node in _SKIP_ATOMS or node.startswith(':') or
                    node.startswith('"') or node.startswith('|') or
                    _is_numeral(node)):
                continue
            # Skip known operators
            if node in _LOGICAL_OPS or node in _RELATIONAL_OPS:
                continue
            vid = _get_or_assign_id(node, var_to_id)
            lits.append(vid)
        elif isinstance(node, list):
            for child in reversed(node):
                work.append(child)


def _is_numeral(s: str) -> bool:
    """Check if string is a numeral (integer or decimal)."""
    if not s:
        return False
    s = s.lstrip('-')
    return s.replace('.', '', 1).isdigit() if s else False


def count_nodes(expr) -> int:
    """Count the total number of nodes in a parsed S-expression tree (iterative)."""
    count = 0
    work = [expr]
    while work:
        node = work.pop()
        count += 1
        if isinstance(node, list):
            work.extend(node)
    return count


def collect_operators(expr) -> Dict[str, int]:
    """Collect operator frequency counts from a parsed S-expression (iterative)."""
    counts: Dict[str, int] = {}
    work = [expr]
    while work:
        node = work.pop()
        if isinstance(node, list) and len(node) > 0:
            op = node[0]
            if isinstance(op, str) and not _is_numeral(op):
                counts[op] = counts.get(op, 0) + 1
            work.extend(node[1:])
    return counts


def tree_depth(expr) -> int:
    """Compute the maximum depth of a parsed S-expression tree (iterative)."""
    max_depth = 0
    # Stack items: (node, current_depth)
    work: List[Tuple[Any, int]] = [(expr, 0)]
    while work:
        node, depth = work.pop()
        if isinstance(node, str) or not isinstance(node, list) or len(node) == 0:
            if depth > max_depth:
                max_depth = depth
        else:
            for child in node:
                work.append((child, depth + 1))
    return max_depth
