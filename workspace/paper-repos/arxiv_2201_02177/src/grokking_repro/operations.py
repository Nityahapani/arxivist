"""Exact finite binary operations used in Power et al. (2022), Appendix A.1.1.

Residues are represented as Python integers.  S5 elements are image tuples
(p(0), ..., p(4)); the product convention follows the authors' released data
code: ``(x * y)[i] = y[x[i]]``.  The paper does not specify a permutation
encoding convention, so this is recorded as a reproducibility assumption.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations, product
import random
from typing import Callable, TypeAlias

P = 97
Residue: TypeAlias = int
Permutation5: TypeAlias = tuple[int, int, int, int, int]
Element: TypeAlias = Residue | Permutation5
S5: tuple[Permutation5, ...] = tuple(permutations(range(5)))  # 120 elements


@dataclass(frozen=True)
class Equation:
    """One labelled example ``left op right = answer``."""

    left: Element
    operator: str
    right: Element
    answer: Element


@dataclass(frozen=True)
class Operation:
    key: str
    paper_name: str
    operator_token: str
    formula: str
    domain: str
    elements: tuple[Element, ...]
    apply: Callable[[Element, Element], Element]
    paper_section: str = "Appendix A.1.1"


def _mod_add(x: Element, y: Element) -> Element:
    return (int(x) + int(y)) % P


def _mod_subtract(x: Element, y: Element) -> Element:
    return (int(x) - int(y)) % P


def _mod_divide(x: Element, y: Element) -> Element:
    if int(y) == 0:
        raise ZeroDivisionError("division is defined only for nonzero divisors")
    return (int(x) * pow(int(y), -1, P)) % P


def _mod_hybrid(x: Element, y: Element) -> Element:
    x_int, y_int = int(x), int(y)
    if y_int % 2:
        return _mod_divide(x_int, y_int)
    return (x_int - y_int) % P


def _mod_squares_sum(x: Element, y: Element) -> Element:
    x_int, y_int = int(x), int(y)
    return (x_int * x_int + y_int * y_int) % P


def _mod_quadratic(x: Element, y: Element) -> Element:
    x_int, y_int = int(x), int(y)
    return (x_int * x_int + x_int * y_int + y_int * y_int) % P


def _mod_quadratic_plus_x(x: Element, y: Element) -> Element:
    x_int, y_int = int(x), int(y)
    return (x_int * x_int + x_int * y_int + y_int * y_int + x_int) % P


def _mod_cubic_xy(x: Element, y: Element) -> Element:
    x_int, y_int = int(x), int(y)
    return (x_int**3 + x_int * y_int) % P


def _mod_cubic_xy2_y(x: Element, y: Element) -> Element:
    x_int, y_int = int(x), int(y)
    return (x_int**3 + x_int * y_int * y_int + y_int) % P


def s5_product(left: Permutation5, right: Permutation5) -> Permutation5:
    """Author-code convention: ``right[left[i]]`` for each i in 0..4."""
    return tuple(right[left[i]] for i in range(5))  # type: ignore[return-value]


def s5_inverse(value: Permutation5) -> Permutation5:
    inverse = [0] * 5
    for i, image in enumerate(value):
        inverse[image] = i
    return tuple(inverse)  # type: ignore[return-value]


def _s5_product(x: Element, y: Element) -> Element:
    return s5_product(x, y)  # type: ignore[arg-type]


def _s5_conjugation(x: Element, y: Element) -> Element:
    x_perm, y_perm = x, y
    return s5_product(s5_product(x_perm, y_perm), s5_inverse(x_perm))  # type: ignore[arg-type]


def _s5_xyx(x: Element, y: Element) -> Element:
    x_perm, y_perm = x, y
    return s5_product(s5_product(x_perm, y_perm), x_perm)  # type: ignore[arg-type]


RESIDUES: tuple[int, ...] = tuple(range(P))

OPERATIONS: dict[str, Operation] = {
    "mod_add": Operation(
        "mod_add", "modular addition", "mod_add",
        r"x + y (mod 97)", "x,y in {0,...,96}", RESIDUES, _mod_add,
    ),
    "mod_subtract": Operation(
        "mod_subtract", "modular subtraction", "mod_subtract",
        r"x - y (mod 97)", "x,y in {0,...,96}", RESIDUES, _mod_subtract,
    ),
    "mod_divide": Operation(
        "mod_divide", "modular division", "mod_divide",
        r"x / y (mod 97)", "x in {0,...,96}; y in {1,...,96}", RESIDUES, _mod_divide,
    ),
    "mod_hybrid_div_sub": Operation(
        "mod_hybrid_div_sub", "hybrid division/subtraction", "mod_hybrid_div_sub",
        r"x/y (mod 97) if y is odd; otherwise x-y (mod 97)",
        "x,y in {0,...,96}", RESIDUES, _mod_hybrid,
    ),
    "mod_squares_sum": Operation(
        "mod_squares_sum", "sum of squares", "mod_squares_sum",
        r"x^2 + y^2 (mod 97)", "x,y in {0,...,96}", RESIDUES, _mod_squares_sum,
    ),
    "mod_quadratic": Operation(
        "mod_quadratic", "quadratic form", "mod_quadratic",
        r"x^2 + xy + y^2 (mod 97)", "x,y in {0,...,96}", RESIDUES, _mod_quadratic,
    ),
    "mod_quadratic_plus_x": Operation(
        "mod_quadratic_plus_x", "quadratic form plus x", "mod_quadratic_plus_x",
        r"x^2 + xy + y^2 + x (mod 97)", "x,y in {0,...,96}", RESIDUES, _mod_quadratic_plus_x,
    ),
    "mod_cubic_xy": Operation(
        "mod_cubic_xy", "cubic polynomial 1", "mod_cubic_xy",
        r"x^3 + xy (mod 97)", "x,y in {0,...,96}", RESIDUES, _mod_cubic_xy,
    ),
    "mod_cubic_xy2_y": Operation(
        "mod_cubic_xy2_y", "cubic polynomial 2", "mod_cubic_xy2_y",
        r"x^3 + xy^2 + y (mod 97)", "x,y in {0,...,96}", RESIDUES, _mod_cubic_xy2_y,
    ),
    "s5_product": Operation(
        "s5_product", "S5 product", "s5_product",
        r"x * y for x,y in S5", "x,y in S5 (120 permutations)", S5, _s5_product,
    ),
    "s5_conjugation": Operation(
        "s5_conjugation", "S5 conjugation", "s5_conjugation",
        r"x * y * x^{-1} for x,y in S5", "x,y in S5 (120 permutations)", S5, _s5_conjugation,
    ),
    "s5_xyx": Operation(
        "s5_xyx", "S5 x*y*x", "s5_xyx",
        r"x * y * x for x,y in S5", "x,y in S5 (120 permutations)", S5, _s5_xyx,
    ),
}


def element_token(value: Element) -> str:
    """Convert a residue or permutation into the paper's atomic symbol token."""
    if isinstance(value, tuple):
        return "".join(str(i) for i in value)
    return str(value)


def all_symbol_tokens() -> tuple[str, ...]:
    """The shared symbol inventory: 97 residues and all 120 S5 elements."""
    return tuple(map(str, RESIDUES)) + tuple(element_token(p) for p in S5)


def generate_equations(
    operation: str | Operation,
    *,
    shuffle: bool = True,
    seed: int = 0,
) -> list[Equation]:
    """Materialize the complete ordered-pair truth table for one operation."""
    op = OPERATIONS[operation] if isinstance(operation, str) else operation
    rows: list[Equation] = []
    for left, right in product(op.elements, repeat=2):
        if op.key == "mod_divide" and int(right) == 0:
            continue
        answer = op.apply(left, right)
        rows.append(Equation(left, op.operator_token, right, answer))
    if shuffle:
        random.Random(seed).shuffle(rows)
    return rows


def validate_operation_registry() -> None:
    """Raise AssertionError if any generated truth table violates its domain."""
    assert len(OPERATIONS) == 12
    for key, op in OPERATIONS.items():
        rows = generate_equations(op, shuffle=False)
        expected = 97 * 96 if key == "mod_divide" else len(op.elements) ** 2
        assert len(rows) == expected, (key, len(rows), expected)
        for row in rows:
            assert row.operator == op.operator_token
            assert row.answer in op.elements, (key, row)
