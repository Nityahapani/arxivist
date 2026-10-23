from grokking_repro.operations import (
    OPERATIONS,
    P,
    S5,
    generate_equations,
    s5_inverse,
    s5_product,
    validate_operation_registry,
)


def test_registry_has_all_twelve_paper_operations():
    assert len(OPERATIONS) == 12
    validate_operation_registry()


def test_modular_examples_match_appendix_formulas():
    assert OPERATIONS["mod_add"].apply(96, 2) == 1
    assert OPERATIONS["mod_subtract"].apply(3, 5) == 95
    assert OPERATIONS["mod_divide"].apply(12, 3) == 4
    assert OPERATIONS["mod_divide"].apply(0, 7) == 0
    assert OPERATIONS["mod_hybrid_div_sub"].apply(12, 3) == 4
    assert OPERATIONS["mod_hybrid_div_sub"].apply(12, 2) == 10
    assert OPERATIONS["mod_squares_sum"].apply(3, 4) == 25
    assert OPERATIONS["mod_quadratic"].apply(2, 3) == 19
    assert OPERATIONS["mod_quadratic_plus_x"].apply(2, 3) == 21
    assert OPERATIONS["mod_cubic_xy"].apply(2, 3) == 14
    assert OPERATIONS["mod_cubic_xy2_y"].apply(2, 3) == 29
    assert P == 97


def test_table_sizes_and_division_domain():
    assert len(generate_equations("mod_add", shuffle=False)) == 97**2
    division = generate_equations("mod_divide", shuffle=False)
    assert len(division) == 97 * 96
    assert all(row.right != 0 for row in division)
    assert len(generate_equations("s5_product", shuffle=False)) == 120**2
    assert all(row.answer in S5 for row in generate_equations("s5_product", shuffle=False))


def test_s5_product_is_a_group_operation():
    identity = tuple(range(5))
    for item in (S5[0], S5[17], S5[73], S5[-1]):
        assert s5_product(identity, item) == item
        assert s5_product(item, identity) == item
        assert s5_product(item, s5_inverse(item)) == identity
        assert s5_product(s5_inverse(item), item) == identity
    for a, b, c in ((S5[7], S5[31], S5[99]), (S5[0], S5[44], S5[13])):
        assert s5_product(s5_product(a, b), c) == s5_product(a, s5_product(b, c))
