import math

from trasr.judge import cohen_kappa


def test_kappa_perfect_and_chance():
    assert cohen_kappa([True, False, True, False], [True, False, True, False]) == 1.0
    assert abs(cohen_kappa([True, True, False, False], [True, False, True, False])) < 1e-9


def test_kappa_all_same_label():
    assert math.isnan(cohen_kappa([True, True], [True, True]))
