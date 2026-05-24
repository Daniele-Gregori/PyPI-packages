"""Tests for binary_search_from and binary_search_from_by.

Translated from BinarySearchFrom-tests.wlt (Wolfram Language) with
0-indexed adjustments: WL 1-indexed position p → Python index p-1,
WL positive start s → Python start s-1, negative starts unchanged.
"""

import random
import time
from fractions import Fraction

import pytest
from gallop_search import binary_search_from, binary_search_from_by


# ===================================================================
# Helpers
# ===================================================================

def wl_range(*args):
    """Wolfram Range[n], Range[a,b], Range[a,b,step]."""
    if len(args) == 1:
        return list(range(1, args[0] + 1))
    elif len(args) == 2:
        return list(range(args[0], args[1] + 1))
    else:
        return list(range(args[0], args[1] + 1, args[2]))


# ===================================================================
# 1. Search from beginning (default start=0)
# ===================================================================

class TestSearchFromBeginning:
    def test_find_first(self):
        assert binary_search_from([10, 20, 30, 40, 50], 10) == 0

    def test_find_second(self):
        assert binary_search_from([10, 20, 30, 40, 50], 20) == 1

    def test_find_middle(self):
        assert binary_search_from([10, 20, 30, 40, 50], 30) == 2

    def test_find_fourth(self):
        assert binary_search_from([10, 20, 30, 40, 50], 40) == 3

    def test_find_last(self):
        assert binary_search_from([10, 20, 30, 40, 50], 50) == 4

    def test_two_elem_first(self):
        assert binary_search_from([1, 2], 1) == 0

    def test_two_elem_second(self):
        assert binary_search_from([1, 2], 2) == 1

    @pytest.mark.parametrize("elem,expected", [
        (1, 0), (2, 1), (3, 2), (5, 4), (10, 9),
        (50, 49), (99, 98), (100, 99),
    ])
    def test_range100(self, elem, expected):
        assert binary_search_from(wl_range(100), elem) == expected

    @pytest.mark.parametrize("elem", [4, 8, 16, 32, 64, 128, 256, 512, 1024])
    def test_powers_of_two_boundary(self, elem):
        assert binary_search_from(wl_range(1024), elem) == elem - 1

    def test_not_found_too_large(self):
        assert binary_search_from([10, 20, 30, 40, 50], 99) is None

    def test_not_found_between(self):
        assert binary_search_from([10, 20, 30, 40, 50], 25) is None

    def test_not_found_too_small(self):
        assert binary_search_from([10, 20, 30, 40, 50], 1) is None

    def test_odd_in_even_list(self):
        assert binary_search_from(wl_range(2, 200, 2), 51) is None

    @pytest.mark.parametrize("elem,expected", [
        (1, 0), (5000, 4999), (10000, 9999), (9999, 9998),
    ])
    def test_range10k(self, elem, expected):
        assert binary_search_from(wl_range(10000), elem) == expected

    def test_float_elements(self):
        assert binary_search_from([1.5, 2.5, 3.5, 4.5, 5.5], 3.5) == 2

    def test_rational_elements(self):
        seq = [Fraction(1, 3), Fraction(2, 3), Fraction(1, 1),
               Fraction(4, 3), Fraction(5, 3)]
        assert binary_search_from(seq, 1) == 2

    def test_negative_elements(self):
        assert binary_search_from([-50, -40, -30, -20, -10], -30) == 2

    def test_mixed_sign_elements(self):
        assert binary_search_from([-2, -1, 0, 1, 2], 0) == 2


# ===================================================================
# 2. Forward search (start < target)
# ===================================================================

class TestForwardSearch:
    @pytest.mark.parametrize("elem,start,expected", [
        (10, 0, 9),
        (50, 0, 49),
        (100, 0, 99),
        (50, 29, 49),
        (31, 29, 30),
        (100, 29, 99),
        (51, 49, 50),
        (75, 49, 74),
        (99, 49, 98),
        (95, 89, 94),
        (50, 48, 49),
        (50, 47, 49),
    ])
    def test_range100(self, elem, start, expected):
        assert binary_search_from(wl_range(100), elem, start=start) == expected

    @pytest.mark.parametrize("elem,start,expected", [
        (5000, 0, 4999),
        (5000, 3999, 4999),
        (5000, 4998, 4999),
        (10000, 0, 9999),
        (10000, 9989, 9999),
    ])
    def test_range10k(self, elem, start, expected):
        assert binary_search_from(wl_range(10000), elem, start=start) == expected

    def test_element_at_start_1(self):
        assert binary_search_from(wl_range(100), 1, start=0) == 0

    def test_element_at_start_50(self):
        assert binary_search_from(wl_range(100), 50, start=49) == 49

    def test_element_at_start_100(self):
        assert binary_search_from(wl_range(100), 100, start=99) == 99

    def test_forward_odd_in_even(self):
        assert binary_search_from(wl_range(2, 100, 2), 51, start=0) is None

    def test_forward_too_large(self):
        assert binary_search_from(wl_range(2, 100, 2), 200, start=0) is None


# ===================================================================
# 3. Backward search (start > target)
# ===================================================================

class TestBackwardSearch:
    @pytest.mark.parametrize("elem,start,expected", [
        (1, 99, 0),
        (50, 99, 49),
        (99, 99, 98),
        (1, 49, 0),
        (25, 49, 24),
        (49, 49, 48),
        (30, 69, 29),
        (1, 69, 0),
        (69, 69, 68),
        (50, 50, 49),
        (50, 51, 49),
    ])
    def test_range100(self, elem, start, expected):
        assert binary_search_from(wl_range(100), elem, start=start) == expected

    @pytest.mark.parametrize("elem,start,expected", [
        (5000, 9999, 4999),
        (5000, 5999, 4999),
        (5000, 5000, 4999),
        (1, 9999, 0),
        (1, 99, 0),
    ])
    def test_range10k(self, elem, start, expected):
        assert binary_search_from(wl_range(10000), elem, start=start) == expected

    def test_backward_odd_in_even(self):
        assert binary_search_from(wl_range(2, 100, 2), 51, start=49) is None

    def test_backward_too_small(self):
        assert binary_search_from(wl_range(2, 100, 2), 1, start=49) is None


# ===================================================================
# 4. Negative indices
# ===================================================================

class TestNegativeIndices:
    def test_neg1_find_last(self):
        assert binary_search_from(wl_range(100), 100, start=-1) == 99

    def test_neg1_find_99_backward(self):
        assert binary_search_from(wl_range(100), 99, start=-1) == 98

    def test_neg1_find_50_backward(self):
        assert binary_search_from(wl_range(100), 50, start=-1) == 49

    def test_neg100_find_first(self):
        assert binary_search_from(wl_range(100), 1, start=-100) == 0

    def test_neg100_find_50_forward(self):
        assert binary_search_from(wl_range(100), 50, start=-100) == 49

    def test_neg50_find_at_51(self):
        assert binary_search_from(wl_range(100), 51, start=-50) == 50

    def test_neg50_find_75_forward(self):
        assert binary_search_from(wl_range(100), 75, start=-50) == 74

    def test_neg50_find_25_backward(self):
        assert binary_search_from(wl_range(100), 25, start=-50) == 24


# ===================================================================
# 5. Error handling
# ===================================================================

class TestErrorHandling:
    def test_start_beyond_length(self):
        with pytest.raises(IndexError):
            binary_search_from(wl_range(10), 5, start=10)

    def test_start_far_beyond_length(self):
        with pytest.raises(IndexError):
            binary_search_from(wl_range(10), 5, start=999)

    def test_start_neg_beyond_length(self):
        with pytest.raises(IndexError):
            binary_search_from(wl_range(10), 5, start=-11)

    def test_start_neg_far_beyond(self):
        with pytest.raises(IndexError):
            binary_search_from(wl_range(10), 5, start=-100)


# ===================================================================
# 6. Edge cases
# ===================================================================

class TestEdgeCases:
    def test_single_element_found(self):
        assert binary_search_from([42], 42) == 0

    def test_three_elem_first(self):
        assert binary_search_from([10, 20, 30], 10) == 0

    def test_three_elem_middle(self):
        assert binary_search_from([10, 20, 30], 20) == 1

    def test_three_elem_last(self):
        assert binary_search_from([10, 20, 30], 30) == 2

    def test_three_elem_not_found(self):
        assert binary_search_from([10, 20, 30], 15) is None

    def test_single_elem_start0_found(self):
        assert binary_search_from([42], 42, start=0) == 0

    def test_single_elem_start0_not_found(self):
        assert binary_search_from([42], 99, start=0) is None

    def test_two_elem_start0_find_at0(self):
        assert binary_search_from([10, 20], 10, start=0) == 0

    def test_two_elem_start0_find_at1(self):
        assert binary_search_from([10, 20], 20, start=0) == 1

    def test_two_elem_start1_find_at1(self):
        assert binary_search_from([10, 20], 20, start=1) == 1

    def test_two_elem_start1_find_at0(self):
        assert binary_search_from([10, 20], 10, start=1) == 0

    def test_duplicates_valid_position_for_3(self):
        result = binary_search_from([1, 1, 2, 2, 3, 3, 4, 4, 5, 5], 3)
        assert result in {4, 5}

    def test_duplicates_find3_from_start0(self):
        result = binary_search_from([1, 1, 2, 2, 3, 3, 4, 4, 5, 5], 3, start=0)
        assert result in {4, 5}

    def test_large_gaps(self):
        assert binary_search_from([1, 1000, 2000, 3000, 4000], 3000) == 3

    def test_all_same_from_beginning(self):
        assert binary_search_from([7] * 20, 7) == 0

    def test_all_same_from_middle(self):
        result = binary_search_from([7] * 20, 7, start=9)
        assert 0 <= result <= 19

    def test_100k_find_first(self):
        assert binary_search_from(wl_range(100000), 1) == 0

    def test_100k_find_middle(self):
        assert binary_search_from(wl_range(100000), 50000) == 49999

    def test_100k_find_last(self):
        assert binary_search_from(wl_range(100000), 100000) == 99999

    def test_100k_start0_find_99999(self):
        assert binary_search_from(wl_range(100000), 99999, start=0) == 99998

    def test_100k_start_last_find_1(self):
        assert binary_search_from(wl_range(100000), 1, start=99999) == 0

    def test_empty_list(self):
        assert binary_search_from([], 5) is None

    def test_tuple_sequence(self):
        assert binary_search_from(tuple(range(1, 101)), 50) == 49


# ===================================================================
# 7. Consistency across start positions
# ===================================================================

class TestConsistency:
    def test_all_50_starts_find_25(self):
        lst = wl_range(50)
        results = {binary_search_from(lst, 25, start=s) for s in range(50)}
        assert results == {24}

    def test_all_starts_find_first(self):
        lst = wl_range(50)
        results = [binary_search_from(lst, 1, start=s) for s in range(50)]
        assert all(r == 0 for r in results)

    def test_all_starts_find_last(self):
        lst = wl_range(50)
        results = [binary_search_from(lst, 50, start=s) for s in range(50)]
        assert all(r == 49 for r in results)


# ===================================================================
# 8. Default start=0 agrees with explicit start=0
# ===================================================================

class TestAgreement:
    def test_all_elements_default_vs_start0(self):
        lst = wl_range(100)
        for e in lst:
            assert binary_search_from(lst, e) == binary_search_from(lst, e, start=0)

    def test_even_list_default_vs_start0(self):
        lst = wl_range(2, 200, 2)
        for e in lst:
            assert binary_search_from(lst, e) == binary_search_from(lst, e, start=0)


# ===================================================================
# 9. Exhaustive correctness
# ===================================================================

class TestExhaustive:
    def test_all_30_elements_default(self):
        lst = wl_range(30)
        for e in range(1, 31):
            assert binary_search_from(lst, e) == e - 1

    def test_all_900_pairs(self):
        lst = wl_range(30)
        for e in range(1, 31):
            for s in range(30):
                result = binary_search_from(lst, e, start=s)
                assert result == e - 1, f"elem={e}, start={s}: got {result}"

    def test_non_contiguous_all_pairs(self):
        lst = [10 * i for i in range(1, 21)]
        for e in lst:
            expected = lst.index(e)
            for s in range(len(lst)):
                result = binary_search_from(lst, e, start=s)
                assert result == expected, f"elem={e}, start={s}: got {result}"


# ===================================================================
# 10. Performance sanity check
# ===================================================================

class TestPerformance:
    def test_1m_from_beginning(self):
        assert binary_search_from(wl_range(1000000), 999990) == 999989

    def test_1m_from_nearby_hint(self):
        assert binary_search_from(
            wl_range(1000000), 999990, start=999978
        ) == 999989

    def test_nearby_hint_faster(self):
        lst = wl_range(1000000)
        t0 = time.perf_counter()
        for _ in range(100):
            binary_search_from(lst, 999990)
        t_begin = time.perf_counter() - t0

        t0 = time.perf_counter()
        for _ in range(100):
            binary_search_from(lst, 999990, start=999978)
        t_hint = time.perf_counter() - t0

        assert t_hint < t_begin or t_hint < 0.001


# ===================================================================
# 11. Bug regression tests
# ===================================================================

class TestBugRegression:
    @pytest.mark.parametrize("start", [2, 6, 14, 30])
    def test_bug_a_backward_find_first(self, start):
        assert binary_search_from(wl_range(50), 1, start=start) == 0

    def test_bug_b_neg_start_neg1(self):
        assert binary_search_from(wl_range(10), 10, start=-1) == 9

    def test_bug_b_neg_start_neg10(self):
        assert binary_search_from(wl_range(10), 1, start=-10) == 0

    def test_bug_b_neg_start_neg5(self):
        assert binary_search_from(wl_range(10), 6, start=-5) == 5

    def test_bug_c_three_elem_last(self):
        assert binary_search_from([10, 20, 30], 30) == 2

    def test_bug_c_four_elem_last(self):
        assert binary_search_from([1, 2, 3, 4], 4) == 3

    def test_bug_c_four_elem_third(self):
        assert binary_search_from([1, 2, 3, 4], 3) == 2


# ===================================================================
# 12. Random stress test
# ===================================================================

class TestRandomStress:
    def test_random_10k_queries(self):
        rng = random.Random(42)
        for _ in range(10_000):
            n = rng.randint(1, 500)
            seq = sorted(set(rng.randint(-10000, 10000) for _ in range(n)))
            n = len(seq)
            idx = rng.randint(0, n - 1)
            value = seq[idx]
            start = rng.randint(0, n - 1)
            result = binary_search_from(seq, value, start=start)
            assert result is not None and seq[result] == value, (
                f"n={n}, value={value}, start={start}: got {result}"
            )


# ===================================================================
# binary_search_from_by — basic
# ===================================================================

class TestSearchFromBy:
    def test_basic_dict_key(self):
        data = [{"v": 10}, {"v": 20}, {"v": 30}, {"v": 40}, {"v": 50}]
        assert binary_search_from_by(data, 30, key=lambda d: d["v"]) == 2

    def test_first_element(self):
        data = [{"v": 10}, {"v": 20}, {"v": 30}]
        assert binary_search_from_by(data, 10, key=lambda d: d["v"]) == 0

    def test_last_element(self):
        data = [{"v": 10}, {"v": 20}, {"v": 30}]
        assert binary_search_from_by(data, 30, key=lambda d: d["v"]) == 2

    def test_not_found(self):
        data = [{"v": 10}, {"v": 20}, {"v": 30}]
        assert binary_search_from_by(data, 25, key=lambda d: d["v"]) is None

    def test_empty(self):
        assert binary_search_from_by([], 5, key=lambda x: x) is None

    def test_single_found(self):
        assert binary_search_from_by([{"v": 5}], 5, key=lambda d: d["v"]) == 0

    def test_single_not_found(self):
        assert binary_search_from_by([{"v": 5}], 9, key=lambda d: d["v"]) is None

    def test_tuple_key(self):
        data = [(1, "a"), (2, "b"), (3, "c"), (4, "d"), (5, "e")]
        assert binary_search_from_by(data, 3, key=lambda t: t[0]) == 2

    def test_object_attribute_key(self):
        class Obj:
            def __init__(self, x):
                self.x = x
        data = [Obj(10), Obj(20), Obj(30), Obj(40)]
        assert binary_search_from_by(data, 30, key=lambda o: o.x) == 2

    def test_string_length_key(self):
        data = ["a", "bb", "ccc", "dddd", "eeeee"]
        assert binary_search_from_by(data, 3, key=len) == 2

    def test_identity_key(self):
        seq = [10, 20, 30, 40, 50]
        assert binary_search_from_by(seq, 30, key=lambda x: x) == 2


# ===================================================================
# binary_search_from_by — forward from hint
# ===================================================================

class TestSearchFromByForward:
    def test_near_target(self):
        data = [(i, chr(65 + i % 26)) for i in range(100)]
        assert binary_search_from_by(data, 50, key=lambda t: t[0], start=48) == 50

    def test_at_target(self):
        data = [(i, chr(65 + i % 26)) for i in range(100)]
        assert binary_search_from_by(data, 50, key=lambda t: t[0], start=50) == 50

    def test_from_beginning(self):
        data = [(i,) for i in range(100)]
        assert binary_search_from_by(data, 99, key=lambda t: t[0], start=0) == 99

    def test_not_found(self):
        data = [(i * 2,) for i in range(50)]
        assert binary_search_from_by(data, 51, key=lambda t: t[0], start=0) is None


# ===================================================================
# binary_search_from_by — backward from hint
# ===================================================================

class TestSearchFromByBackward:
    def test_near_target(self):
        data = [(i,) for i in range(100)]
        assert binary_search_from_by(data, 50, key=lambda t: t[0], start=52) == 50

    def test_from_end(self):
        data = [(i,) for i in range(100)]
        assert binary_search_from_by(data, 0, key=lambda t: t[0], start=99) == 0

    def test_one_after(self):
        data = [(i,) for i in range(100)]
        assert binary_search_from_by(data, 50, key=lambda t: t[0], start=51) == 50

    def test_not_found(self):
        data = [(i * 2,) for i in range(50)]
        assert binary_search_from_by(data, 51, key=lambda t: t[0], start=49) is None


# ===================================================================
# binary_search_from_by — negative indices
# ===================================================================

class TestSearchFromByNegative:
    def test_neg1(self):
        data = [(i,) for i in range(100)]
        assert binary_search_from_by(data, 99, key=lambda t: t[0], start=-1) == 99

    def test_neg1_backward(self):
        data = [(i,) for i in range(100)]
        assert binary_search_from_by(data, 50, key=lambda t: t[0], start=-1) == 50

    def test_neg_full(self):
        data = [(i,) for i in range(100)]
        assert binary_search_from_by(data, 0, key=lambda t: t[0], start=-100) == 0


# ===================================================================
# binary_search_from_by — error handling
# ===================================================================

class TestSearchFromByErrors:
    def test_start_beyond(self):
        with pytest.raises(IndexError):
            binary_search_from_by([(i,) for i in range(10)], 5,
                                  key=lambda t: t[0], start=10)

    def test_start_neg_beyond(self):
        with pytest.raises(IndexError):
            binary_search_from_by([(i,) for i in range(10)], 5,
                                  key=lambda t: t[0], start=-11)


# ===================================================================
# binary_search_from_by — exhaustive correctness
# ===================================================================

class TestSearchFromByExhaustive:
    def test_all_elements_all_starts(self):
        data = [(i,) for i in range(30)]
        for value in range(30):
            for start in range(30):
                result = binary_search_from_by(
                    data, value, key=lambda t: t[0], start=start
                )
                assert result == value, (
                    f"value={value}, start={start}: got {result}"
                )

    def test_non_contiguous_all_pairs(self):
        data = [(i * 10,) for i in range(1, 21)]
        for idx, elem in enumerate(data):
            for start in range(len(data)):
                result = binary_search_from_by(
                    data, elem[0], key=lambda t: t[0], start=start
                )
                assert result == idx, (
                    f"elem={elem[0]}, start={start}: got {result}"
                )

    def test_random_stress_by(self):
        rng = random.Random(99)
        for _ in range(5_000):
            n = rng.randint(1, 200)
            raw = sorted(set(rng.randint(-5000, 5000) for _ in range(n)))
            data = [(v,) for v in raw]
            n = len(data)
            idx = rng.randint(0, n - 1)
            target = raw[idx]
            start = rng.randint(0, n - 1)
            result = binary_search_from_by(
                data, target, key=lambda t: t[0], start=start
            )
            assert result is not None and data[result][0] == target


# ===================================================================
# binary_search_from_by — consistency with binary_search_from
# ===================================================================

class TestByConsistency:
    def test_identity_key_matches_plain(self):
        seq = wl_range(100)
        for e in seq:
            plain = binary_search_from(seq, e)
            by_result = binary_search_from_by(seq, e, key=lambda x: x)
            assert plain == by_result

    def test_identity_key_matches_with_start(self):
        seq = wl_range(50)
        for e in seq:
            for s in range(len(seq)):
                plain = binary_search_from(seq, e, start=s)
                by_result = binary_search_from_by(seq, e, key=lambda x: x, start=s)
                assert plain == by_result, (
                    f"elem={e}, start={s}: plain={plain}, by={by_result}"
                )


# ===================================================================
# binary_search_from_by — performance
# ===================================================================

class TestSearchFromByPerformance:
    def test_hint_faster_than_beginning(self):
        data = [(i,) for i in range(1_000_000)]
        target = 999990
        key_fn = lambda t: t[0]

        t0 = time.perf_counter()
        for _ in range(100):
            binary_search_from_by(data, target, key=key_fn)
        t_begin = time.perf_counter() - t0

        t0 = time.perf_counter()
        for _ in range(100):
            binary_search_from_by(data, target, key=key_fn, start=999980)
        t_hint = time.perf_counter() - t0

        assert t_hint < t_begin or t_hint < 0.001
