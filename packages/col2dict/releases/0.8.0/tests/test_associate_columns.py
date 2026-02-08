"""
Comprehensive test suite for col2dict.associate_columns.

Translates the 78 Wolfram Language VerificationTests from AssociateColumns-tests.wlt.
"""

import warnings
from statistics import mean

import pandas as pd
import pytest

from col2dict import associate_columns

# ---------------------------------------------------------------------------
# Test Data
# ---------------------------------------------------------------------------

@pytest.fixture
def tab_simple():
    return pd.DataFrame({
        "Name": ["Alice", "Bob", "Carol"],
        "Age": [30, 25, 35],
        "City": ["NYC", "LA", "Chicago"],
    })

@pytest.fixture
def tab_dup():
    return pd.DataFrame({
        "Year": [2020, 2020, 2021, 2021, 2022],
        "Name": ["Alice", "Bob", "Carol", "Dave", "Eve"],
        "Score": [90, 85, 92, 88, 95],
    })

@pytest.fixture
def tab_single():
    return pd.DataFrame({"A": [1], "B": [2], "C": [3]})

@pytest.fixture
def tab_multi():
    return pd.DataFrame({
        "Dept": ["Eng", "Eng", "Eng", "Sales", "Sales"],
        "Team": ["Alpha", "Alpha", "Beta", "Gamma", "Gamma"],
        "Person": ["Alice", "Bob", "Carol", "Dave", "Eve"],
        "Role": ["Lead", "Dev", "Dev", "Lead", "Rep"],
        "Salary": [100, 80, 85, 95, 70],
    })

@pytest.fixture
def tab_all_same():
    return pd.DataFrame({"K": [1, 1, 1], "V": ["a", "b", "c"]})

@pytest.fixture
def tab_nest_dup():
    return pd.DataFrame({
        "A": [1, 1, 1, 2],
        "B": ["x", "x", "y", "z"],
        "C": [100, 200, 300, 400],
    })

@pytest.fixture
def tab_dup_multi_key():
    return pd.DataFrame({
        "A": [1, 1, 2],
        "B": ["x", "x", "y"],
        "C": [10, 20, 30],
    })

@pytest.fixture
def tab_orders():
    return pd.DataFrame({
        "Store": ["A", "A", "A", "B", "B", "B"],
        "Dept": ["Elec", "Elec", "Food", "Elec", "Elec", "Food"],
        "Item": ["TV", "Radio", "Milk", "TV", "TV", "Bread"],
        "Brand": ["Sony", "Bose", "Org", "LG", "Sony", "Local"],
        "Price": [500, 100, 3, 450, 500, 2],
        "Qty": [2, 5, 50, 3, 1, 100],
    })


# ========================================================================
# GROUP 1: Basic two-column association (no duplicate keys)
# ========================================================================

class TestBasicTwoColumn:
    def test_name_to_age(self, tab_simple):
        assert associate_columns(tab_simple, ("Name", "Age")) == {
            "Alice": 30, "Bob": 25, "Carol": 35
        }

    def test_name_to_city(self, tab_simple):
        assert associate_columns(tab_simple, ("Name", "City")) == {
            "Alice": "NYC", "Bob": "LA", "Carol": "Chicago"
        }

    def test_reversed(self, tab_simple):
        assert associate_columns(tab_simple, ("Age", "Name")) == {
            30: "Alice", 25: "Bob", 35: "Carol"
        }

    def test_single_row(self, tab_single):
        assert associate_columns(tab_single, ("A", "B")) == {1: 2}


# ========================================================================
# GROUP 2: Two-column with duplicate keys + merging functions
# ========================================================================

class TestDuplicatesMerge:
    def test_merge_sorted(self, tab_dup):
        assert associate_columns(tab_dup, ("Year", "Name"), merge=sorted) == {
            2020: ["Alice", "Bob"], 2021: ["Carol", "Dave"], 2022: ["Eve"]
        }

    def test_merge_total(self, tab_dup):
        assert associate_columns(tab_dup, ("Year", "Score"), merge=sum) == {
            2020: 175, 2021: 180, 2022: 95
        }

    def test_merge_max(self, tab_dup):
        assert associate_columns(tab_dup, ("Year", "Score"), merge=max) == {
            2020: 90, 2021: 92, 2022: 95
        }

    def test_merge_min(self, tab_dup):
        assert associate_columns(tab_dup, ("Year", "Score"), merge=min) == {
            2020: 85, 2021: 88, 2022: 95
        }

    def test_merge_mean(self, tab_dup):
        result = associate_columns(tab_dup, ("Year", "Score"), merge=lambda v: mean(v))
        assert result == {2020: 87.5, 2021: 90, 2022: 95}

    def test_merge_first(self, tab_dup):
        assert associate_columns(tab_dup, ("Year", "Name"), merge=lambda v: v[0]) == {
            2020: "Alice", 2021: "Carol", 2022: "Eve"
        }

    def test_merge_last(self, tab_dup):
        assert associate_columns(tab_dup, ("Year", "Name"), merge=lambda v: v[-1]) == {
            2020: "Bob", 2021: "Dave", 2022: "Eve"
        }

    def test_merge_length(self, tab_dup):
        assert associate_columns(tab_dup, ("Year", "Name"), merge=len) == {
            2020: 2, 2021: 2, 2022: 1
        }

    def test_merge_identity(self, tab_dup):
        assert associate_columns(tab_dup, ("Year", "Name"), merge=list) == {
            2020: ["Alice", "Bob"], 2021: ["Carol", "Dave"], 2022: ["Eve"]
        }


# ========================================================================
# GROUP 3: Three-column nested association
# ========================================================================

class TestNestedThreeColumn:
    def test_three_col(self, tab_dup):
        assert associate_columns(tab_dup, ("Year", "Name", "Score")) == {
            2020: {"Alice": 90, "Bob": 85},
            2021: {"Carol": 92, "Dave": 88},
            2022: {"Eve": 95},
        }

    def test_three_col_no_dups(self, tab_simple):
        assert associate_columns(tab_simple, ("City", "Name", "Age")) == {
            "NYC": {"Alice": 30},
            "LA": {"Bob": 25},
            "Chicago": {"Carol": 35},
        }

    def test_three_col_single_row(self, tab_single):
        assert associate_columns(tab_single, ("A", "B", "C")) == {
            1: {2: 3}
        }


# ========================================================================
# GROUP 4: Multiple columns as keys or values
# ========================================================================

class TestMultiKeyMultiValue:
    def test_multi_value(self, tab_simple):
        assert associate_columns(tab_simple, ("Name", ["Age", "City"])) == {
            "Alice": [30, "NYC"], "Bob": [25, "LA"], "Carol": [35, "Chicago"]
        }

    def test_multi_key(self, tab_dup):
        result = associate_columns(tab_dup, (["Year", "Name"], "Score"))
        assert result == {
            (2020, "Alice"): 90, (2020, "Bob"): 85,
            (2021, "Carol"): 92, (2021, "Dave"): 88,
            (2022, "Eve"): 95,
        }

    def test_multi_key_dup_merge(self, tab_dup_multi_key):
        result = associate_columns(
            tab_dup_multi_key, (["A", "B"], "C"), merge=sum
        )
        assert result == {(1, "x"): 30, (2, "y"): 30}

    def test_multi_key_dup_identity(self, tab_dup_multi_key):
        result = associate_columns(
            tab_dup_multi_key, (["A", "B"], "C"), merge=list
        )
        assert result == {(1, "x"): [10, 20], (2, "y"): [30]}


# ========================================================================
# GROUP 5: Deep nesting (4+ columns)
# ========================================================================

class TestDeepNesting:
    def test_four_col(self, tab_multi):
        result = associate_columns(tab_multi, ("Dept", "Team", "Person", "Role"))
        assert result == {
            "Eng": {
                "Alpha": {"Alice": "Lead", "Bob": "Dev"},
                "Beta": {"Carol": "Dev"},
            },
            "Sales": {
                "Gamma": {"Dave": "Lead", "Eve": "Rep"},
            },
        }

    def test_five_col(self, tab_multi):
        result = associate_columns(
            tab_multi, ("Dept", "Team", "Person", "Role", "Salary")
        )
        assert result == {
            "Eng": {
                "Alpha": {"Alice": {"Lead": 100}, "Bob": {"Dev": 80}},
                "Beta": {"Carol": {"Dev": 85}},
            },
            "Sales": {
                "Gamma": {"Dave": {"Lead": 95}, "Eve": {"Rep": 70}},
            },
        }


# ========================================================================
# GROUP 6: Different merging functions per nesting level
# ========================================================================

class TestMergePerLevel:
    def test_sort_then_total(self, tab_nest_dup):
        result = associate_columns(
            tab_nest_dup, ("A", "B", "C"), merge=[sorted, sum]
        )
        assert result == {
            1: {"x": 300, "y": 300},
            2: {"z": 400},
        }

    def test_sort_then_identity(self, tab_nest_dup):
        result = associate_columns(
            tab_nest_dup, ("A", "B", "C"), merge=[sorted, list]
        )
        assert result == {
            1: {"x": [100, 200], "y": [300]},
            2: {"z": 400},
        }


# ========================================================================
# GROUP 7: Input types (DataFrame and list of dicts)
# ========================================================================

class TestInputTypes:
    def test_list_of_dicts(self):
        data = [
            {"Name": "Alice", "Age": 30, "City": "NYC"},
            {"Name": "Bob", "Age": 25, "City": "LA"},
            {"Name": "Carol", "Age": 35, "City": "Chicago"},
        ]
        assert associate_columns(data, ("Name", "Age")) == {
            "Alice": 30, "Bob": 25, "Carol": 35
        }

    def test_list_equiv_dataframe(self):
        data = [
            {"Name": "Alice", "Age": 30},
            {"Name": "Bob", "Age": 25},
        ]
        df = pd.DataFrame(data)
        assert associate_columns(data, ("Name", "Age")) == associate_columns(
            df, ("Name", "Age")
        )


# ========================================================================
# GROUP 8: Duplicates warning
# ========================================================================

class TestDuplicatesWarning:
    def test_warning_emitted(self, tab_dup):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            associate_columns(tab_dup, ("Year", "Name"))
            assert len(w) == 1
            assert "Duplicate keys" in str(w[0].message)

    def test_warning_suppressed(self, tab_dup):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            associate_columns(
                tab_dup, ("Year", "Name"), duplicates_warning=False
            )
            assert len(w) == 0

    def test_no_warning_with_merge(self, tab_dup):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            associate_columns(tab_dup, ("Year", "Name"), merge=sorted)
            assert len(w) == 0

    def test_identity_equiv_no_warning(self, tab_dup):
        r1 = associate_columns(tab_dup, ("Year", "Name"), merge=list)
        r2 = associate_columns(
            tab_dup, ("Year", "Name"), duplicates_warning=False
        )
        assert r1 == r2


# ========================================================================
# GROUP 9: Structure checks
# ========================================================================

class TestStructure:
    def test_result_is_dict(self, tab_simple):
        assert isinstance(
            associate_columns(tab_simple, ("Name", "Age")), dict
        )

    def test_correct_length(self, tab_simple):
        assert len(associate_columns(tab_simple, ("Name", "Age"))) == 3

    def test_correct_keys(self, tab_simple):
        assert list(associate_columns(tab_simple, ("Name", "Age")).keys()) == [
            "Alice", "Bob", "Carol"
        ]

    def test_correct_values(self, tab_simple):
        assert list(associate_columns(tab_simple, ("Name", "Age")).values()) == [
            30, 25, 35
        ]

    def test_nested_values_are_dicts(self, tab_dup):
        result = associate_columns(tab_dup, ("Year", "Name", "Score"))
        assert all(isinstance(v, dict) for v in result.values())


# ========================================================================
# GROUP 10: Edge cases
# ========================================================================

class TestEdgeCases:
    def test_all_same_keys(self, tab_all_same):
        assert associate_columns(tab_all_same, ("K", "V"), merge=sorted) == {
            1: ["a", "b", "c"]
        }

    def test_all_same_keys_length(self, tab_all_same):
        assert associate_columns(tab_all_same, ("K", "V"), merge=len) == {
            1: 3
        }

    def test_mixed_type_keys(self):
        data = [
            {"K": 1, "V": "one"},
            {"K": "two", "V": 2},
            {"K": 3.0, "V": "three"},
        ]
        assert associate_columns(data, ("K", "V")) == {
            1: "one", "two": 2, 3.0: "three"
        }

    def test_string_keys_numeric_values(self):
        data = [{"K": "a", "V": 1}, {"K": "b", "V": 2}, {"K": "c", "V": 3}]
        assert associate_columns(data, ("K", "V")) == {"a": 1, "b": 2, "c": 3}


# ========================================================================
# GROUP 11: Multi-value at leaf in nested association
# ========================================================================

class TestMultiValueLeaf:
    def test_leaf_is_list(self, tab_multi):
        result = associate_columns(tab_multi, ("Dept", "Person", ["Role", "Salary"]))
        assert result["Eng"]["Alice"] == ["Lead", 100]

    def test_leaf_is_list_2(self, tab_multi):
        result = associate_columns(tab_multi, ("Dept", "Person", ["Role", "Salary"]))
        assert result["Sales"]["Eve"] == ["Rep", 70]


# ========================================================================
# GROUP 12: Custom merging functions
# ========================================================================

class TestCustomMerge:
    def test_string_join(self, tab_dup):
        result = associate_columns(
            tab_dup, ("Year", "Score"),
            merge=lambda v: "".join(str(x) for x in v),
        )
        assert result == {2020: "9085", 2021: "9288", 2022: "95"}

    def test_reverse(self, tab_dup):
        result = associate_columns(
            tab_dup, ("Year", "Score"), merge=lambda v: list(reversed(v))
        )
        assert result == {2020: [85, 90], 2021: [88, 92], 2022: [95]}

    def test_string_riffle(self, tab_dup):
        result = associate_columns(
            tab_dup, ("Year", "Name"), merge=lambda v: ", ".join(v)
        )
        assert result == {2020: "Alice, Bob", 2021: "Carol, Dave", 2022: "Eve"}


# ========================================================================
# GROUP 13: Nested associations with list arguments at various levels
# ========================================================================

class TestNestedMultiColumn:
    def test_n1_single_key_multi_key_single_val(self, tab_orders):
        """Store -> {Dept, Item} -> Price"""
        result = associate_columns(
            tab_orders, ("Store", ["Dept", "Item"], "Price")
        )
        assert result["A"] == {
            ("Elec", "TV"): 500,
            ("Elec", "Radio"): 100,
            ("Food", "Milk"): 3,
        }
        # Store B has duplicate (Elec, TV)
        assert result["B"][("Elec", "TV")] == [450, 500]
        assert result["B"][("Food", "Bread")] == [2]

    def test_n2_single_key_multi_key_multi_val(self, tab_orders):
        """Store -> {Dept, Item} -> {Price, Qty}"""
        result = associate_columns(
            tab_orders, ("Store", ["Dept", "Item"], ["Price", "Qty"])
        )
        assert result["A"][("Elec", "TV")] == [500, 2]
        assert result["A"][("Elec", "Radio")] == [100, 5]
        # Duplicate inner key → list of lists
        assert result["B"][("Elec", "TV")] == [[450, 3], [500, 1]]
        assert result["B"][("Food", "Bread")] == [[2, 100]]

    def test_n3_multi_key_single_col_multi_val(self, tab_orders):
        """{Store, Dept} -> Item -> {Price, Qty}"""
        result = associate_columns(
            tab_orders, (["Store", "Dept"], "Item", ["Price", "Qty"])
        )
        assert result[("A", "Elec")] == {"TV": [500, 2], "Radio": [100, 5]}
        assert result[("A", "Food")] == {"Milk": [3, 50]}
        assert result[("B", "Elec")] == {"TV": [[450, 3], [500, 1]]}
        assert result[("B", "Food")] == {"Bread": [2, 100]}

    def test_n4_multi_key_multi_key_single_val(self, tab_orders):
        """{Store, Dept} -> {Item, Brand} -> Price"""
        result = associate_columns(
            tab_orders, (["Store", "Dept"], ["Item", "Brand"], "Price")
        )
        assert result[("A", "Elec")] == {
            ("TV", "Sony"): 500, ("Radio", "Bose"): 100
        }
        assert result[("B", "Elec")] == {
            ("TV", "LG"): 450, ("TV", "Sony"): 500
        }

    def test_n5_4level_multi_in_middle(self, tab_orders):
        """Store -> {Dept, Item} -> Brand -> Price"""
        result = associate_columns(
            tab_orders, ("Store", ["Dept", "Item"], "Brand", "Price")
        )
        assert result["A"][("Elec", "TV")] == {"Sony": 500}
        assert result["B"][("Elec", "TV")] == {"LG": 450, "Sony": 500}
        assert result["B"][("Food", "Bread")] == {"Local": 2}

    def test_n6_merge_sort_identity(self, tab_orders):
        """Dept -> Item -> {Price, Qty} with merge=[sorted, list]"""
        result = associate_columns(
            tab_orders, ("Dept", "Item", ["Price", "Qty"]),
            merge=[sorted, list],
        )
        # sorted at level 1 sorts inner dict keys (Radio < TV)
        assert list(result["Elec"].keys()) == ["Radio", "TV"]
        assert result["Elec"]["Radio"] == [[100, 5]]
        # TV has 3 entries (row order from original DataFrame)
        assert len(result["Elec"]["TV"]) == 3
        assert set(map(tuple, result["Elec"]["TV"])) == {(500, 2), (450, 3), (500, 1)}
        assert result["Food"] == {
            "Bread": [2, 100],
            "Milk": [3, 50],
        }

    def test_n7_merge_identity_total(self, tab_orders):
        """{Store, Dept} -> Item -> Qty with merge=[None, sum]"""
        result = associate_columns(
            tab_orders, (["Store", "Dept"], "Item", "Qty"),
            merge=[None, sum],
        )
        assert result[("A", "Elec")] == {"TV": 2, "Radio": 5}
        assert result[("B", "Elec")] == {"TV": 4}
        assert result[("B", "Food")] == {"Bread": 100}

    def test_n8_triple_key_multi_val_total(self, tab_orders):
        """{Store, Dept, Item} -> {Price, Qty} with merge=sum (element-wise)"""
        # sum on list of lists → element-wise sum via custom merge
        def sum_elementwise(vals):
            return [sum(x) for x in zip(*vals)]
        result = associate_columns(
            tab_orders, (["Store", "Dept", "Item"], ["Price", "Qty"]),
            merge=sum_elementwise,
        )
        assert result[("A", "Elec", "TV")] == [500, 2]
        assert result[("B", "Elec", "TV")] == [950, 4]
        assert result[("B", "Food", "Bread")] == [2, 100]

    def test_n9_merge_sort_total(self, tab_orders):
        """Store -> {Dept, Item} -> Price with merge=[sorted, sum]"""
        result = associate_columns(
            tab_orders, ("Store", ["Dept", "Item"], "Price"),
            merge=[sorted, sum],
        )
        assert result["A"] == {
            ("Elec", "Radio"): 100, ("Elec", "TV"): 500, ("Food", "Milk"): 3
        }
        assert result["B"] == {("Elec", "TV"): 950, ("Food", "Bread"): 2}

    def test_n10_3level_multi_key_at_level3(self, tab_orders):
        """Store -> Dept -> {Item, Brand} -> Price"""
        result = associate_columns(
            tab_orders, ("Store", "Dept", ["Item", "Brand"], "Price")
        )
        assert result["A"]["Elec"] == {
            ("TV", "Sony"): 500, ("Radio", "Bose"): 100
        }
        assert result["B"]["Elec"] == {
            ("TV", "LG"): 450, ("TV", "Sony"): 500
        }

    def test_n11_3level_multi_key_multi_val(self, tab_orders):
        """Store -> Dept -> {Item, Brand} -> {Price, Qty}"""
        result = associate_columns(
            tab_orders, ("Store", "Dept", ["Item", "Brand"], ["Price", "Qty"])
        )
        assert result["A"]["Elec"][("TV", "Sony")] == [500, 2]
        assert result["B"]["Elec"][("TV", "LG")] == [450, 3]
        assert result["B"]["Food"][("Bread", "Local")] == [2, 100]

    def test_n12_multi_key_single_val_total(self, tab_orders):
        """{Store, Dept} -> Qty with merge=sum"""
        result = associate_columns(
            tab_orders, (["Store", "Dept"], "Qty"), merge=sum
        )
        assert result == {
            ("A", "Elec"): 7, ("A", "Food"): 50,
            ("B", "Elec"): 4, ("B", "Food"): 100,
        }


# ========================================================================
# GROUP 14: Error handling
# ========================================================================

class TestErrors:
    def test_invalid_input_type(self):
        with pytest.raises(TypeError):
            associate_columns("not a table", ("A", "B"))

    def test_too_few_cols(self, tab_simple):
        with pytest.raises(ValueError):
            associate_columns(tab_simple, ("Name",))

    def test_missing_column(self, tab_simple):
        with pytest.raises(KeyError):
            associate_columns(tab_simple, ("Name", "Nonexistent"))
