"""Tests for ``spreadsheet_trace``.

Direct transcription of the ``VerificationTest`` suite of the Wolfram
resource function ``SpreadsheetTrace`` (1.0.0). Test names carry the original
TestIDs; the example workbooks live in ``tests/data``.
"""

import pytest

from spreadsheet_toolkit import spreadsheet_trace


def wl_depth(expr):
    """Wolfram Language ``Depth``: 1 for atoms, 1 + max child depth for lists."""
    if not isinstance(expr, list):
        return 1
    return 1 + max((wl_depth(e) for e in expr), default=1)


def wl_length(expr):
    """Wolfram Language ``Length``: 0 for atoms (strings, numbers)."""
    return len(expr) if isinstance(expr, list) else 0


# ---------------------------------------------------------------------------
# Shared expected subtrees (file2: example_02.xlsx)
# ---------------------------------------------------------------------------

B1_TREE = ["B1", "SUM(A1:A3)", ["A1", 15.0], ["A2", 22.0], ["A3", 8.0]]
B2_TREE = ["B2", "SUM(A4:A6)", ["A4", 31.0], ["A5", 12.0], ["A6", 19.0]]
B4_TREE = ["B4", "AVERAGE(A1:A6)",
           ["A1", 15.0], ["A2", 22.0], ["A3", 8.0],
           ["A4", 31.0], ["A5", 12.0], ["A6", 19.0]]
C1_TREE = ["C1", "B1+B2", B1_TREE, B2_TREE]
C3_TREE = ["C3", "B4*2", B4_TREE]
C4_TREE = ["C4", "C1-C3", C1_TREE, C3_TREE]
C5_TREE = ["C5", "IF(C4>0,C4,0)", C4_TREE, C4_TREE]

# Shared expected subtrees (file3: example_03.xlsx)

E2_TREE = ["E2", "C2*D2", ["C2", 25.5], ["D2", 100.0]]
E3_TREE = ["E3", "C3*D3", ["C3", 42.0], ["D3", 75.0]]
E4_TREE = ["E4", "C4*D4", ["C4", 18.99], ["D4", 200.0]]
E5_TREE = ["E5", "C5*D5", ["C5", 35.0], ["D5", 150.0]]
E6_TREE = ["E6", "C6*D6", ["C6", 89.99], ["D6", 50.0]]
E7_TREE = ["E7", "C7*D7", ["C7", 120.0], ["D7", 30.0]]
E_TREES = [E2_TREE, E3_TREE, E4_TREE, E5_TREE, E6_TREE, E7_TREE]

CATEGORY_LEAVES = [["B2", "Electronics"], ["B3", "Electronics"],
                   ["B4", "Home"], ["B5", "Home"],
                   ["B6", "Industrial"], ["B7", "Industrial"]]

# Shared expected leaves (file4: example_04.xlsx)

REVENUE_LEAVES = [["B2", 12000.0], ["B3", 15000.0], ["B4", 13500.0],
                  ["B5", 16000.0], ["B6", 18000.0], ["B7", 22000.0],
                  ["B8", 25000.0], ["B9", 24000.0], ["B10", 20000.0],
                  ["B11", 17000.0], ["B12", 14000.0], ["B13", 19000.0]]


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class TestValidation:
    def test_validation_bad_cell_unevaluated(self, file1):
        with pytest.raises((TypeError, ValueError)):
            spreadsheet_trace(file1, "ZZZ")

    def test_validation_empty_string_unevaluated(self, file1):
        with pytest.raises((TypeError, ValueError)):
            spreadsheet_trace(file1, "")

    def test_validation_numeric_string_unevaluated(self, file1):
        with pytest.raises((TypeError, ValueError)):
            spreadsheet_trace(file1, 123)


# ---------------------------------------------------------------------------
# Leafs
# ---------------------------------------------------------------------------


class TestLeafs:
    def test_ex1_A1_numeric_leaf(self, file1):
        assert spreadsheet_trace(file1, "A1") == ["A1", 10.0]

    def test_ex1_A5_string_leaf(self, file1):
        assert spreadsheet_trace(file1, "A5") == ["A5", "hello"]

    def test_ex1_C3_two_branches_shared_leaf(self, file1):
        assert spreadsheet_trace(file1, "C3") == \
            ["C3", "B3-A3", ["B3", "A3^2", ["A3", 5.0]], ["A3", 5.0]]


# ---------------------------------------------------------------------------
# First level
# ---------------------------------------------------------------------------


class TestFirstLevel:
    def test_ex1_C1_two_deps(self, file1):
        assert spreadsheet_trace(file1, "C1") == \
            ["C1", "A1+B1", ["A1", 10.0], ["B1", 20.0]]

    def test_ex1_B3_single_dep(self, file1):
        assert spreadsheet_trace(file1, "B3") == ["B3", "A3^2", ["A3", 5.0]]

    def test_ex1_B5_LEN_text(self, file1):
        assert spreadsheet_trace(file1, "B5") == \
            ["B5", "LEN(A5)", ["A5", "hello"]]


# ---------------------------------------------------------------------------
# Higher levels
# ---------------------------------------------------------------------------


class TestHigherLevels:
    def test_ex1_D1_two_level_chain(self, file1):
        assert spreadsheet_trace(file1, "D1") == \
            ["D1", "C1*2", ["C1", "A1+B1", ["A1", 10.0], ["B1", 20.0]]]

    def test_ex1_E1_three_level_chain(self, file1):
        assert spreadsheet_trace(file1, "E1") == \
            ["E1", "D1+A1",
             ["D1", "C1*2", ["C1", "A1+B1", ["A1", 10.0], ["B1", 20.0]]],
             ["A1", 10.0]]

    def test_ex2_D1_deep_converging_5_levels(self, file2):
        assert spreadsheet_trace(file2, "D1") == \
            ["D1", "C1+C5", C1_TREE, C5_TREE]

    def test_ex3_B18_IF_formula_string(self, file3):
        b16_tree = (["B16", 'SUMIF(B2:B7,"Electronics",E2:E7)']
                    + CATEGORY_LEAVES + E_TREES)
        e9_tree = ["E9", "SUM(E2:E7)"] + E_TREES
        assert spreadsheet_trace(file3, "B18") == \
            ["B18", 'IF(B17>0.5,"Dominant","Minor")',
             ["B17", "B16/E9", b16_tree, e9_tree]]


# ---------------------------------------------------------------------------
# Dollar
# ---------------------------------------------------------------------------


class TestDollar:
    def test_ex3_F2_absolute_ref_leaf(self, file3):
        assert spreadsheet_trace(file3, "F2") == \
            ["F2", "E2*$B$10", E2_TREE, ["B10", 0.22]]

    def test_ex3_G2_chain_through_absolute_ref(self, file3):
        assert spreadsheet_trace(file3, "G2") == \
            ["G2", "E2+F2", E2_TREE,
             ["F2", "E2*$B$10", E2_TREE, ["B10", 0.22]]]

    def test_ex3_F9_SUM_of_absolute_ref_formulas(self, file3):
        assert spreadsheet_trace(file3, "F9") == \
            ["F9", "SUM(F2:F7)",
             ["F2", "E2*$B$10", E2_TREE, ["B10", 0.22]],
             ["F3", "E3*$B$10", E3_TREE, ["B10", 0.22]],
             ["F4", "E4*$B$10", E4_TREE, ["B10", 0.22]],
             ["F5", "E5*$B$10", E5_TREE, ["B10", 0.22]],
             ["F6", "E6*$B$10", E6_TREE, ["B10", 0.22]],
             ["F7", "E7*$B$10", E7_TREE, ["B10", 0.22]]]


# ---------------------------------------------------------------------------
# Duplicates
# ---------------------------------------------------------------------------


class TestDuplicates:
    def test_ex2_C5_IF_deep_duplicate_branches(self, file2):
        assert spreadsheet_trace(file2, "C5") == C5_TREE

    def test_ex2_C5_IF_deep_no_duplicate_branches(self, file2):
        assert spreadsheet_trace(file2, "C5", trace_duplicates=False) == \
            ["C5", "IF(C4>0,C4,0)", C4_TREE]


# ---------------------------------------------------------------------------
# Cell ranges
# ---------------------------------------------------------------------------


class TestCellRanges:
    def test_ex2_B1_SUM_range(self, file2):
        assert spreadsheet_trace(file2, "B1") == B1_TREE

    def test_ex2_B4_AVERAGE_full_range(self, file2):
        assert spreadsheet_trace(file2, "B4") == B4_TREE

    def test_ex2_B5_MAX_range(self, file2):
        assert spreadsheet_trace(file2, "B5") == \
            ["B5", "MAX(A1:A6)",
             ["A1", 15.0], ["A2", 22.0], ["A3", 8.0],
             ["A4", 31.0], ["A5", 12.0], ["A6", 19.0]]

    def test_ex2_C4_two_deep_branches(self, file2):
        assert spreadsheet_trace(file2, "C4") == C4_TREE

    def test_ex2_D3_diamond_shared_leaves(self, file2):
        assert spreadsheet_trace(file2, "D3") == \
            ["D3", "B1*B4", B1_TREE, B4_TREE]

    def test_ex3_E9_SUM_of_formulas(self, file3):
        assert spreadsheet_trace(file3, "E9") == \
            ["E9", "SUM(E2:E7)"] + E_TREES

    def test_ex3_B15_COUNTIF_string_range(self, file3):
        assert spreadsheet_trace(file3, "B15") == \
            ["B15", 'COUNTIF(B2:B7,"Electronics")'] + CATEGORY_LEAVES

    def test_ex3_B16_SUMIF_two_ranges(self, file3):
        assert spreadsheet_trace(file3, "B16") == \
            (["B16", 'SUMIF(B2:B7,"Electronics",E2:E7)']
             + CATEGORY_LEAVES + E_TREES)

    def test_ex4_D14_SUM_12_formulas_length_14(self, file4):
        assert len(spreadsheet_trace(file4, "D14")) == 14

    def test_ex4_D14_first_subtree_D2(self, file4):
        assert spreadsheet_trace(file4, "D14")[2] == \
            ["D2", "B2-C2", ["B2", 12000.0],
             ["C2", "B2*0.65", ["B2", 12000.0]]]


# ---------------------------------------------------------------------------
# Cross sheet
# ---------------------------------------------------------------------------


class TestCrossSheet:
    def test_ex4_C2_single_dep_first_sheet(self, file4):
        assert spreadsheet_trace(file4, "C2") == \
            ["C2", "B2*0.65", ["B2", 12000.0]]

    def test_compat_ex4_B14_first_sheet_default(self, file4):
        assert spreadsheet_trace(file4, "B14") == \
            ["B14", "SUM(B2:B13)"] + REVENUE_LEAVES

    def test_ex4_Summary_B3_cross_sheet_SUM(self, file4):
        assert spreadsheet_trace(file4, "Summary!B3") == \
            ["Summary!B3", "Input!B14",
             ["Input!B14", "SUM(B2:B13)"] + REVENUE_LEAVES]

    def test_ex4_Summary_B8_cross_sheet_AVERAGE_range(self, file4):
        assert spreadsheet_trace(file4, "Summary!B8") == \
            ["Summary!B8", "AVERAGE(Input!B2:Input!B13)",
             ["Input!B2", 12000.0], ["Input!B3", 15000.0],
             ["Input!B4", 13500.0], ["Input!B5", 16000.0],
             ["Input!B6", 18000.0], ["Input!B7", 22000.0],
             ["Input!B8", 25000.0], ["Input!B9", 24000.0],
             ["Input!B10", 20000.0], ["Input!B11", 17000.0],
             ["Input!B12", 14000.0], ["Input!B13", 19000.0]]

    def test_ex4_Summary_B5_subtree_has_12_formulas(self, file4):
        assert len(spreadsheet_trace(file4, "Summary!B5")[2]) == 14

    def test_ex4_Summary_B6_two_branches(self, file4):
        assert len(spreadsheet_trace(file4, "Summary!B6")) == 4

    def test_ex6_Analysis_B3_cell_range_cross_sheet(self, file6):
        assert spreadsheet_trace(file6, "Analysis!B3") == \
            ["Analysis!B3", "SUM(PnL!B3:H3)",
             ["PnL!B3", "Revenue!B8",
              ["Revenue!B8", "B5+B7",
               ["B5", "B3*B4",
                ["B3", "Assumptions!B7", ["Assumptions!B7", 1000.0]],
                ["B4", "Assumptions!B8", ["Assumptions!B8", 50.0]]],
               ["B7", 2000.0]]]]

    def test_ex9_Config_B16_sheeted_input(self, file9):
        assert spreadsheet_trace(file9, "Config!B16") == \
            ["Config!B16", "AVERAGE(B5,B6)", ["B5", 0.25], ["B6", 0.3]]

    def test_ex9_B16_plain_matches_Config(self, file9):
        assert spreadsheet_trace(file9, "B16") == \
            ["B16", "AVERAGE(B5,B6)", ["B5", 0.25], ["B6", 0.3]]

    def test_ex9_Config_B17_chain(self, file9):
        assert spreadsheet_trace(file9, "Config!B17") == \
            ["Config!B17", "1-B16",
             ["B16", "AVERAGE(B5,B6)", ["B5", 0.25], ["B6", 0.3]]]

    def test_ex10_B2_cross_sheet_leaf(self, file10):
        assert spreadsheet_trace(file10, "B2") == \
            ["B2", "Data!A1", ["Data!A1", 10.0]]

    def test_ex10_B4_cross_sheet_to_formula(self, file10):
        assert spreadsheet_trace(file10, "B4") == \
            ["B4", "Data!B1",
             ["Data!B1", "SUM(A1:A3)",
              ["A1", 10.0], ["A2", 20.0], ["A3", 30.0]]]

    def test_ex10_B5_local_to_cross_sheet(self, file10):
        assert spreadsheet_trace(file10, "B5") == \
            ["B5", "B2+B3",
             ["B2", "Data!A1", ["Data!A1", 10.0]],
             ["B3", "Data!A2", ["Data!A2", 20.0]]]

    def test_ex10_B6_mixed_deps(self, file10):
        assert spreadsheet_trace(file10, "B6") == \
            ["B6", "B2+Data!A3",
             ["B2", "Data!A1", ["Data!A1", 10.0]],
             ["Data!A3", 30.0]]


# ---------------------------------------------------------------------------
# Column ranges
# ---------------------------------------------------------------------------

PRODUCT_LEAVES = [
    ["Products!A1", "ProductID"], ["Products!A2", 101.0],
    ["Products!A3", 102.0], ["Products!A4", 103.0],
    ["Products!A5", 104.0], ["Products!A6", 105.0],
    ["Products!A7", 106.0], ["Products!A8", 107.0],
    ["Products!A9", 108.0],
    ["Products!B1", "Name"], ["Products!B2", "Laptop"],
    ["Products!B3", "Mouse"], ["Products!B4", "Desk"],
    ["Products!B5", "Chair"], ["Products!B6", "Monitor"],
    ["Products!B7", "Keyboard"], ["Products!B8", "Lamp"],
    ["Products!B9", "Headset"],
    ["Products!C1", "UnitPrice"], ["Products!C2", 999.99],
    ["Products!C3", 29.99], ["Products!C4", 249.99],
    ["Products!C5", 349.99], ["Products!C6", 449.99],
    ["Products!C7", 79.99], ["Products!C8", 59.99],
    ["Products!C9", 129.99],
]


class TestColumnRanges:
    def test_col_range_ex5_Orders_D2_length_30(self, file5):
        assert len(spreadsheet_trace(file5, "Orders!D2")) == 30

    def test_col_range_ex5_all_VLOOKUP_rows_dimension(self, file5):
        traces = [spreadsheet_trace(file5, f"Orders!D{i}") for i in range(2, 12)]
        assert [len(t) for t in traces] == [30] * 10

    def test_col_range_ex5_G2_deep_chain_explicit(self, file5):
        assert spreadsheet_trace(file5, "Orders!G2") == \
            ["Orders!G2", "E2*(1-F2)",
             ["E2", "C2*D2",
              ["C2", 2.0],
              (["D2", "VLOOKUP(B2,Products!A:C,3,FALSE)", ["B2", 101.0]]
               + PRODUCT_LEAVES)],
             ["F2", "IF(C2>=5,0.1,0)", ["C2", 2.0]]]

    def test_col_range_ex5_Dashboard_B3_depth_6(self, file5):
        assert wl_depth(spreadsheet_trace(file5, "Dashboard!B3")) == 6

    def test_col_range_ex5_Dashboard_B3_depth_list(self, file5):
        trace = spreadsheet_trace(file5, "Dashboard!B3")
        assert [wl_depth(e) for e in trace[2]] == \
            [1, 1, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4]

    def test_col_range_ex5_Dashboard_B3_depth_list_2(self, file5):
        trace = spreadsheet_trace(file5, "Dashboard!B3")
        assert [[wl_depth(e) for e in subtree] for subtree in trace[2][2:]] == \
            [[1, 1, 2, 3]] * 10

    def test_col_range_ex7_Budget_B4_len_total(self, file7):
        assert len(spreadsheet_trace(file7, "Budget!B4")) == 39

    def test_col_range_ex7_Budget_B4_len_upto_22(self, file7):
        trace = spreadsheet_trace(file7, "Budget!B4")
        assert [wl_length(e) for e in trace[:22]] == [0, 0] + [2] * 20

    def test_col_range_ex7_Budget_B4_depth_after_22(self, file7):
        trace = spreadsheet_trace(file7, "Budget!B4")
        assert [wl_depth(e) for e in trace[22:]] == [5] * 15 + [2, 6]


# ---------------------------------------------------------------------------
# Nested formulas
# ---------------------------------------------------------------------------


class TestNestedFormulas:
    def test_ex3_B19_nested_ROUND_AVERAGE(self, file3):
        assert spreadsheet_trace(file3, "B19") == \
            ["B19", "ROUND(AVERAGE(E2:E7)*1.1, 2)"] + E_TREES

    def test_ex6_B16_PMT_three_leaf_deps(self, file6):
        assert spreadsheet_trace(file6, "B16") == \
            ["B16", "PMT(B13,B14,-B12)",
             ["B13", 0.05], ["B14", 5.0], ["B12", 100000.0]]

    def test_ex6_B18_nested_ROUND_with_chain(self, file6):
        assert spreadsheet_trace(file6, "B18") == \
            ["B18", "ROUND(B10/(B8-B17),0)",
             ["B10", 5000.0], ["B8", 50.0],
             ["B17", "B9*(1+B5)", ["B9", 30.0], ["B5", 0.03]]]

    def test_ex8_B10_INDEX_MATCH_MAX_nested(self, file8):
        assert spreadsheet_trace(file8, "B10") == \
            ["B10", "INDEX(B2:B6,MATCH(MAX(E2:E6),E2:E6,0))",
             ["B2", "Acme Corp"], ["B3", "GlobalParts"], ["B4", "EastMfg"],
             ["B5", "NordicSupply"], ["B6", "MediterraneanCo"],
             ["E2", 0.95], ["E3", 0.92], ["E4", 0.88],
             ["E5", 0.97], ["E6", 0.9],
             ["E2", 0.95], ["E3", 0.92], ["E4", 0.88],
             ["E5", 0.97], ["E6", 0.9]]

    def test_col_range_ex8_Catalog_I2_nested(self, file8):
        assert spreadsheet_trace(file8, "Catalog!I2") == \
            ["Catalog!I2",
             'IFERROR(INDEX(Suppliers!D:D,MATCH(D2,Suppliers!A:A,0)),"N/A")',
             ["Suppliers!D1", "Lead Time (days)"], ["Suppliers!D2", 7.0],
             ["Suppliers!D3", 14.0], ["Suppliers!D4", 21.0],
             ["Suppliers!D5", 10.0], ["Suppliers!D6", 12.0],
             ["Suppliers!D7", ""], ["Suppliers!D8", ""],
             ["Suppliers!D9", ""], ["Suppliers!D10", ""],
             ["Suppliers!D11", ""],
             ["D2", "SUP01"],
             ["Suppliers!A1", "SupplierID"], ["Suppliers!A2", "SUP01"],
             ["Suppliers!A3", "SUP02"], ["Suppliers!A4", "SUP03"],
             ["Suppliers!A5", "SUP04"], ["Suppliers!A6", "SUP05"],
             ["Suppliers!A7", ""], ["Suppliers!A8", "Avg Lead Time"],
             ["Suppliers!A9", "Avg Reliability"],
             ["Suppliers!A10", "Best Supplier"],
             ["Suppliers!A11", "Gold Suppliers"]]


# ---------------------------------------------------------------------------
# Input forms (beyond the Wolfram suite)
# ---------------------------------------------------------------------------


class TestInputForms:
    def test_triple_input_matches_file_input(self, file1):
        from spreadsheet_toolkit import import_all

        book = import_all(file1)
        assert spreadsheet_trace(book, "D1") == spreadsheet_trace(file1, "D1")

    def test_out_of_range_cell_gives_empty_trace(self, file1):
        assert spreadsheet_trace(file1, "ZZ999") == []

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            spreadsheet_trace("no_such_file.xlsx", "A1")

    def test_invalid_source_raises(self):
        with pytest.raises((TypeError, ValueError)):
            spreadsheet_trace(12345, "A1")

    def test_invalid_triple_raises(self):
        with pytest.raises((TypeError, ValueError)):
            spreadsheet_trace(([1, 2], [], []), "A1")
