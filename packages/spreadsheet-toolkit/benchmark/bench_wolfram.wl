(* ::Package:: *)

(* Benchmark ResourceFunction["SpreadsheetTrace"] on the example workbooks.

   Run from anywhere:

       wolframscript -file benchmark/bench_wolfram.wl

   Methodology: for each (file, cell) case the trace runs once as a warm-up
   (which also fills the ImportOnce cache), then repeatedly for ~5 seconds
   via RepeatedTiming; the trimmed-mean wall-clock time is reported. *)

st = ResourceFunction["SpreadsheetTrace"];

dataDir = FileNameJoin[{DirectoryName[$InputFileName], "..", "tests", "data"}];

cases = {
    {"example_01.xlsx", "D1"},
    {"example_01.xlsx", "E1"},
    {"example_02.xlsx", "C5"},
    {"example_02.xlsx", "D1"},
    {"example_03.xlsx", "B18"},
    {"example_03.xlsx", "F9"},
    {"example_04.xlsx", "Summary!B3"},
    {"example_05.xlsx", "Orders!D2"},
    {"example_05.xlsx", "Dashboard!B3"},
    {"example_07.xlsx", "Budget!B4"},
    {"example_08.xlsx", "Catalog!I2"}
};

results = Table[
    Block[{file = FileNameJoin[{dataDir, case[[1]]}], cell = case[[2]], timing},
        st[file, cell]; (* warm-up: also fills the ImportOnce cache *)
        timing = First@RepeatedTiming[st[file, cell], 5];
        <|"file" -> case[[1]], "cell" -> cell,
          "mean_ms" -> Round[timing*1000, 0.001]|>],
    {case, cases}];

Print@ExportString[
    <|"wolfram" -> $Version, "results" -> results|>,
    "JSON", "Compact" -> False];
