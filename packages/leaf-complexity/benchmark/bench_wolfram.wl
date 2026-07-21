(* Benchmark the Wolfram Language LeafComplexity resource function
   against the Python leaf_complexity port.

   Run from the package root (packages/leaf-complexity):

       wolframscript -file benchmark/bench_wolfram.wl

   and compare with the Python side:

       python benchmark/bench_python.py

   The definitions are loaded from the LeafComplexity 1.0.0 kernel file
   (wolfram/LeafComplexity-1-0-0-kernel.wl). *)

Get[FileNameJoin[{DirectoryName[$InputFileName], "..", "wolfram",
  "LeafComplexity-1-0-0-kernel.wl"}]];

small = (x + 2)/(y - 2);
medium = (2 x^(1/3) + I)/(x - 2 - 3 I) - 5/x^2;
poly2v = Expand[(x + y + 1)^15];
poly3v = Expand[(x + y + z + 1)^12];
nested = Table[{i, j, i/j}, {i, 20}, {j, 20}];
scaledF = Log[1. + Abs[#]]&;

cases = {
  {"small", Hold[LeafComplexity[small]]},
  {"medium", Hold[LeafComplexity[medium]]},
  {"poly-2var-15", Hold[LeafComplexity[poly2v]]},
  {"poly-3var-12", Hold[LeafComplexity[poly3v]]},
  {"nested-table", Hold[LeafComplexity[nested]]},
  {"poly-2var-15-scaled", Hold[LeafComplexity[poly2v, scaledF]]},
  {"poly-2var-15-proper", Hold[LeafComplexity[poly2v, Heads -> False]]}
};

results = Map[
  Function[{case},
    Module[{name = case[[1]], held = case[[2]], t, value},
      value = ReleaseHold[held];
      {t, value} = RepeatedTiming[ReleaseHold[held], 2];
      <|"case" -> name, "ms" -> t*1000, "value" -> N[value]|>]],
  cases];

Print[ExportString[
  <|"wolfram" -> $VersionNumber,
    "machine" -> $SystemID,
    "results" -> results|>,
  "JSON", "Compact" -> False]];
