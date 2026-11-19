(* Benchmark the Wolfram Language functions TransformTabular, TransformRows
   (resource functions) and TransformColumns (built-in) against their Python
   ports in the transform-tabular package.

   Run from the package root (packages/transform-tabular):

       wolframscript -file benchmark/bench_wolfram.wl

   and compare with the Python side:

       python benchmark/bench_python.py

   The definitions are loaded from the kernel files
   wolfram/TransformTabular-1-0-1-kernel.wl and
   wolfram/TransformRows-0-6-1-kernel.wl. *)

Get[FileNameJoin[{DirectoryName[$InputFileName], "..", "wolfram",
  "TransformTabular-1-0-1-kernel.wl"}]];
Get[FileNameJoin[{DirectoryName[$InputFileName], "..", "wolfram",
  "TransformRows-0-6-1-kernel.wl"}]];

(* a table of integers with columns "a" to "j", the same as on the Python side *)
table[n_] := Tabular[
  Mod[Outer[Plus, 7919 Range[n], 104729 Range[10]], 1000003],
  Alphabet[][[;; 10]]];

(* positions of m scattered rows *)
selection[m_, n_] := Union[Mod[7919 Range[m], n]] + 1;

small = table[10^5];
large = table[10^6];
rowsSmall = selection[10^4, 10^5];
rowsLarge = selection[10^5, 10^6];

cases = {
  {"tabular-elementwise",
   Hold[TransformTabular[small, # + 1 &]]},
  {"tabular-columnwise-value",
   Hold[TransformTabular[small, # - ColumnwiseValue[Min[#]] &]]},
  {"tabular-columnwise-thread",
   Hold[TransformTabular[small, ColumnwiseThread[Accumulate[#]] &]]},
  {"columns-arithmetic",
   Hold[TransformColumns[large, "a" -> Function[2 #a]]]},
  {"columns-new-column",
   Hold[TransformColumns[large, "k" -> Function[#a + #b]]]},
  {"columns-columnwise-value",
   Hold[TransformColumns[large, "a" -> Function[#a - ColumnwiseValue[Median[#b]]]]]},
  {"columns-strings",
   Hold[TransformColumns[small, "s" -> Function[ToString[#a] <> "-" <> ToString[#b]]]]},
  {"rows-list",
   Hold[TransformRows[large, rowsLarge, "a" -> Function[2 #a]]]},
  {"rows-list-unordered",
   Hold[TransformRows[large, rowsLarge, "a" -> Function[2 #a], "KeepOrder" -> False]]},
  {"rows-criterion",
   Hold[TransformRows[small, EvenQ[#c] &, "a" -> Function[2 #a + 2 #b]]]},
  {"rows-all-cells",
   Hold[TransformRows[small, rowsSmall, #^2 &]]}
};

(* total of the numeric cells, plus the total length of the strings *)
checksum[t_] := Total[
  Function[key,
    With[{column = Normal[t[All, key]]},
      If[StringQ[First[column]], Total[StringLength[column]], Total[column]]]] /@
   ColumnKeys[t]];

results = Map[
  Function[{case},
    Module[{name = case[[1]], held = case[[2]], t, result},
      ReleaseHold[held];
      {t, result} = RepeatedTiming[ReleaseHold[held], 2];
      <|"case" -> name, "ms" -> t*1000,
        "value" -> N[checksum[result]],
        "head" -> N[Normal[result[;; 3, "a"]]]|>]],
  cases];

Print[ExportString[
  <|"wolfram" -> $VersionNumber,
    "machine" -> $SystemID,
    "results" -> results|>,
  "JSON", "Compact" -> False]];
