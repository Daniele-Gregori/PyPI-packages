(* Benchmark the Wolfram Language LeafComplexity resource function
   against the Python leaf_complexity port.

   Run from the package root (packages/leaf-complexity):

       wolframscript -file benchmark/bench_wolfram.wl

   and compare with the Python side:

       python benchmark/bench_python.py

   The definitions below are the ones of the LeafComplexity 1.0.0
   definition notebook (wolfram/LeafComplexity-1-0-0-definition.nb). *)

ClearAll[lt];
lt[e : _Integer | _Real] := AddTo[s, Abs@e];
lt[e_Symbol] := If[NumericQ[e], AddTo[s, Abs@e], Increment[s]];
lt[e_DirectedInfinity] := AddTo[s, Infinity];
lt[e_] := Increment[s];
lt[e_Complex] := Scan[lt[#]&, ReIm[e], Heads -> True];
lt[e_Rational] := Scan[lt[#]&, NumeratorDenominator[e], Heads -> True];
lt[e_Association] := Scan[lt[#]&, Values[e], Heads -> True];
lt[e_] /; !AtomQ[e] := Scan[lt, e, Heads -> True];
leafTotal[expr_] := Block[{s = 1}, lt[expr]; s];

ClearAll[plt];
plt[e : _Integer | _Real] := AddTo[s, Abs@e];
plt[e_Symbol] := If[NumericQ[e], AddTo[s, Abs@e], Increment[s]];
plt[e_DirectedInfinity] := AddTo[s, Infinity];
plt[e_] := Increment[s];
plt[e_Complex] := Scan[plt[#]&, ReIm[e], Heads -> False];
plt[e_Rational] := Scan[plt[#]&, NumeratorDenominator[e], Heads -> False];
plt[e_Association] := Scan[plt[#]&, Values[e], Heads -> False];
plt[e_] /; !AtomQ[e] := Scan[plt, e, Heads -> False];
properLeafTotal[expr_] := Block[{s = 1}, plt[expr]; s];

ClearAll[slt];
slt[e : _Integer | _Real, f_] := AddTo[s, f[e]];
slt[e_Symbol, f_] := If[NumericQ[e], AddTo[s, f[e]], AddTo[s, f[1]]];
slt[e_DirectedInfinity, f_] := AddTo[s, f[e]];
slt[e_, f_] := AddTo[s, f[1]];
slt[e_Complex, f_] := Scan[slt[#, f]&, ReIm[e], Heads -> True];
slt[e_Rational, f_] := Scan[slt[#, f]&, NumeratorDenominator[e], Heads -> True];
slt[e_Association, f_] := Scan[slt[#, f]&, Values[e], Heads -> True];
slt[e_, f_] /; !AtomQ[e] := Scan[slt[#, f]&, e, Heads -> True];
scalingLeafTotal[expr_, f_] := Block[{s = 1}, slt[expr, f]; s];

ClearAll[LeafComplexity];
Options[LeafComplexity] = {Heads -> True};
LeafComplexity[expr_, opt : OptionsPattern[LeafComplexity]] :=
  If[OptionValue[Heads], leafTotal[expr], properLeafTotal[expr]];
LeafComplexity[expr_, f : _Function | _Symbol, opt : OptionsPattern[LeafComplexity]] :=
  If[OptionValue[Heads], scalingLeafTotal[expr, f], Null];

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
