(* ::Package:: *)

(*
Benchmark the Wolfram Language AlgebraicRange 2.0 on the five reference
cases of its test suite (group 10).

Run from the package root:

    wolframscript -file benchmark/bench_wolfram.wl

Prints results as JSON, including the kernel version.
*)

Get[FileNameJoin[{DirectoryName[$InputFileName], "..", "wolfram",
  "AlgebraicRange-2-0-0.wl"}]];

cases = {
  {"AlgebraicRange[200]",
    Hold[AlgebraicRange[200]]},
  {"AlgebraicRange[200, -200, -1]",
    Hold[AlgebraicRange[200, -200, -1]]},
  {"AlgebraicRange[-100, 100, 1/2]",
    Hold[AlgebraicRange[-100, 100, 1/2]]},
  {"AlgebraicRange[60, -60, -1/3]",
    Hold[AlgebraicRange[60, -60, -3^(-1)]]},
  {"AlgebraicRange[1 - 10^-13, 1 + 10^-13, 10^-17, wp=30]",
    Hold[AlgebraicRange[1 - 10^(-13), 1 + 10^(-13), 10^(-17),
      WorkingPrecision -> 30]]}
};

results = Table[
  Module[{name, held, len, t},
    {name, held} = c;
    len = Length[ReleaseHold[held]];  (* warm-up *)
    t = First[RepeatedTiming[ReleaseHold[held];, 20]];
    <|"case" -> name, "mean_s" -> Round[t, 0.0001], "length" -> len|>
  ], {c, cases}];

Print[ExportString[<|
  "package" -> "AlgebraicRange 2.0.0 (Wolfram Function Repository)",
  "wolfram" -> ToString[$VersionNumber],
  "results" -> results
|>, "JSON", "Compact" -> False]];
