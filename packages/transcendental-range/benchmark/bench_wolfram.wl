(* Benchmark TranscendentalRange (Wolfram Language) — paired with
   bench_python.py.

   Times the resource function version 1.1.0 over the same cases as the
   Python script and prints the results as JSON.

   Run from the package root:  wolframscript -file benchmark/bench_wolfram.wl *)

TR = ResourceFunction["TranscendentalRange", ResourceVersion -> "1.1.0"];

cases = {
   {Exp, {1, 100, 1}, {}},
   {Exp, {100, -100, -1}, {}},
   {Log, {1, 100, 1}, {}},
   {Log, {100, -100, -1}, {}},
   {Tanh, {1, 10, 1}, {}},
   {Coth, {1, 30, 1}, {}},
   {Sech, {1, 100, 1}, {}},
   {Csch, {1, 100, 1}, {}},
   {ArcSinh, {1, 100, 1}, {}},
   {ArcCosh, {1, 100, 1}, {}},
   {ArcTanh, {0, 100, 1/10}, {}},
   {ArcSin, {0, 100, 1/10}, {}},
   {ArcCos, {0, 10, 1/10}, {}},
   {ArcTan, {1, 100, 1}, {}},
   {ArcSec, {1, 100, 1}, {}},
   {ArcCsc, {1, 100, 1}, {}},
   {Power, {1, 20, 1}, {"GeneratorsDomain" -> Algebraics}},
   {All, {-2, 2, 1/3}, {}}
   };

timeCase[{f_, {x_, y_, z_}, opts_}] :=
  Module[{res, t},
   res = TR[x, y, z, Method -> f, Sequence @@ opts];  (* warm-up *)
   t = First@RepeatedTiming[TR[x, y, z, Method -> f, Sequence @@ opts];, 5];
   <|"method" -> ToString[f], "args" -> ToString[{x, y, z}, InputForm],
    "mean_ms" -> Round[1000 t, 0.001], "length" -> Length[res]|>];

results = timeCase /@ cases;

Print@ExportString[
   <|"wolfram" -> ToString[$VersionNumber],
    "function" -> "TranscendentalRange 1.1.0",
    "system" -> $System, "cases" -> results|>, "JSON", "Compact" -> False];
