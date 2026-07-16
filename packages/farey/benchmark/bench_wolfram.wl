(* Benchmark FareyRange (Wolfram Language) — paired with bench_python.py.

   Times the FareyRange resource function over the same cases as the Python
   script and prints the results as JSON.

   Run from the package root:  wolframscript -file benchmark/bench_wolfram.wl *)

FR = ResourceFunction["FareyRange"];

cases = {
   {0, 1, 3},
   {0, 10, 5},
   {-20, 20, 6},
   {0, 30, 7},
   {0, 50, 4},
   {0, 100, 3},
   {0, 200, 5},
   {0, 1000, 2}
   };

timeCase[{x_, y_, z_}] :=
  Module[{res, t},
   res = FR[x, y, z];  (* warm-up *)
   t = First@RepeatedTiming[FR[x, y, z];, 5];
   <|"args" -> {x, y, z}, "mean_ms" -> Round[1000 t, 0.0001],
    "length" -> Length[res]|>];

results = timeCase /@ cases;

Print@ExportString[
   <|"wolfram" -> ToString[$VersionNumber],
    "function" -> "FareyRange resource function",
    "system" -> $System, "cases" -> results|>, "JSON", "Compact" -> False];
