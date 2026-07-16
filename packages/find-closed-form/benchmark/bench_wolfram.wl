(* Benchmark FindClosedForm (Wolfram Language) — paired with
   bench_python.py.

   Times the resource function version 1.0.0 over the same cases as the
   Python script and prints the results as JSON.  WolframAlpha queries
   are disabled ("WolframAlphaQueries" -> 0) for a fair comparison with
   the offline Python port.

   Run from the package root:  wolframscript -file benchmark/bench_wolfram.wl *)

FCF = ResourceFunction["FindClosedForm", ResourceVersion -> "1.0.0"];
AR = ResourceFunction["AlgebraicRange", ResourceVersion -> "2.0.0"];

noWA = "WolframAlphaQueries" -> 0;

cases = {
   {"default -> log(3/2)", Hold@FCF[0.405465, noWA]},
   {"default -> 1/6 + gamma(1/4)", Hold@FCF[3.792277, noWA]},
   {"default -> 1/sqrt(Catalan)", Hold@FCF[1.044866, noWA]},
   {"1/zeta(#)^2 -> zeta(1/5)^-2", Hold@FCF[1.85653, 1/Zeta[#]^2 &]},
   {"asinh -> sqrt(5)/6*asinh(4)", Hold@FCF[0.780653, ArcSinh]},
   {"log(1+exp(#)) -> 10*log(1+exp(1/10))",
    Hold@FCF[7.443967, Log[1 + Exp[#]] &]},
   {"{sinh,cosh,sech,csch} -> 6*sech(2/5)",
    Hold@FCF[5.550045, {Sinh, Cosh, Sech, Csch}]},
   {"gamma(#1)/gamma(#2) (2-arg)", Hold@FCF[4.688231, Gamma[#1]/Gamma[#2] &]},
   {"log, 10 results", Hold@FCF[0.405465, Log, 10]},
   {"search_range=Algebraic -> exp(sqrt(2))",
    Hold@FCF[4.1132503787829275, "SearchRange" -> (AR[-#, #, 1/#] &)]}
   };

timeCase[{label_, held_}] :=
  Module[{res, t},
   res = ReleaseHold[held];  (* warm-up *)
   t = First@RepeatedTiming[ReleaseHold[held];, 12];
   Print[label, "   ", t, " s   ", ToString[res, InputForm]];
   <|"case" -> label, "mean_s" -> Round[t, 0.001],
    "result" -> ToString[res, InputForm]|>];

results = timeCase /@ cases;

Print@ExportString[
   <|"wolfram" -> ToString[$VersionNumber],
    "function" -> "FindClosedForm 1.0.0",
    "system" -> $System, "cases" -> results|>, "JSON", "Compact" -> False];
