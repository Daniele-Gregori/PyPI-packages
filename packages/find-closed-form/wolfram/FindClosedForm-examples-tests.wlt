(* FindClosedForm Examples Tests *)
(* Derived from the definition notebook input/output pairs *)
(* Run with: wolframscript -file FindClosedForm-examples-tests.wlt *)

BeginTestSection["FindClosedForm Definition Notebook Examples"]

fcf = ResourceFunction["FindClosedForm"];


(* ══════════════════════════════════════════════════════════════════════ *)
(* Basic Examples (cells 20-25, outputs 1-6) *)
(* ══════════════════════════════════════════════════════════════════════ *)

BeginTestSection["BasicExamples"]

(* Input 20 → Output 1: fcf[0.405465] = Log[3/2] *)
VerificationTest[
    fcf[0.405465],
    Log[3/2],
    TestID -> "Basic-01-Log3over2"
]

(* Input 21 → Output 2: fcf[3.792277] = 1/6 + Gamma[1/4] *)
VerificationTest[
    fcf[3.792277],
    1/6 + Gamma[1/4],
    TestID -> "Basic-02-Gamma1over4"
]

(* Input 22 → Output 3: fcf[3.311601] = 5/3 + Pi^2/6 *)
VerificationTest[
    fcf[3.311601],
    5/3 + Pi^2/6,
    TestID -> "Basic-03-PiSqOver6"
]

(* Input 23 → Output 4: fcf[1.044866] = 1/Sqrt[Catalan] *)
VerificationTest[
    fcf[1.044866],
    1/Sqrt[Catalan],
    TestID -> "Basic-04-InvSqrtCatalan"
]

(* Input 24 → Output 5: fcf[1.85653, 1/Zeta[#]^2&] = Zeta[1/5]^(-2) *)
VerificationTest[
    fcf[1.85653, 1/Zeta[#1]^2 &],
    Zeta[1/5]^(-2),
    TestID -> "Basic-05-InvZetaSq"
]

(* Input 25 → Output 6: fcf[-0.309033, PolyLog[#1,#2]&] = PolyLog[2,-1/3] *)
VerificationTest[
    fcf[-0.309033, PolyLog[#1, #2] &],
    PolyLog[2, -1/3],
    TestID -> "Basic-06-PolyLog"
]

EndTestSection[]


(* ══════════════════════════════════════════════════════════════════════ *)
(* Scope Examples (cells 26-39, outputs 7-20) *)
(* ══════════════════════════════════════════════════════════════════════ *)

BeginTestSection["ScopeExamples"]

(* Input 26 → Output 7: fcf[3.3116, 4] returns a list of 4 *)
VerificationTest[
    results = fcf[3.3116, 4];
    ListQ[results] && Length[results] == 4,
    True,
    TestID -> "Scope-07-MultipleResults4"
]

(* Input 28 → Output 9: fcf[0.69314718, 4] includes Log[2] *)
VerificationTest[
    results = fcf[0.69314718, 4];
    ListQ[results] && MemberQ[results, Log[2]],
    True,
    TestID -> "Scope-09-Log2InResults"
]

(* Input 30 → Output 11: fcf[0.405465, Log, 10] starts with Log[3/2] *)
VerificationTest[
    results = fcf[0.405465, Log, 10];
    ListQ[results] && Length[results] == 10 && results[[1]] === Log[3/2],
    True,
    TestID -> "Scope-11-Log10Results"
]

(* Input 32 → Output 13: fcf[-1.185732, PolyGamma[#]&] *)
VerificationTest[
    result = fcf[-1.185732, PolyGamma[#1] &];
    result =!= None,
    True,
    TestID -> "Scope-13-PolyGamma"
]

(* Input 33 → Output 14: fcf[0.780653, ArcSinh] *)
VerificationTest[
    result = fcf[0.780653, ArcSinh];
    result =!= None,
    True,
    TestID -> "Scope-14-ArcSinh"
]

(* Input 34 → Output 15: fcf[1.054136, BarnesG[Sqrt[2] #]&] *)
VerificationTest[
    result = fcf[1.054136, BarnesG[Sqrt[2]*#1] &];
    result =!= None,
    True,
    TestID -> "Scope-15-BarnesG"
]

(* Input 35 → Output 16: fcf[7.443967, Log[1+Exp[#]]&] = 10 Log[1+E^(1/10)] *)
VerificationTest[
    result = fcf[7.443967, Log[1 + Exp[#1]] &];
    result =!= None,
    True,
    TestID -> "Scope-16-LogExp"
]

(* Input 36 → Output 17: fcf[4.688231, Gamma[#1]/Gamma[#2]&] *)
VerificationTest[
    result = fcf[4.688231, Gamma[#1]/Gamma[#2] &];
    result =!= None,
    True,
    TestID -> "Scope-17-GammaRatio"
]

(* Input 37 → Output 18: fcf[0.7299085, HypergeometricU[#1,#2,#3]&] *)
VerificationTest[
    result = fcf[0.7299085, HypergeometricU[#1, #2, #3] &];
    result =!= None,
    True,
    TestID -> "Scope-18-HypergeometricU"
]

(* Input 38 → Output 19: fcf[5.550045, {Sinh, Cosh, Sech, Csch}] = 6 Sech[2/5] *)
VerificationTest[
    result = fcf[5.550045, {Sinh, Cosh, Sech, Csch}];
    result =!= None,
    True,
    TestID -> "Scope-19-HyperbolicList"
]

(* Input 39 → Output 20: fcf[-1.479735, {EllipticK, EllipticE, EllipticPi}] *)
VerificationTest[
    result = fcf[-1.479735, {EllipticK[#1] &, EllipticE[#1] &, EllipticPi[#1, #2] &}];
    result =!= None,
    True,
    TestID -> "Scope-20-EllipticList"
]

EndTestSection[]


(* ══════════════════════════════════════════════════════════════════════ *)
(* Options: AlgebraicAdd/AlgebraicFactor (cells 40-45, outputs 21-26) *)
(* ══════════════════════════════════════════════════════════════════════ *)

BeginTestSection["AlgebraicOptions"]

(* Input 40 → Output 21: AlgebraicAdd->False speeds up *)
VerificationTest[
    {t1, r1} = AbsoluteTiming[fcf[0.1013578, 1/(Gamma[#1]*Gamma[#2]) &, "AlgebraicAdd" -> False]];
    r1 =!= None,
    True,
    TestID -> "Options-21-AlgAddFalse"
]

(* Input 42 → Output 23: AlgebraicFactor->False *)
VerificationTest[
    {t1, r1} = AbsoluteTiming[fcf[-9.6530201, PolyGamma[#1] + PolyGamma[#2] &, "AlgebraicFactor" -> False]];
    r1 =!= None,
    True,
    TestID -> "Options-23-AlgFactorFalse"
]

(* Input 44 → Output 25: Both restrictions *)
VerificationTest[
    {t1, r1} = AbsoluteTiming[fcf[0.3057349, PolyLog[#1, #2] &, "AlgebraicFactor" -> False, "AlgebraicAdd" -> False]];
    r1 =!= None,
    True,
    TestID -> "Options-25-BothRestrictions"
]

EndTestSection[]


(* ══════════════════════════════════════════════════════════════════════ *)
(* Options: FormulaComplexity (cells 46-51, outputs 27-32) *)
(* ══════════════════════════════════════════════════════════════════════ *)

BeginTestSection["ComplexityOptions"]

(* Input 46 → Output 27: fcf[38.94017, Gamma] *)
VerificationTest[
    result = fcf[38.94017, Gamma];
    result =!= None,
    True,
    TestID -> "Options-27-GammaDefault"
]

(* Input 48 → Output 29: FormulaComplexity->15 gives simpler result *)
VerificationTest[
    result = fcf[38.94017, Gamma, "FormulaComplexity" -> 15];
    result =!= None,
    True,
    TestID -> "Options-29-GammaLowComplexity"
]

(* Input 50 → Output 31: 3-arg function with high complexity *)
VerificationTest[
    result = fcf[4.51766814, Gamma[#2]^3/Gamma[#1]^2 + Identity[#3]*Gamma[#1]^2/Gamma[#2] &, "FormulaComplexity" -> 100];
    result =!= None,
    True,
    TestID -> "Options-31-HighComplexity3Arg"
]

EndTestSection[]


(* ══════════════════════════════════════════════════════════════════════ *)
(* Options: SearchRange & MaxSearchRounds (cells 52-54, outputs 33-35) *)
(* ══════════════════════════════════════════════════════════════════════ *)

BeginTestSection["SearchRangeOptions"]

(* Input 52 → Output 33: Plain range finds Gamma[1/50] *)
VerificationTest[
    result = fcf[49.44221, Gamma, "AlgebraicAdd" -> False, "AlgebraicFactor" -> False, "SearchRange" -> "Plain"];
    result =!= None,
    True,
    TestID -> "Options-33-PlainRange"
]

(* Input 53 → Output 34: Beyond default rounds returns None *)
VerificationTest[
    fcf[59.43902, Gamma, "AlgebraicAdd" -> False, "AlgebraicFactor" -> False, "SearchRange" -> "Plain"],
    None,
    TestID -> "Options-34-BeyondDefaultNone"
]

(* Input 54 → Output 35: MaxSearchRounds->100 finds result *)
VerificationTest[
    result = fcf[59.43902, Gamma, "MaxSearchRounds" -> 100, "AlgebraicAdd" -> False, "AlgebraicFactor" -> False, "SearchRange" -> "Plain"];
    result =!= None,
    True,
    TestID -> "Options-35-IncreasedRounds"
]

EndTestSection[]


(* ══════════════════════════════════════════════════════════════════════ *)
(* Options: OutputArguments (cells 57-60, outputs 38-41) *)
(* ══════════════════════════════════════════════════════════════════════ *)

BeginTestSection["OutputArgumentsOptions"]

(* Input 57 → Output 38: OutputArguments->True returns Association *)
VerificationTest[
    result = fcf[1.32325, Gamma[#1]/Gamma[#2] &, "OutputArguments" -> True];
    Head[result] === Association,
    True,
    TestID -> "Options-38-OutputArgs"
]

(* Input 59 → Output 40: Det matrix with OutputArguments *)
VerificationTest[
    result = fcf[19/18, Det[{{#1, #2}, {#3, #4}}] &, "OutputArguments" -> True, "AlgebraicAdd" -> False, "AlgebraicFactor" -> False];
    Head[result] === Association,
    True,
    TestID -> "Options-40-DetMatrix"
]

EndTestSection[]


(* ══════════════════════════════════════════════════════════════════════ *)
(* Options: RationalSolutions (cells 61-66, outputs 42-47) *)
(* ══════════════════════════════════════════════════════════════════════ *)

BeginTestSection["RationalSolutionsOptions"]

(* Input 61 → Output 42: RationalSolutions->False *)
VerificationTest[
    result = fcf[0.25, Sin[Pi*#1] &, "RationalSolutions" -> False, "AlgebraicAdd" -> False];
    result =!= None,
    True,
    TestID -> "Options-42-NoRationalSolutions"
]

(* Input 63 → Output 44: RationalSolutions->True *)
VerificationTest[
    fcf[0.25, Sin[Pi*#1] &, "RationalSolutions" -> True, "AlgebraicAdd" -> False],
    1/4,
    TestID -> "Options-44-RationalSolutions"
]

(* Input 64 → Output 45: Identity always returns rational *)
VerificationTest[
    fcf[0.25, Identity[#1] &],
    1/4,
    TestID -> "Options-45-IdentityRational"
]

EndTestSection[]


(* ══════════════════════════════════════════════════════════════════════ *)
(* Options: RootApproximantMethod (cells 67-70, outputs 48-51) *)
(* ══════════════════════════════════════════════════════════════════════ *)

BeginTestSection["RootApproximantOptions"]

(* Input 67 → Output 48: Default method *)
VerificationTest[
    result = fcf[1.32471796, Identity, "FormulaComplexity" -> 300];
    result =!= None && Abs[N[result, 10] - 1.32471796] < 10^-5,
    True,
    TestID -> "Options-48-DefaultRoot"
]

(* Input 69 → Output 50: BuiltIn method *)
VerificationTest[
    result = fcf[1.32471796, Identity, "FormulaComplexity" -> 300, "RootApproximantMethod" -> "BuiltIn"];
    result =!= None && Abs[N[result, 10] - 1.32471796] < 10^-5,
    True,
    TestID -> "Options-50-BuiltInRoot"
]

EndTestSection[]


(* ══════════════════════════════════════════════════════════════════════ *)
(* Options: SearchArguments (cells 81-83, outputs 62-64) *)
(* ══════════════════════════════════════════════════════════════════════ *)

BeginTestSection["SearchArgumentsOptions"]

(* Input 81 → Output 62: SearchArguments list *)
VerificationTest[
    {t, result} = AbsoluteTiming[fcf[4.678938, Gamma[#1] &, "SearchArguments" -> {3, 1, 1/3}]];
    result =!= None,
    True,
    TestID -> "Options-62-SearchArgsList"
]

(* Input 82 → Output 63: SearchArguments per-slot lists *)
VerificationTest[
    result = fcf[1.32325, Gamma[#1]/Gamma[#2] &, "SearchArguments" -> {{1, 1/2}, {3, 1, 1/3}}];
    result =!= None,
    True,
    TestID -> "Options-63-SearchArgsPerSlot"
]

EndTestSection[]


(* ══════════════════════════════════════════════════════════════════════ *)
(* Options: SearchRange variants (cells 90-97, outputs 71-78) *)
(* ══════════════════════════════════════════════════════════════════════ *)

BeginTestSection["SearchRangeVariants"]

(* Input 90 → Output 71: Farey default for PolyLog *)
VerificationTest[
    {t, result} = AbsoluteTiming[fcf[0.442109, PolyLog[#1, #2] &]];
    result =!= None,
    True,
    TestID -> "Options-71-FareyPolyLog"
]

(* Input 92 → Output 73: Plain range for Gamma product *)
VerificationTest[
    {t, result} = AbsoluteTiming[fcf[14.911818, Gamma[#1]*Gamma[#2] &, "SearchRange" -> "Plain"]];
    result =!= None,
    True,
    TestID -> "Options-73-PlainGammaProduct"
]

(* Input 96 → Output 77: Integer range for Log product *)
VerificationTest[
    {t, result} = AbsoluteTiming[fcf[6.263643, Log[#1]*Log[#2] &, "SearchRange" -> "Integer"]];
    result =!= None,
    True,
    TestID -> "Options-77-IntegerLogProduct"
]

(* Input 97 → Output 78: Custom range function *)
VerificationTest[
    {t, result} = AbsoluteTiming[fcf[13.165149, Log, "SearchRange" -> (Range[0, 100 #, 25] &)]];
    result =!= None,
    True,
    TestID -> "Options-78-CustomRange"
]

EndTestSection[]


(* ══════════════════════════════════════════════════════════════════════ *)
(* Options: SignificantDigits (cells 100-105, outputs 81-86) *)
(* ══════════════════════════════════════════════════════════════════════ *)

BeginTestSection["SignificantDigitsOptions"]

(* Input 100 → Output 81: Relaxed digits *)
VerificationTest[
    result = fcf[0.81248057539, 1/Zeta[#1]^2 &, "SignificantDigits" -> 7];
    result =!= None,
    True,
    TestID -> "Options-81-RelaxedDigits"
]

(* Input 103 → Output 84: Log[2] from 6 digits *)
VerificationTest[
    fcf[0.693147, Log],
    Log[2],
    TestID -> "Options-84-Log2-6digits"
]

(* Input 104 → Output 85: SignificantDigits->10 too strict *)
VerificationTest[
    TimeConstrained[fcf[0.693147, Log, "SignificantDigits" -> 10], 30],
    $Aborted,
    TestID -> "Options-85-TooStrict"
]

EndTestSection[]


(* ══════════════════════════════════════════════════════════════════════ *)
(* Properties and Relations (cells 124-131, outputs 104-112) *)
(* ══════════════════════════════════════════════════════════════════════ *)

BeginTestSection["PropertiesRelations"]

(* Input 124 → Output 105: Identity rationalization *)
VerificationTest[
    fcf[0.666, Identity],
    2/3,
    TestID -> "Properties-105-Rationalize"
]

(* Input 126 → Output 107: Root approximation 3 Sqrt[2] *)
VerificationTest[
    fcf[4.243, Identity],
    3*Sqrt[2],
    TestID -> "Properties-107-RootApprox"
]

(* Input 128 → Output 109: Radical denesting *)
VerificationTest[
    result = fcf[0.5848, Identity];
    result =!= None && Abs[N[result, 10] - 0.5848] < 10^-3,
    True,
    TestID -> "Properties-109-RadicalDenest"
]

(* Input 131 → Output 112: BesselK root-finding *)
VerificationTest[
    result = fcf[3.22921, BesselK[1, #1] &, "AlgebraicAdd" -> False, "AlgebraicFactor" -> False];
    result =!= None,
    True,
    TestID -> "Properties-112-BesselK"
]

EndTestSection[]


(* ══════════════════════════════════════════════════════════════════════ *)
(* Applications (cells 113-122, outputs 93-103) *)
(* ══════════════════════════════════════════════════════════════════════ *)

BeginTestSection["Applications"]

(* Input 113 → Output 94: EllipticK and Gamma relation *)
VerificationTest[
    result = fcf[N[EllipticK[2]], Gamma[#1]^2/Gamma[#2] &, "AlgebraicAdd" -> False];
    result =!= None,
    True,
    TestID -> "Applications-94-EllipticGamma"
]

(* Input 114 → Output 95: EllipticPi complex formula *)
VerificationTest[
    result = fcf[N[EllipticPi[1/2, 1/2]], Gamma[#2]^3/Gamma[#1]^2 + Identity[#3]*Gamma[#1]^2/Gamma[#2] &, "AlgebraicAdd" -> False, "FormulaComplexity" -> 100];
    result =!= None,
    True,
    TestID -> "Applications-95-EllipticPiGamma"
]

EndTestSection[]


EndTestSection[]
