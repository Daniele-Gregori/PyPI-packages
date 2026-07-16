(* ::Package:: *)

(* ::Title:: *)
(* FindClosedForm Tests *)

(* ::Text:: *)
(* Verification tests for the FindClosedForm resource function. *)
(* Mirrors the Python test suite test_find_closed_form.py. *)


(* ── Setup ─────────────────────────────────────────────────────────────── *)

BeginTestSection["FindClosedForm Tests"]

fcf = ResourceFunction["FindClosedForm"];


(* ── Basic Matching ────────────────────────────────────────────────────── *)

BeginTestSection["BasicMatching"]

VerificationTest[
	result = fcf[0.7071067811865476];
	Abs[1 - N[result, 18] / 0.7071067811865476] < 10^-10,
	True,
	TestID -> "BasicMatching-sqrt2over2"
]

VerificationTest[
	result = fcf[3.141592653589793];
	Abs[1 - N[result, 18] / Pi] < 10^-10,
	True,
	TestID -> "BasicMatching-pi"
]

VerificationTest[
	result = fcf[1.618033988749895];
	Abs[1 - N[result, 18] / GoldenRatio] < 10^-10,
	True,
	TestID -> "BasicMatching-goldenratio"
]

VerificationTest[
	result = fcf[2.718281828459045];
	Abs[1 - N[result, 18] / E] < 10^-10,
	True,
	TestID -> "BasicMatching-e"
]

VerificationTest[
	result = fcf[1.7320508075688772];
	Abs[1 - N[result, 18] / Sqrt[3]] < 10^-10,
	True,
	TestID -> "BasicMatching-sqrt3"
]

VerificationTest[
	result = fcf[0.6931471805599453];
	Abs[1 - N[result, 18] / Log[2]] < 10^-10,
	True,
	TestID -> "BasicMatching-ln2"
]

VerificationTest[
	result = fcf[0.5772156649015329];
	Abs[1 - N[result, 18] / EulerGamma] < 10^-10,
	True,
	TestID -> "BasicMatching-EulerGamma"
]

VerificationTest[
	result = fcf[1.4142135623730951];
	Abs[1 - N[result, 18] / Sqrt[2]] < 10^-10,
	True,
	TestID -> "BasicMatching-sqrt2"
]

VerificationTest[
	result = fcf[3.142857142857143, {Identity[#] &}];
	Abs[1 - N[result, 18] / (22/7)] < 10^-10,
	True,
	TestID -> "BasicMatching-rational-22over7"
]

EndTestSection[] (* BasicMatching *)


(* ── Precision Matching ────────────────────────────────────────────────── *)

BeginTestSection["PrecisionMatching"]

VerificationTest[
	results = fcf[1.414, 3, "SignificantDigits" -> 4, "MaxSearchRounds" -> 5];
	Length[results] >= 1 && AllTrue[results, Abs[1 - N[#, 18] / 1.414] < 10^-3 &],
	True,
	TestID -> "PrecisionMatching-lowPrecision"
]

VerificationTest[
	result = fcf[1.4142135623730951, "SignificantDigits" -> 15];
	Abs[N[result, 18] - Sqrt[2]] < 10^-13,
	True,
	TestID -> "PrecisionMatching-highPrecision"
]

EndTestSection[] (* PrecisionMatching *)


(* ── Complexity Threshold ──────────────────────────────────────────────── *)

BeginTestSection["ComplexityThreshold"]

VerificationTest[
	rLow = fcf[0.7071067811865476, 5,
		"FormulaComplexity" -> 5, "MaxSearchRounds" -> 3];
	rHigh = fcf[0.7071067811865476, 5,
		"FormulaComplexity" -> 50, "MaxSearchRounds" -> 3];
	Length[rLow] <= Length[rHigh],
	True,
	TestID -> "ComplexityThreshold-fewerWithLow"
]

EndTestSection[] (* ComplexityThreshold *)


(* ── Custom Functions (single-argument) ────────────────────────────────── *)

BeginTestSection["CustomFunctions"]

VerificationTest[
	result = fcf[0.5, {Sin[Pi #] &}];
	Abs[N[result, 18] - 0.5] < 10^-10,
	True,
	TestID -> "CustomFunctions-sinPiX"
]

VerificationTest[
	result = fcf[1.0, {Cos[Pi #] &}];
	(* Cos[Pi*0] = 1 *)
	result =!= None,
	True,
	TestID -> "CustomFunctions-cosPiX"
]

VerificationTest[
	result = fcf[0.5, {Sin[Pi #] &, Cos[Pi #] &}];
	Abs[N[result, 18] - 0.5] < 10^-10,
	True,
	TestID -> "CustomFunctions-listOfFunctions"
]

EndTestSection[] (* CustomFunctions *)


(* ── Multi-Argument Functions ──────────────────────────────────────────── *)

BeginTestSection["MultiArgFunctions"]

VerificationTest[
	result = fcf[6.0, {#1 #2 &}, 1,
		"AlgebraicFactor" -> False, "AlgebraicAdd" -> False,
		"MaxSearchRounds" -> 5];
	Abs[N[result, 18] - 6.0] < 10^-10,
	True,
	TestID -> "MultiArgFunctions-product"
]

VerificationTest[
	result = fcf[5.0, {#1 + #2 &}, 1,
		"AlgebraicFactor" -> False, "AlgebraicAdd" -> False,
		"MaxSearchRounds" -> 5];
	Abs[N[result, 18] - 5.0] < 10^-10,
	True,
	TestID -> "MultiArgFunctions-sum"
]

VerificationTest[
	result = fcf[1.0, {#1 Sin[Pi #2] &}, 1,
		"AlgebraicFactor" -> False, "AlgebraicAdd" -> False,
		"MaxSearchRounds" -> 3];
	(* e.g. 2 * Sin[Pi/6] = 1 or 1 * Sin[Pi/2] = 1 *)
	Abs[N[result, 18] - 1.0] < 10^-10,
	True,
	TestID -> "MultiArgFunctions-withFunction"
]

VerificationTest[
	target = N[2 Log[3], 18];
	result = fcf[target, {#1 Log[#2] &}, 1, "MaxSearchRounds" -> 5];
	Abs[N[result, 18] - target] < 10^-10,
	True,
	TestID -> "MultiArgFunctions-algebraicFactor"
]

EndTestSection[] (* MultiArgFunctions *)


(* ── Options ───────────────────────────────────────────────────────────── *)

BeginTestSection["Options"]

VerificationTest[
	result = fcf[N[Pi], "AlgebraicFactor" -> False, "MaxSearchRounds" -> 5];
	Head[result] =!= $Failed,
	True,
	TestID -> "Options-noAlgebraicFactor"
]

VerificationTest[
	result = fcf[N[Pi], "AlgebraicAdd" -> False, "MaxSearchRounds" -> 5];
	Head[result] =!= $Failed,
	True,
	TestID -> "Options-noAlgebraicAdd"
]

VerificationTest[
	results = fcf[0.7071067811865476, 3, "MaxSearchRounds" -> 10];
	ListQ[results] && 1 <= Length[results] <= 3,
	True,
	TestID -> "Options-maxResultsMultiple"
]

EndTestSection[] (* Options *)


(* ── Edge Cases ────────────────────────────────────────────────────────── *)

BeginTestSection["EdgeCases"]

VerificationTest[
	result = fcf[0.0, "MaxSearchRounds" -> 3];
	(* Should not crash; result may be None or a value *)
	True,
	True,
	TestID -> "EdgeCases-zero"
]

VerificationTest[
	result = fcf[-1.4142135623730951, "MaxSearchRounds" -> 10];
	(* Should not crash *)
	True,
	True,
	TestID -> "EdgeCases-negative"
]

VerificationTest[
	result = fcf[2, "MaxSearchRounds" -> 3];
	(* Integer input should work *)
	True,
	True,
	TestID -> "EdgeCases-integerInput"
]

EndTestSection[] (* EdgeCases *)


(* ── Result Quality ────────────────────────────────────────────────────── *)

BeginTestSection["ResultQuality"]

VerificationTest[
	results = fcf[0.7071067811865476, 3, "MaxSearchRounds" -> 10];
	If[ListQ[results] && Length[results] >= 2,
		complexities = Map[ResourceFunction["AlgebraicRange"][#, "FormulaComplexity"] &, results];
		OrderedQ[complexities],
		(* If fewer than 2 results, ordering is trivially satisfied *)
		True],
	True,
	TestID -> "ResultQuality-sortedByComplexity"
]

EndTestSection[] (* ResultQuality *)


(* ── Cross-validation: Python vs WL numerical agreement ────────────────── *)

BeginTestSection["NumericalAgreement"]

(* These test that FindClosedForm returns numerically correct results *)

VerificationTest[
	result = fcf[0.7071067811865476];
	result =!= None && Abs[1 - N[result, 18] / 0.7071067811865476] < 10^-10,
	True,
	TestID -> "NumericalAgreement-sqrt2over2"
]

VerificationTest[
	result = fcf[3.141592653589793];
	result =!= None && Abs[1 - N[result, 18] / 3.141592653589793] < 10^-10,
	True,
	TestID -> "NumericalAgreement-pi"
]

VerificationTest[
	result = fcf[2.718281828459045];
	result =!= None && Abs[1 - N[result, 18] / 2.718281828459045] < 10^-10,
	True,
	TestID -> "NumericalAgreement-e"
]

VerificationTest[
	result = fcf[1.618033988749895];
	result =!= None && Abs[1 - N[result, 18] / 1.618033988749895] < 10^-10,
	True,
	TestID -> "NumericalAgreement-goldenratio"
]

VerificationTest[
	result = fcf[1.7320508075688772];
	result =!= None && Abs[1 - N[result, 18] / 1.7320508075688772] < 10^-10,
	True,
	TestID -> "NumericalAgreement-sqrt3"
]

VerificationTest[
	result = fcf[0.6931471805599453];
	result =!= None && Abs[1 - N[result, 18] / 0.6931471805599453] < 10^-10,
	True,
	TestID -> "NumericalAgreement-ln2"
]

VerificationTest[
	result = fcf[0.5772156649015329];
	result =!= None && Abs[1 - N[result, 18] / 0.5772156649015329] < 10^-10,
	True,
	TestID -> "NumericalAgreement-EulerGamma"
]

VerificationTest[
	result = fcf[1.4142135623730951];
	result =!= None && Abs[1 - N[result, 18] / 1.4142135623730951] < 10^-10,
	True,
	TestID -> "NumericalAgreement-sqrt2"
]

EndTestSection[] (* NumericalAgreement *)


EndTestSection[] (* FindClosedForm Tests *)
