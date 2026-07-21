(* ::Package:: *)

(* ::Title:: *)
(*LeafComplexity (1.0.0)*)


(* ::Subtitle:: *)
(*Wolfram Resource Function contributed by Daniele Gregori*)


(* ::Section:: *)
(*Package Header*)


BeginPackage["LeafComplexity`"];


LeafComplexity::usage=
"LeafComplexity[expr] gives the sum of all the numeric indivisible subexpressions in expr;
LeafComplexity[expr, f] apply a function f to each indivisible subexpression in expr and take the total;
LeafComplexity[expr, f, g] apply a function f to each indivisible subexpression in expr and apply to it recursively another unary or binary function g.";


Begin["`Private`"];


(* ::Section:: *)
(*Definition*)


(* ::Subsection::Closed:: *)
(*Identity scaling*)


(* ::Subsubsection:: *)
(*All nodes*)


(* ::Input:: *)
(*(*use of global symbols and redundant function definitions aims at improving efficiency of interpreted code*)*)


(* ::Input::Initialization:: *)
ClearAll[lt]

lt[e:_Integer|_Real]:=AddTo[s,Abs@e];
lt[e_Symbol]:=If[NumericQ[e],AddTo[s,Abs@e],Increment[s]];
lt[e_DirectedInfinity]:=AddTo[s,Infinity]
lt[e_]:=Increment[s]

lt[e_Complex]:=Scan[lt[#]&,ReIm[e],Heads->True];
lt[e_Rational]:=Scan[lt[#]&,NumeratorDenominator[e],Heads->True]			
lt[e_Association]:=Scan[lt[#]&,Values[e],Heads->True]		
	
lt[e_]/;!AtomQ[e]:=Scan[lt,e,Heads->True]


(* ::Input::Initialization:: *)
leafTotal[expr_]:=Block[{s=1(*see important definition detail*)},lt[expr];s]


(* ::Subsubsection:: *)
(*Proper leaf nodes*)


(* ::Input::Initialization:: *)
ClearAll[plt]

plt[e:_Integer|_Real]:=AddTo[s,Abs@e];
plt[e_Symbol]:=If[NumericQ[e],AddTo[s,Abs@e],Increment[s]];
plt[e_DirectedInfinity]:=AddTo[s,Infinity]
plt[e_]:=Increment[s]

plt[e_Complex]:=Scan[plt[#]&,ReIm[e],Heads->False];
plt[e_Rational]:=Scan[plt[#]&,NumeratorDenominator[e],Heads->False]			
plt[e_Association]:=Scan[plt[#]&,Values[e],Heads->False]		
	
plt[e_]/;!AtomQ[e]:=Scan[plt,e,Heads->False]


(* ::Input::Initialization:: *)
properLeafTotal[expr_]:=Block[{s=1},plt[expr];s]


(* ::Subsection::Closed:: *)
(*Custom scaling*)


(* ::Subsubsection:: *)
(*All nodes*)


(* ::Input::Initialization:: *)
ClearAll[slt]

slt[e:_Integer|_Real,f_]:=AddTo[s,f[e]];
slt[e_Symbol,f_]:=If[NumericQ[e],AddTo[s,f[e]],AddTo[s,f[1]]];
slt[e_DirectedInfinity,f_]:=AddTo[s,f[e]]
slt[e_,f_]:=AddTo[s,f[1]]

slt[e_Complex,f_]:=Scan[slt[#,f]&,ReIm[e],Heads->True];
slt[e_Rational,f_]:=Scan[slt[#,f]&,NumeratorDenominator[e],Heads->True]			
slt[e_Association,f_]:=Scan[slt[#,f]&,Values[e],Heads->True]		
	
slt[e_,f_]/;!AtomQ[e]:=Scan[slt[#,f]&,e,Heads->True]


(* ::Input::Initialization:: *)
scalingLeafTotal[expr_,f_]:=Block[{s=1},slt[expr,f];s]


(* ::Subsubsection:: *)
(*Proper leaf nodes*)


(* ::Input::Initialization:: *)
ClearAll[splt]

splt[e:_Integer|_Real,f_]:=AddTo[s,f[e]];
splt[e_Symbol,f_]:=If[NumericQ[e],AddTo[s,f[e]],AddTo[s,f[1]]];
splt[e_DirectedInfinity,f_]:=AddTo[s,f[e]]
splt[e_,f_]:=AddTo[s,f[1]]

splt[e_Complex,f_]:=Scan[splt[#,f]&,ReIm[e],Heads->False];
splt[e_Rational,f_]:=Scan[splt[#,f]&,NumeratorDenominator[e],Heads->False]			
splt[e_Association,f_]:=Scan[splt[#,f]&,Values[e],Heads->False]		
	
splt[e_,f_]/;!AtomQ[e]:=Scan[splt[#,f]&,e,Heads->False]


(* ::Input::Initialization:: *)
scalingProperLeafTotal[expr_,f_]:=Block[{s=1},splt[expr,f];s]


(* ::Subsection::Closed:: *)
(*Custom wrapping*)


(* ::Subsubsection:: *)
(*All nodes*)


(* ::Input::Initialization:: *)
ClearAll[wlt]

wlt[e:_Integer|_Real,f_,g_]:=s=g[s,f[e]];
wlt[e_Symbol,f_,g_]:=If[NumericQ[e],s=g[s,f[e]],s=g[s,f[1]]];
wlt[e_DirectedInfinity,f_,g_]:=s=g[s,f[e]]
wlt[e_,f_,g_]:=s=g[s,f[1]]

wlt[e_Complex,f_,g_]:=Scan[wlt[#,f,g]&,ReIm[e],Heads->True];
wlt[e_Rational,f_,g_]:=Scan[wlt[#,f,g]&,NumeratorDenominator[e],Heads->True]			
wlt[e_Association,f_,g_]:=Scan[wlt[#,f,g]&,Values[e],Heads->True]		
	
wlt[e_,f_,g_]/;!AtomQ[e]:=Scan[wlt[#,f,g]&,e,Heads->True]


(* ::Input::Initialization:: *)
wrappingLeafTotal[expr_,f_,g_]:=Block[{s=1},wlt[expr,f,g];s]


(* ::Subsubsection:: *)
(*Proper leaf nodes*)


(* ::Input::Initialization:: *)
ClearAll[pwlt]

pwlt[e:_Integer|_Real,f_,g_]:=s=g[s,f[e]];
pwlt[e_Symbol,f_,g_]:=If[NumericQ[e],s=g[s,f[e]],s=g[s,f[1]]];
pwlt[e_DirectedInfinity,f_,g_]:=s=g[s,f[e]]
pwlt[e_,f_,g_]:=s=g[s,f[1]]

pwlt[e_Complex,f_,g_]:=Scan[pwlt[#,f,g]&,ReIm[e],Heads->False];
pwlt[e_Rational,f_,g_]:=Scan[pwlt[#,f,g]&,NumeratorDenominator[e],Heads->False]			
pwlt[e_Association,f_,g_]:=Scan[pwlt[#,f,g]&,Values[e],Heads->False]		
	
pwlt[e_,f_,g_]/;!AtomQ[e]:=Scan[pwlt[#,f,g]&,e,Heads->False]


(* ::Input::Initialization:: *)
wrappingProperLeafTotal[expr_,f_,g_]:=Block[{s=1},pwlt[expr,f,g];s]


(* ::Subsection::Closed:: *)
(*Main*)


(* ::Input::Initialization:: *)
ClearAll[LeafComplexity]

Options[LeafComplexity]={Heads->True};

LeafComplexity[expr_,opt:OptionsPattern[LeafComplexity]]:=
	If[OptionValue[Heads],
				leafTotal[expr],
				properLeafTotal[expr]]

LeafComplexity[expr_,f:_Function|_Symbol,opt:OptionsPattern[LeafComplexity]]:=
	If[OptionValue[Heads],
				scalingLeafTotal[expr,f],
				scalingProperLeafTotal[expr,f]]

LeafComplexity[expr_,f:_Function|_Symbol,g:_Function|_Symbol,opt:OptionsPattern[LeafComplexity]]:=
	If[OptionValue[Heads],
				wrappingLeafTotal[expr,f,g],
				wrappingProperLeafTotal[expr,f,g]]


(* ::Section:: *)
(*Package Footer*)


End[];
EndPackage[];
