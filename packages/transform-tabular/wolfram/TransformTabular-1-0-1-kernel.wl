(* ::Package:: *)

(* ::Title:: *)
(*TransformTabular (1.0.1)*)


(* ::Subtitle:: *)
(*Wolfram Resource Function contributed by Daniele Gregori*)


(* ::Section:: *)
(*Package Header*)


BeginPackage["TransformTabular`"];


TransformTabular::usage=
"TransformTabular[tab, f] transforms all columns of a tabular through a single function;
TransformTabular[tab, f, i;;j] transforms a span of columns of a tabular through a single function;
TransformTabular[tab, f, {col1, col2, ...}] transforms a list of columns of a tabular through a single function;
TransformTabular[f, {col1, col2, ...}] represents the operator form of TransformTabular.";


Begin["`Private`"];


(* ::Section:: *)
(*Definition*)


(* ::Input::Initialization:: *)
ClearAll[TransformTabular]

TransformTabular[tab_Tabular,fun:_Function,cols_List]/;SubsetQ[ColumnKeys[tab],cols]:=
	Block[{schema},
	schema=KeyDrop[Normal[TabularSchema[tab]],"ColumnProperties"];
	Tabular[
		TransformColumns[tab,
						Table[
							ckey->ReplaceAll[fun,Slot[1]->Slot[ckey]],
							{ckey,cols}]],
		schema]]


(*other possible column specifications:*)

TransformTabular[tab_Tabular,fun:_Function,cols:_Span|{_Integer..}|All]:=
	TransformTabular[tab,fun,ColumnKeys[tab][[cols]]]
TransformTabular[tab_Tabular,fun:_Function,col_Integer]:=
	TransformTabular[tab,fun,ColumnKeys[tab][[{col}]]]
TransformTabular[tab_Tabular,fun:_Function,col_String]/;MemberQ[ColumnKeys[tab],col]:=
	TransformTabular[tab,fun,{col}]

(*all columns:*)

TransformTabular[tab_Tabular,fun:_Function]:=
	TransformTabular[tab,fun,ColumnKeys[tab]]

(*operator forms:*)

TransformTabular[fun:_Function,cols:_Span|_Integer|_String|_List|All][tab_Tabular]:=
	TransformTabular[tab,fun,cols]
TransformTabular[fun:_Function][tab_Tabular]:=
	TransformTabular[tab,fun]

(*catch-all function specification*)

TransformTabular[tab_Tabular,f_,cols_]:=
	TransformTabular[tab,f[#]&,cols]
TransformTabular[tab_Tabular,f_]:=
	TransformTabular[tab,f[#]&]
TransformTabular[f_,cols_][tab_Tabular]:=
	TransformTabular[f[#]&,cols][tab]
TransformTabular[f_][tab_Tabular]:=
	TransformTabular[f[#]&][tab]


(* ::Section:: *)
(*Package Footer*)


End[];
EndPackage[];
