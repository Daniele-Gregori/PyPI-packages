(* ::Package:: *)

(* ::Title:: *)
(*TransformRows (0.6.1)*)


(* ::Subtitle:: *)
(*Wolfram Resource Function contributed by Daniele Gregori*)


(* ::Section:: *)
(*Package Header*)


BeginPackage["TransformRows`"];


TransformRows::usage=
"TransformRows[tab, row, transformation] transforms only a certain row of a given Tabular;
TransformRows[tab, {row1, row2, ...}, transformation] transforms a list of rows;
TransformRows[tab, crit, transformation] transforms all rows selected by the condition crit.";


Begin["`Private`"];


(* ::Section:: *)
(*Definition*)


(* ::Subsection:: *)
(*Resources*)


(* ::Input::Initialization:: *)
transformTabular=ResourceFunction["TransformTabular"];


(* ::Subsection:: *)
(*Helpers*)


(* ::Input::Initialization:: *)
slotList[fun_]:=
	Cases[fun,Verbatim[Slot][ _Integer | _String],{0,Infinity}]//Union


(* ::Input::Initialization:: *)
transformF[tab_,fun_]:=
	Block[{slots},
		slots=slotList[fun];
		If[slots==={#1},
			transformTabular[tab,fun],
			TransformColumns[tab,fun]]]


(* ::Subsection:: *)
(*Internal*)


(* ::Input::Initialization:: *)
ClearAll[iTransformRows]

iTransformRows[tab_Tabular,rows_List,fun_,ko_,schema_,opts_]:=
	Block[{allrows,complrows,subtab,newtab,perm,join},

		allrows=Range@Length[tab];
		complrows=Complement[allrows,rows];

		newtab=transformF[tab,fun][rows];
		subtab=tab[complrows];

		join=If[ko,
				perm=Permute[allrows,
							FindPermutation[
							Join[complrows,rows],allrows]];
				Join[subtab,newtab][perm],
				Join[newtab,subtab]];

		Tabular[join,schema,opts]]

iTransformRows[tab_Tabular,row_Integer,fun_,ko_,schema_,opts_]:=
	iTransformRows[tab,{row},fun,ko,schema,opts]

iTransformRows[tab_Tabular,sel_Function,fun_,ko_,schema_,opts_]:=
	iTransformRows[tab,Select[tab,sel->"Index"],fun,ko,schema,opts]


(* ::Subsection:: *)
(*Main*)


(* ::Input::Initialization:: *)
ClearAll[TransformRows]
Options[TransformRows]={"KeepOrder"->True};
TransformRows[tab_Tabular,rowsel_,fun_,opt:OptionsPattern[TransformRows]]:=
	Block[{schema,ko=OptionValue["KeepOrder"]},
	schema=KeyDrop[Normal[TabularSchema[tab]],"ColumnProperties"];
	iTransformRows[tab,rowsel,fun,ko,schema,Options[tab]]]


(* ::Section:: *)
(*Package Footer*)


End[];
EndPackage[];
