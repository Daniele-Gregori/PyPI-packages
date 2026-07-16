(* ::Package:: *)

(* ::Title:: *)
(*TranscendentalRange (1.1.0)*)


(* ::Subtitle:: *)
(*Wolfram Resource Function contributed by Daniele Gregori*)


(* ::Section:: *)
(*Package Header*)


BeginPackage["TranscendentalRange`"];


TranscendentalRange::usage=
"TranscendentalRange[x] gives all transcendental numbers of the form t = b\[CenterDot]e^a with 1 \[LessEqual] t \[LessEqual] x, where a and b are algebraics in Range[x];
TranscendentalRange[x, y] gives all transcendental numbers of the form t = b\[CenterDot]e^a with x \[LessEqual] t \[LessEqual] y, where a and b are algebraics in Range[x, y];
TranscendentalRange[x, y, s] gives all transcendental numbers of the form t = b\[CenterDot]e^a with x \[LessEqual] t \[LessEqual] y and s > 0, where a and b are algebraics in Range[x, y, s];
TranscendentalRange[x, y, s, d] requires a lower bound on the steps d.";


Begin["`Private`"];


(* ::Section:: *)
(*Definition*)


(* ::Subsection:: *)
(*Auxiliary functions*)


(* ::Subsubsection::Closed:: *)
(*FormulaComplexity*)


(* ::Input::Initialization:: *)
ClearAll[formulaComplexity]

digitSum[int_Integer]:=If[$VersionNumber>=14,DigitSum[int],Total@IntegerDigits[int]]

formulaComplexity[form_?NumericQ]:=
	N@Total[Cases[
			Inactivate[form]
		/.const_?(Quiet@MemberQ[Attributes[#],Constant]&):>1
			/.c_Complex:>ReIm[c]
				/.r:Rational[i1_Integer,i2_Integer]:>{i1,i2}
					/.Inactive[Sqrt][arg_]:>Inactive[Sqrt[{arg,arg}]]
						/.Alternatives[
							Inactive[Power][b_,Inactive[Times][m_,Inactive[Power][n_,-1]]|{m_,n_}],
							Inactive[Power][{b_},Inactive[Times][m_,Inactive[Power][n_,-1]]|{m_,n_}]]:>
								Table[b,Abs[n]+Abs[m]],
			_Integer,{0,Infinity}]/. j_Integer?NonPositive:>-j+1
				/.i_Integer:>Mean[1/2{5IntegerLength[i],digitSum[i],Total[Last/@FactorInteger[i]],Sqrt[Abs[N[i]]]}]]


(* ::Subsubsection::Closed:: *)
(*Restricted range*)


(* ::Input::Initialization:: *)
complexitySelect[list_List,c_]:=
	Select[list,formulaComplexity[#]<=c&]

ClearAll[stepSelect]
stepSelect[list_List,d_]:=
	Block[{sel,eln,i},
	eln=list[[1]];
	sel=CreateDataStructure["DynamicArray"];
	sel["Append",eln];

	For[i=2,i<=Length[list],i++,

		If[Abs[list[[i]]-eln]>=d,

			eln=list[[i]];
			sel["Append",eln],

			Continue[]]];

	Normal@sel]~Quiet~N::meprec

stepSelect[{},d_]:=
	{}

stepSelect[{},d_,t_:0]:=
	{}


(* ::Input::Initialization:: *)
restrictRange[main_,compl_,d_:0]:=
	If[d!=0,stepSelect[#,d],#]&[complexitySelect[main,compl]]


(* ::Subsection:: *)
(*Generator ranges*)


(* ::Subsubsection::Closed:: *)
(*AlgebraicRange*)


(* ::Input::Initialization:: *)
ClearAll[algebraicRange]
algebraicRange[args__]:=ResourceFunction["AlgebraicRange"][args];


(* ::Subsubsection::Closed:: *)
(*FareyRange*)


(* ::Input::Initialization:: *)
ClearAll[fareyRange]
fareyRange[r1_,r2_,r3_]:=
	If[r3>=1,
		ResourceFunction["FareyRange"][r1,r2,r3],
		Which[(*intuitive alternative reciprocal step*)
			MatchQ[r3,Rational[1,_]],
				ResourceFunction["FareyRange"][r1,r2,1/r3],
			MatchQ[r3,Rational[-1,_]],
				Reverse[ResourceFunction["FareyRange"][r2,r1,-1/r3]],
			True,
				failureFareyStep[r3]]]


(* ::Subsubsection::Closed:: *)
(*Basic range*)


(* ::Input::Initialization:: *)
range[x_,y_,z_,opt_String]:=
Switch[opt,
		"Rational",
			Range[x,y,z],
		"Algebraic",
			algebraicRange[x,y,z],
		"RationalFarey",
			fareyRange[x,y,z],
		"AlgebraicFarey",
			algebraicRange[x,y,z,"FareyRange"->True]]


(* ::Subsubsection::Closed:: *)
(*Argument failures*)


(* ::Input::Initialization:: *)
failureNotAlgebraics=
Failure["ArgsNotAlgebraic",<|"MessageTemplate"->"The range arguments provided `ls` are not all algebraic numbers.","MessageParameters"-><|"ls"->#|>|>]&;


(* ::Input::Initialization:: *)
failureFareyStep=Failure["FareyStep",<|"MessageTemplate"->"The step parameter `s` is not allowed with the Farey range option.","MessageParameters"-><|"s"->#|>|>]&;


(* ::Subsection:: *)
(*Method features*)


(* ::Subsubsection::Closed:: *)
(*Transcendental types*)


(* ::Text:: *)
(*All possible methods include the following transcendental functions types:*)


(* ::Input::Initialization:: *)
$trig={Sin,Cos,Tan,Cot,Sec,Csc};
$hyp={Sinh,Cosh,Tanh,Coth,Sech,Csch};
$invtrig={ArcSin,ArcCos,ArcTan,ArcCot,ArcSec,ArcCsc};
$invhyp={ArcSinh,ArcCosh,ArcTanh,ArcCoth,ArcSech,ArcCsch};
$trsctypes=Join[{Exp,Log,Power},$trig,$hyp,$invtrig,$invhyp];


(* ::Subsubsection::Closed:: *)
(*Domains*)


(* ::Text:: *)
(*Domains for the transcendental functions (namely, for the second arguments of Outer or monotonicOuter):*)


(* ::Input::Initialization:: *)
ClearAll[domain]
domain[_]=True&;
domain[Log]=#>0&;
domain[Power]=#>0&;
domain[Coth]=#!=0&;
domain[Csch]=#!=0&;
domain[ArcCosh]=#>=1&;
domain[ArcTanh]=-1<#<1&;
domain[ArcCoth]=Abs[#]>1&;
domain[ArcSech]=0<#<=1&;
domain[ArcCsch]=#!=0&;
domain[Tan]=Abs@FractionalPart[#/\[Pi]]!=1/2&;
domain[Sec]=Abs@FractionalPart[#/\[Pi]]!=1/2&;
domain[Cot]=FractionalPart[#/\[Pi]]!=0&;
domain[Csc]=FractionalPart[#/\[Pi]]!=0&;
domain[ArcSin]=Abs[#]<=1&;
domain[ArcCos]=Abs[#]<=1&;
domain[ArcCot]=#!=0&;
domain[ArcSec]=Abs[#]>=1&;
domain[ArcCsc]=Abs[#]>=1&;


(* ::Subsubsection::Closed:: *)
(*Turning points*)


(* ::Input::Initialization:: *)
turning[Sech,2]=1.1996786402577338339163698486411419442614587884186072089154777839181247252238474799990869921465093798859055084152992885360972178040896927668397975864210455901058557964724175753128553208906269191698657332448131052287075914748630366427440133082155799942721734256655069966143444470563164230057692043247489031265201307085790014912003399776696436986439748912548615777187811288374137950992076841745184393198667506764368371596365491614153402516039184082319627130193295329591196622738714420280530632359907736850113404270016907041313721058255097372805899876759275956586915101399092579666079294066380624790347342258402538842785309098110628017951680269086797741369710360546711669340271234106682950933277105045783881899238434275271058676877566415006355465162567059563426708149738219386927460413758456500330564478415290540016678779191401214582555423223492792011045769305102525934838833146509338813489679566942703705429701575293800309363707379469515406928261504019581432610649384677565523318326483365941764424827272107434379332462505643`1000.(*Entity["MathematicalConstant","CothFixedPointConstant"][EntityProperty["MathematicalConstant","NumericalApproximation"]]*);

rooteq[ArcSech,2]=D[x ArcSech[x],x]==0//PowerExpand//Simplify;
turning[ArcSech,2]=With[{prec=1000},FindRoot[rooteq[ArcSech,2],{x,0.5},PrecisionGoal->prec,WorkingPrecision->prec][[1,2]]];

rooteq[ArcCos,2]=D[x ArcCos[x],x]==0//PowerExpand//Simplify;
turning[ArcCos,2]=With[{prec=1000},FindRoot[rooteq[ArcCos,2],{x,0.5},PrecisionGoal->prec,WorkingPrecision->prec][[1,2]]];

rooteq[ArcSec,2]=D[ x ArcSec[x],x]==0//PowerExpand//Simplify;
turning[ArcSec,2]=With[{prec=1000},FindRoot[rooteq[ArcSec,2],{x,-10},PrecisionGoal->prec,WorkingPrecision->prec][[1,2]]];

rooteq[Power,2]=D[ (-x)^x,x]==0//Simplify;
turning[Power,2]=With[{prec=1000},FindRoot[rooteq[Power,2],{x,-0.1},PrecisionGoal->prec,WorkingPrecision->prec][[1,2]]];


(* ::Subsection:: *)
(*Method implementations*)


(* ::Subsubsection::Closed:: *)
(*General testing method*)


(* ::Text:: *)
(*The following general method is simpler to implement but also "naive" in the sense that it uses the default inefficient version of Outer. Then it can be used as a method for testing the more sophisticated efficient implementation with monotonicOuter.*)


(* ::Text:: *)
(*(It is also currently the default method for direct trigonometric functions.)*)


(* ::Input::Initialization:: *)
(*elementary testing range with only one function at a time*)

ClearAll[elemNaiveRange]

elemNaiveRange[x_,y_,z_,{fun:_Symbol|_String:Exp,1},opt_String:"Rational"]/;NumericQ[z]:=
Block[
{rgmain,min=If[x<y,x,y],max=If[x<y,y,x]},

(*arguments basic range, to be restricted below*)
rgmain=range[x,y,z,opt];

DeleteCases[
Select[

Outer[

(*function with two arguments*)
Which[
MemberQ[{Exp,Log,Splice@Join[$trig,$hyp,$invtrig,$invhyp]},fun],
	#1 fun[#2],
fun===Power,
	#2^#1,
True,
	$Failed]&,

(*first argument*)
Which[
	MemberQ[{Exp,Log,Splice@Join[$trig,$hyp,$invtrig,$invhyp]},fun],
		rgmain,
	fun===Power,
			DeleteCases[rgmain,_Rationals]],

(*second argument*)
Which[

		MemberQ[{Log,Power,Coth,Csch,ArcCosh,ArcTanh,ArcCoth,ArcSech,ArcCsch,Tan,Sec,Cot,Csc,ArcSin,ArcCos,ArcCot,ArcSec,ArcCsc},fun],
			Select[rgmain,domain[fun]],

		True,
			rgmain
		]]//Flatten,
min<=#<=max&],
_?(Element[#,Algebraics]&)]]


(* ::Input::Initialization:: *)
elemNaiveRange[x_,y_,z_,{fun:_Symbol|_String:Exp,m_Integer},opt_String:"Rational",prec_:MachinePrecision]/;m>1&&NumericQ[z]:=Block[{base,combined,op,min=If[x<y,x,y],max=If[x<y,y,x],rpl},rpl=If[fun=!=Power,Times[a_,fun[arg_]]|fun[arg_]:>Abs@N[arg,prec],Power[b_,e_]:>Abs@N[b,prec]+Abs@N[e,prec]];
base=Values@Map[First@MinimalBy[#,#/.rpl&,1]&,GroupBy[elemNaiveRange[x,y,z,{fun,1},opt],N[#,prec]&]];
op=If[fun===Power,Times,Plus];
combined=base;
Do[combined=Values@Map[First@MinimalBy[#,#/.rpl&,1]&,GroupBy[Select[Union@Flatten@Outer[op,combined,base],min<=#<=max&],N[#,prec]&]],{m-1}];
DeleteCases[combined,_?(Element[#,Algebraics]||Abs@N[#,prec]<10^(-prec+1)&)]]


ClearAll[combinedNaiveRange];

Options[combinedNaiveRange]={WorkingPrecision->MachinePrecision,"GeneratorsDomain"->Rationals,"Multiplicity"->1};

combinedNaiveRange[x_,y_,z_:1,funs_List,opts:OptionsPattern[combinedNaiveRange]]/;NumericQ[z]:=
	Block[{funL,patf,extrarg,join,
		prec=OptionValue[WorkingPrecision],
		m=OptionValue["Multiplicity"],
		optalg=optGenerators[OptionValue["GeneratorsDomain"]]},
		funL=If[funs=!={All},
			Flatten@funs,
			$trsctypes];

		patf=_?(MemberQ[$trsctypes,#]&);
		extrarg=#/.Times[a_, patf[arg_]]|patf[arg_]:>Abs@N[arg,prec]&;

		join=If[MemberQ[{1,{1}},m],
			Join@@Map[elemNaiveRange[x,y,z,{#,m},optalg]&,funL],
			Flatten[Join@@@Map[elemNaiveRange[x,y,z,{#,m},optalg]&,{
						Cases[funL,Alternatives@@Join[{Exp},$trig,$hyp]],
						Cases[funL,Alternatives@@Join[{Log},$invtrig,$invhyp]],
						Cases[funL,Power]},{2}],1]];

		Values@Map[First@SortBy[#,extrarg]&,GroupBy[join,N[#,prec]&]]]


sortNaiveRange[x_,y_,z_:1,fun:_Symbol|_String|_List|All,opts:OptionsPattern[combinedNaiveRange]]/;NumericQ[z]:=
	If[z>0,SortBy,ReverseSortBy][combinedNaiveRange[x,y,z,{fun},opts],N[#,OptionValue[WorkingPrecision]]&]


(* ::Subsubsection::Closed:: *)
(*Monotonic outer*)


(* ::Input::Initialization:: *)
monotonicOuter[fun_Function,{min_,max_},ls1_List,ls2_List,{mi_,mj_},sign_]:=
	Block[{i,j,new,begin,end,rgQ,outer},

		outer=CreateDataStructure["DynamicArray"];

		For[
			Switch[mi,
					1,i=1,
					-1,i=Length[ls1]],
			Switch[mi,
					1,i<=Length[ls1],
					-1,i>=1],
			Switch[mi,
					1,i++,
					-1,i--],

			begin=True;

			end=False;

			For[

				Switch[mj,
					1,j=1,
					-1,j=Length[ls2]],
				Switch[mj,
					1,j<=Length[ls2],
					-1,j>=1],
				Switch[mj,
					1,j++,
					-1,j--],



				new=fun[ls1[[i]],ls2[[j]]];

				rgQ=min<=new<=max;

				If[rgQ,
					begin=False];

				Which[new>max&&sign===1,
						end=True,
					new<min&&sign===-1,
						end=True];

				Which[
					rgQ&&!begin,
						outer["Append",new],
					begin&&!end,
						(*this should be avoided with better splitting*)
						Echo[{"redundant",ls1[[i]],ls2[[j]]}];
						Continue[],
					True,
						Break[]]]];

		Normal@outer]//QuietEcho


(* ::Subsubsection::Closed:: *)
(*Range preprocessing*)


(* ::Input::Initialization:: *)
(*splitRange:split a sorted list into segments at given points*)
ClearAll[splitRange]
splitRange[list_List,splitPts_List,z_,segs_,count_Symbol]:=Block[{pts,bounds,n,seg},pts=Sort[splitPts];
bounds=Join[{-Infinity},pts,{Infinity}];
n=Length[bounds]-1;
count=n;
Do[seg=Select[list,bounds[[k]]<=#<=bounds[[k+1]]&];
segs[k]=If[z>0,seg,Reverse[seg]],{k,n}];]
(*assignment to symbolic argument is OK*)


(* ::Input::Initialization:: *)
(*preprRange:prepare coefficient and argument ranges for monotonicOuter.
#1 (coefficient) takes values in the whole range[x,y,z].
#2 (argument to f) takes values only where f is defined.
Each range is split independently at its own critical points for efficient Break[] in monotonicOuter.*)
ClearAll[preprRange]

preprRange[
{x_,y_,z_,opt_},
{cAlg_,cSing_,cSplit_},
{aDom_,aAlg_,aSing_,aSplit_},
{rc_,ra_,cs_,as_,ncs_,nas_}]:=

Block[{base},

base=range[x,y,z,opt];

(*Coefficient range:full range minus algebraic/singular points*)
rc=base;
If[cAlg=!=None,rc=DeleteCases[rc,cAlg]];
If[Length[cSing]>0,rc=DeleteCases[rc,Alternatives@@cSing]];

(*Argument range:domain-restricted,minus algebraic/singular*)
ra=Select[base,aDom];
If[aAlg=!=None,ra=DeleteCases[ra,aAlg]];
If[Length[aSing]>0,ra=DeleteCases[ra,Alternatives@@aSing]];

(*no output since this merely sets block symbols*)
splitRange[rc,cSplit,z,cs,ncs];
splitRange[ra,aSplit,z,as,nas];]


(* ::Subsubsection::Closed:: *)
(*Method specifications*)


(* ::Input::Initialization:: *)
(*Each entry encodes the varying parts of a coreRange definition:
  "coeff" -> {cAlg, cSing, cSplit}  -- coefficient preprocessing parameters
  "arg"   -> {aAlg, aSing, aSplit}  -- argument preprocessing parameters
  "outp"  -> {{ls1Idx, ls2Idx, {mi,mj}, sign}, ...}  -- monotonicOuter tuples for positive range
  "outn"  -> {{ls1Idx, ls2Idx, {mi,mj}, sign}, ...}  -- monotonicOuter tuples for negative range
For non-Power methods: ls1 = coefficient segments (cs), ls2 = argument segments (as).
For Power: ls1 = argument segments (as), ls2 = coefficient segments (cs) [swapped].*)

$methodSpecs=<|

Exp-><|
	"coeff"->{0,{},{0}},
	"arg"->{0,{},{-1}},
	"outp"->{{2,1,{1,-1},1},{2,2,{1,1},1}},
	"outn"->{{1,1,{-1,-1},-1},{1,2,{-1,1},-1}}|>,

Log-><|
	"coeff"->{0,{},{0}},
	"arg"->{1,{},{1/E,1}},
	"outp"->{{1,1,{-1,1},-1},{1,2,{-1,-1},1},{2,3,{1,1},1}},
	"outn"->{{2,1,{1,-1},-1},{2,2,{1,1},1},{1,3,{-1,1},-1}}|>,

Sinh-><|
	"coeff"->{0,{},{0}},
	"arg"->{0,{},{0}},
	"outp"->{{1,1,{-1,-1},1},{2,2,{1,1},1}},
	"outn"->{{2,1,{1,-1},-1},{1,2,{-1,1},-1}}|>,

Cosh-><|
	"coeff"->{0,{},{0}},
	"arg"->{0,{},{0}},
	"outp"->{{2,2,{1,1},1},{2,1,{1,-1},1}},
	"outn"->{{1,2,{-1,1},-1},{1,1,{-1,-1},-1}}|>,

Tanh-><|
	"coeff"->{0,{},{0}},
	"arg"->{0,{},{0}},
	"outp"->{{2,2,{1,1},1},{1,1,{-1,-1},1}},
	"outn"->{{1,2,{-1,1},-1},{2,1,{1,-1},-1}}|>,

Coth-><|
	"coeff"->{0,{},{0}},
	"arg"->{0,{0},{0}},
	"outp"->{{2,2,{1,-1},1},{1,1,{-1,1},1}},
	"outn"->{{1,2,{-1,-1},-1},{2,1,{1,1},-1}}|>,

Sech-><|
	"coeff"->{0,{},{0}},
	"arg"->{0,{},{-turning[Sech,2],0,turning[Sech,2]}},
	"outp"->{{2,4,{1,1},1},{2,3,{1,1},1},{2,2,{1,-1},1},{2,1,{1,-1},1}},
	"outn"->{{1,4,{-1,1},-1},{1,3,{-1,1},-1},{1,2,{-1,-1},-1},{1,1,{-1,-1},-1}}|>,

Csch-><|
	"coeff"->{0,{},{0}},
	"arg"->{0,{0},{0}},
	"outp"->{{2,2,{1,1},-1},{1,1,{-1,-1},-1}},
	"outn"->{{1,2,{-1,1},1},{2,1,{1,-1},1}}|>,

ArcSinh-><|
	"coeff"->{0,{},{0}},
	"arg"->{0,{},{0}},
	"outp"->{{1,1,{-1,-1},1},{2,2,{1,1},1}},
	"outn"->{{2,1,{1,-1},-1},{1,2,{-1,1},-1}}|>,

ArcCosh-><|
	"coeff"->{0,{},{0}},
	"arg"->{1,{},{}},
	"outp"->{{2,1,{1,1},1}},
	"outn"->{{1,1,{-1,1},-1}}|>,

ArcTanh-><|
	"coeff"->{0,{},{0}},
	"arg"->{0,{},{0}},
	"outp"->{{1,1,{-1,-1},1},{2,2,{1,1},1}},
	"outn"->{{2,1,{1,-1},-1},{1,2,{-1,1},-1}}|>,

ArcCoth-><|
	"coeff"->{0,{},{0}},
	"arg"->{0,{-1,1},{-1,1}},
	"outp"->{{2,3,{1,-1},1},{1,1,{-1,1},1}},
	"outn"->{{1,3,{-1,-1},-1},{2,1,{1,1},-1}}|>,

ArcSech-><|
	"coeff"->{0,{},{0}},
	"arg"->{1,{0},{turning[ArcSech,2]}},
	"outp"->{{2,1,{1,-1},1},{2,2,{1,-1},1}},
	"outn"->{{1,1,{-1,-1},-1},{1,2,{-1,-1},-1}}|>,

ArcCsch-><|
	"coeff"->{0,{},{0}},
	"arg"->{0,{0},{0}},
	"outp"->{{2,2,{1,-1},1},{1,1,{-1,1},1}},
	"outn"->{{1,2,{-1,-1},-1},{2,1,{1,1},-1}}|>,

ArcSin-><|
	"coeff"->{0,{},{0}},
	"arg"->{0,{},{0}},
	"outp"->{{1,1,{-1,-1},1},{2,2,{1,1},1}},
	"outn"->{{2,1,{1,-1},-1},{1,2,{-1,1},-1}}|>,

ArcCos-><|
	"coeff"->{0,{},{0}},
	"arg"->{1,{},{turning[ArcCos,2]}},
	"outp"->{{2,2,{1,-1},1},{2,1,{1,-1},1}},
	"outn"->{{1,2,{-1,-1},-1},{1,1,{-1,-1},-1}}|>,

ArcTan-><|
	"coeff"->{0,{},{0}},
	"arg"->{0,{},{0}},
	"outp"->{{1,1,{-1,-1},1},{2,2,{1,1},1}},
	"outn"->{{2,1,{1,-1},-1},{1,2,{-1,1},-1}}|>,

ArcCot-><|
	"coeff"->{0,{},{0}},
	"arg"->{0,{0},{0}},
	"outp"->{{2,2,{1,-1},1},{1,1,{-1,1},1}},
	"outn"->{{1,2,{-1,-1},-1},{2,1,{1,1},-1}}|>,

ArcSec-><|
	"coeff"->{0,{},{0}},
	"arg"->{1,{},{turning[ArcSec,2],-1,1}},
	"outp"->{{2,4,{1,1},1},{2,2,{1,1},1},{2,1,{1,1},1}},
	"outn"->{{1,4,{-1,1},-1},{1,2,{-1,1},-1},{1,1,{-1,1},-1}}|>,

ArcCsc-><|
	"coeff"->{0,{},{0}},
	"arg"->{0,{},{-1,1}},
	"outp"->{{2,3,{1,-1},1},{1,1,{-1,1},1}},
	"outn"->{{1,3,{-1,-1},-1},{2,1,{1,1},-1}}|>,

Power-><|
	"coeff"->{0,{},{turning[Power,2],0}},
	"arg"->{1,{},{1}},
	"outp"->{{1,3,{1,1},1},{2,3,{1,1},1},{1,2,{1,-1},1},{2,2,{1,-1},1},{1,1,{1,-1},1},{2,1,{1,-1},1}},
	"outn"->{}|>

|>;


(* ::Subsubsection::Closed:: *)
(*Core range*)


(* ::Input::Initialization:: *)
ClearAll[computeOuter]
computeOuter[fun_Function,{min_,max_},ls1_,ls2_,tuples_List]:=
	If[tuples==={},
		{},
		Join@@(monotonicOuter[fun,{min,max},ls1[#[[1]]],ls2[#[[2]]],#[[3]],#[[4]]]&/@tuples)]


(* ::Input::Initialization:: *)
ClearAll[coreRange]

coreRange[{f_,1},x_,y_,z_:1,opt_String:"Rational"]/;KeyExistsQ[$methodSpecs,f]:=
Block[{spec=$methodSpecs[f],fun,min,max,rc,ra,cs,as,ncs,nas,
	cAlg,cSing,cSplit,aAlg,aSing,aSplit,ls1,ls2,outp,outn},

	fun=If[f===Power,#1^#2&,# f[#2]&];
	min=If[z>0,x,y];
	max=If[z>0,y,x];

	{cAlg,cSing,cSplit}=spec["coeff"];
	{aAlg,aSing,aSplit}=spec["arg"];

	preprRange[
		{x,y,z,opt},
		{cAlg,cSing,cSplit},
		{domain[f],aAlg,aSing,aSplit},
		{rc,ra,cs,as,ncs,nas}];

	(*Power: filter coefficient segments to non-rationals*)
	If[f===Power,
		Do[cs[k]=Select[cs[k],!Element[#,Rationals]&],{k,ncs}]];

	{ls1,ls2}=If[f===Power,{as,cs},{cs,as}];

	outp:=computeOuter[fun,{min,max},ls1,ls2,spec["outp"]];
	outn:=computeOuter[fun,{min,max},ls1,ls2,spec["outn"]];

	Which[
		0<=x<=y||0<=y<=x,outp,
		x<=y<=0||y<=x<=0,outn,
		x<=0&&y>=0,outn~Join~outp,
		y<=0&&x>=0,outp~Join~outn]]


(* ::Subsection:: *)
(*Main definition*)


(* ::Subsubsection::Closed:: *)
(*Options*)


(* ::Input::Initialization:: *)
ClearAll[TranscendentalRange]

Options[TranscendentalRange]={Method->Exp,"GeneratorsDomain"->Rationals,"FareyRange"->False,"FormulaComplexity"->Infinity,WorkingPrecision->MachinePrecision,"Multiplicity"->1,"Test"->False(*development only option*)};


(* ::Input::Initialization:: *)
ClearAll[optGenerators]
optGenerators[opts:OptionsPattern[TranscendentalRange]]:=
	Block[{plain},
		plain=Switch[OptionValue["GeneratorsDomain"],
						Rationals|{Rationals,Reals},
							"Rational",
						Algebraics|{Algebraics,Reals},
							"Algebraic"];
		If[!OptionValue["FareyRange"],
				plain,
				plain<>"Farey"]]
optGenerators[str_String]:=str


(* ::Subsubsection::Closed:: *)
(*Semifinal range*)


ClearAll[transcendentalRangeSingle]

Options[transcendentalRangeSingle]=Options[TranscendentalRange];

transcendentalRangeSingle[{f_,i_},x_,y_,z_:1,opts:OptionsPattern[transcendentalRangeSingle]]:=

Block[{prec=OptionValue[WorkingPrecision],rg,rpl,sby,ddby},

		(*raw redundant range*)
		rg=coreRange[{f,i},x,y,z,optGenerators[opts]];

		(*delete duplicates by N + choose the one with minimal argument*)
		rpl=Switch[i,
					{1,0},f[arg_]:>Abs@N[arg,prec],
					1,Times[a_, f[arg_]]|f[arg_]:>Abs@N[arg,prec]];
		ddby=Values@Map[
			First@MinimalBy[#,#/.rpl&,1]&,

			GroupBy[rg,N[#,prec]&]];

		(*(rev)sort by N*)
		sby=If[z>0,SortBy,ReverseSortBy];
		sby[ddby,N[#,prec]&]]


(* ::Input::Initialization:: *)
ClearAll[transcendentalRangeMultiple]

transcendentalRangeMultiple[fullrange_List,{method_,multi_},{x_,y_,z_},prec_:$MachinePrecision]:=DeleteCases[
If[z>0,SortBy,ReverseSortBy][DeleteDuplicatesBy[combineMultiplicity[fullrange,{method,multi},{x,y,z},prec],N[#,prec]&],N[#,prec]&],_?(Element[#,Algebraics]||Abs@N[#,prec]<10^(-prec+1)&)]


(* ::Subsubsection::Closed:: *)
(*Combined ranges*)


ClearAll[combineMultiplicity]
combineMultiplicity[range_List,{f_,ord_Integer},{x_,y_,z_},prec_:$MachinePrecision]/;ord>2:=
	Block[{raw,rpl},
		raw=monotonicOuter[
			If[f=!=Power,Plus[#1,#2]&,Times[#1,#2]&],
			{Min[x,y],Max[x,y]},
			If[z>0,#,Reverse@#]&@combineMultiplicity[range,{f,ord-1},{x,y,z},prec],
			If[z>0,#,Reverse@#]&@range,
			If[z>0,{1,1},{-1,-1}],
			If[z>0,1,-1]];
		rpl=If[f=!=Power,
			Times[a_,f[arg_]]|f[arg_]:>Abs@N[arg,prec],
			Power[base_,exp_]:>Abs@N[base,prec]+Abs@N[exp,prec]];
		Values@Map[First@MinimalBy[#,#/.rpl&,1]&,GroupBy[raw,N[#,prec]&]]]
combineMultiplicity[range_List,{f_,2},{x_,y_,z_},prec_:$MachinePrecision]:=
	Block[{raw,rpl},
		raw=monotonicOuter[
			If[f=!=Power,Plus[#1,#2]&,Times[#1,#2]&],
			{Min[x,y],Max[x,y]},
			If[z>0,#,Reverse@#]&@range,
			If[z>0,#,Reverse@#]&@range,
			If[z>0,{1,1},{-1,-1}],
			If[z>0,1,-1]];
		rpl=If[f=!=Power,
			Times[a_,f[arg_]]|f[arg_]:>Abs@N[arg,prec],
			Power[base_,exp_]:>Abs@N[base,prec]+Abs@N[exp,prec]];
		Values@Map[First@MinimalBy[#,#/.rpl&,1]&,GroupBy[raw,N[#,prec]&]]]
combineMultiplicity[range_List,{f_,1},{x_,y_,z_},prec_:$MachinePrecision]:=
	range


(* ::Input::Initialization:: *)
ClearAll[combineMethod]
combineMethod[funL_List,x_,y_,z_,d_,opts:OptionsPattern[TranscendentalRange]]:=
	With[{prec=OptionValue[WorkingPrecision]},
	If[z>0,SortBy,ReverseSortBy][
	DeleteDuplicatesBy[
	Flatten[
Map[TranscendentalRange[x,y,z,d,Method->#,opts]&,
	funL],1],
		N[#,prec]&],
	N[#,prec]&]]


(* ::Subsubsection::Closed:: *)
(*Final range*)


(* ::Input::Initialization:: *)
Clear[TranscendentalRange]

TranscendentalRange[x_,opts:OptionsPattern[TranscendentalRange]]:=
	TranscendentalRange[1,x,1,0,opts]

TranscendentalRange[x_,y_,z_:1,d_:0,opts:OptionsPattern[TranscendentalRange]]/;NumericQ[d]&&NumericQ[z]:=
	Block[{optmeth,optmulti,optprec,optcompl,opttest,optalg,fullrange,restrcompl,restrstep},

	If[!Element[{x,y,z},Algebraics],
			Return@failureNotAlgebraics[{x,y,z}]];

	{optmeth,optmulti,optprec,optcompl,opttest}=
	OptionValue[{Method,"Multiplicity",WorkingPrecision,"FormulaComplexity","Test"}];

	optalg=optGenerators[opts];

	fullrange=
	If[opttest||MemberQ[$trig,optmeth],


		sortNaiveRange[
			x,y,z,
			optmeth,
			"Multiplicity"->optmulti,
			"GeneratorsDomain"->optalg,
			WorkingPrecision->optprec],


		Which[

				KeyExistsQ[$methodSpecs,optmeth],
				Which[
			optmulti===1,
			#,

			IntegerQ[optmulti]&&optmulti>=2,
				transcendentalRangeMultiple[
				#,
				{optmeth,optmulti},
				{x,y,z},
				optprec],

			True,
			$Failed]&@
						transcendentalRangeSingle[
							{optmeth,1},
							x,y,z,
							opts],

				ListQ[optmeth],
			combineMethod[optmeth,x,y,z,d,opts],

				optmeth===All,
			combineMethod[$trsctypes,x,y,z,d,opts],

				True,
				$Failed]];
			

		restrcompl=If[optcompl<Infinity,
			complexitySelect[fullrange,optcompl],
			fullrange];

		restrstep=If[d>0,
			stepSelect[restrcompl,d],
			restrcompl]]


(* ::Section:: *)
(*Package Footer*)


End[];
EndPackage[];
