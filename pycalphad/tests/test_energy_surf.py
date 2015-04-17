"""
The energy surface test module verifies that energy_surf() calculates
the energy surface correctly.
"""

import nose.tools
from pycalphad import Database, energy_surf

TDB_TEST_STRING = """
ELEMENT /-          ELECTRON_GAS         0         0         0 !
ELEMENT VA                VACUUM         0         0         0 !
ELEMENT AL                FCC_A1    26.982      4540      28.3 !
ELEMENT CR                BCC_A2    51.996      4050    23.543 !
ELEMENT NI                FCC_A1     58.69      4787    29.796 !

$------------------------------------------------------------------------------
 FUNCTION GHSERAL  298.15    -7976.15+137.093038*T-24.3671976*T*LN(T)
    -1.884662E-3*T**2-0.877664E-6*T**3+74092*T**(-1);                     700 Y
    -11276.24+223.048446*T-38.5844296*T*LN(T)+18.531982E-3*T**2
    -5.764227E-6*T**3+74092*T**(-1);                                   933.47 Y
    -11278.378+188.684153*T-31.748192*T*LN(T)-1230.524E25*T**(-9);     2900 N !
    
 FUNCTION GALBCC   298.15    +2106.85+132.280038*T-24.3671976*T*LN(T)
    -1.884662E-3*T**2-0.877664E-6*T**3+74092*T**(-1);                     700 Y
    -1193.24+218.235446*T-38.5844296*T*LN(T)+18.531982E-3*T**2-5.764227E-6*T**3
    +74092*T**(-1);                                                    933.47 Y
    -1195.378+183.871153*T-31.748192*T*LN(T)-1230.524E25*T**(-9);      2900 N !
    
 FUNCTION GALLIQ   298.15     +3028.879+125.251171*T-24.3671976*T*LN(T)
    -1.884662E-3*T**2-0.877664E-6*T**3+74092*T**(-1)+79.337E-21*T**7;     700 Y
    -271.21+211.206579*T-38.5844296*T*LN(T)+18.531982E-3*T**2-5.764227E-6*T**3
    +74092*T**(-1)+79.337E-21*T**7;                                    933.47 Y
    -795.996+177.430178*T-31.748192*T*LN(T);                           2900 N !


 FUNCTION GHSERCR   298.15    -8856.94+157.48*T-26.908*T*LN(T)+1.89435E-3*T**2
    -1.47721E-6*T**3+139250*T**(-1);                                     2180 Y
    -34869.344+344.18*T-50*T*LN(T)-2885.26E29*T**(-9);                 6000 N !
$ PARAMETER  TC(BCC_A2,CR:VA;0)   298.15  -311.50;  6000.00 N !
$ PARAMETER  BM(BCC_A2,CR:VA;0)   298.15    -0.008;  6000.00 N !

 FUNCTION GCRLIQ   298.15      +15483.015+146.059775*T-26.908*T*LN(T)
    +1.89435E-3*T**2-1.47721E-6*T**3+139250*T**(-1)+237.615E-23*T**7;    2180 Y
    -16459.984+335.616316*T-50*T*LN(T);                                6000 N !
    
 FUNCTION GCRFCC   298.15      -1572.94+157.643*T-26.908*T*LN(T)
    +1.89435E-3*T**2-1.47721E-6*T**3+139250*T**(-1);                     2180 Y
    -27585.344+344.343*T-50*T*LN(T)-2885.26E29*T**(-9);                6000 N !
 
 FUNCTION GHSERNI   298.15     -5179.159+117.854*T-22.096*T*LN(T)
     -4.8407E-3*T**2;                                                    1728 Y
    -27840.655+279.135*T-43.1*T*LN(T)+1127.54E28*T**(-9);              3000 N !
 
 FUNCTION GNILIQ   298.15      +11235.527+108.457*T-22.096*T*LN(T)
    -4.8407E-3*T**2-382.318E-23*T**7;                                    1728 Y
    -9549.775+268.598*T-43.1*T*LN(T);                                  3000 N !
    
 FUNCTION GNIBCC   298.15      +3535.925+114.298*T-22.096*T*LN(T)
    -4.8407E-3*T**2;                                                     1728 Y
    -19125.571+275.579*T-43.1*T*LN(T)+1127.54E28*T**(-9);              3000 N !
 

 FUNCTION L0ALNI   298.15  +ZERO;                             6000 N !
 FUNCTION L1ALNI   298.15  +7204-3.743*T;                     6000 N !
 FUNCTION L0ALCR   298.15  +ZERO;                             6000 N !
 FUNCTION L1ALCR   298.15  +ZERO;                             6000 N !
 FUNCTION L0CRNI   298.15  +ZERO;                             6000 N !
 FUNCTION L1CRNI   298.15  +ZERO;                             6000 N !
 FUNCTION U1ALNI   298.15  -14808.67+2.93067*T;               6000 N !
 FUNCTION U1CRNI   298.15  -1980;                             6000 N !
 FUNCTION U1ALCR   298.15  -830;                              6000 N !
 FUNCTION UALCRNI2 298.15  +U1ALCR+2*U1ALNI+2*U1CRNI+6650;    6000 N !
 FUNCTION ALPHALVA 298.15  +10000-T;                          6000 N !
 FUNCTION ALPHNIVA 298.15  +162397-27.406*T;                  6000 N !
 FUNCTION ALPHALNI 298.15  -152397+26.406*T;                  6000 N !
 FUNCTION LAMBALVA 298.15  +150000;                           6000 N !
 FUNCTION LAMBNIVA 298.15  -64024+26.494*T;                   6000 N !
 FUNCTION LAMBALNI 298.15  -52441+11.301*T;                   6000 N !
 FUNCTION ZERO     298.15  +0;                                6000 N !
 FUNCTION UN_ASS   298.15  +0;                                6000 N !


$------------------------------------------------------------------------------
 TYPE_DEFINITION % SEQ *!
 DEFINE_SYSTEM_DEFAULT ELEMENT 2 !
 DEFAULT_COMMAND DEF_SYS_ELEMENT VA /- !
 
$------------------------------------------------------------------------------

TYPE_DEFINITION & GES A_P_D BCC_A2   MAGNETIC -1       0.4!
TYPE_DEFINITION ) GES A_P_D B2       DIS_PART BCC_A2   ,,,!
TYPE_DEFINITION ( GES A_P_D FCC_A1   MAGNETIC -3      0.28!
TYPE_DEFINITION ' GES A_P_D L12_FCC  DIS_PART FCC_A1   ,,,!

$------------------------------------------------------------------------------
$ PARAMETERS FOR LIQUID PHASE
$------------------------------------------------------------------------------
 PHASE LIQUID % 1 1 !
   CONSTITUENT LIQUID :AL,CR,NI: !
  PARAMETER G(LIQUID,AL;0)          298.15  +GALLIQ;            6000 N !
  PARAMETER G(LIQUID,CR;0)          298.15  +GCRLIQ;            6000 N !
  PARAMETER G(LIQUID,NI;0)          298.15  +GNILIQ;            6000 N !
  PARAMETER G(LIQUID,AL,CR;0)       298.15  -29000;             6000 N !
  PARAMETER G(LIQUID,AL,CR;1)       298.15  -11000;             6000 N ! 
  PARAMETER G(LIQUID,AL,NI;0)       298.15  -207109+41.315*T;   6000 N ! 
  PARAMETER G(LIQUID,AL,NI;1)       298.15  -10186+5.871*T;     6000 N ! 
  PARAMETER G(LIQUID,AL,NI;2)       298.15  +81205-31.957*T;    6000 N ! 
  PARAMETER G(LIQUID,AL,NI;3)       298.15  +4365-2.516*T;      6000 N ! 
  PARAMETER G(LIQUID,AL,NI;4)       298.15  -22101.64+13.163*T; 6000 N ! 
  PARAMETER G(LIQUID,CR,NI;0)       298.15  +318-7.33*T;        6000 N ! 
  PARAMETER G(LIQUID,CR,NI;1)       298.15  +16941-6.37*T;      6000 N ! 
  PARAMETER G(LIQUID,AL,CR,NI;0)    298.15  +16000;             6000 N !
  PARAMETER G(LIQUID,AL,CR,NI;1)    298.15  +16000;             6000 N !
  PARAMETER G(LIQUID,AL,CR,NI;2)    298.15  +16000;             6000 N !

$------------------------------------------------------------------------------
$ PARAMETERS FOR BCC PHASE
$------------------------------------------------------------------------------

 PHASE BCC_A2 %& 1 1 !
   CONSTITUENT BCC_A2 :AL,CR,NI,VA: !
  PARAMETER G(BCC_A2,AL;0)        298.15  +GALBCC;              6000 N !
  PARAMETER G(BCC_A2,CR;0)        298.15  +GHSERCR;             6000 N !
  PARAMETER G(BCC_A2,NI;0)        298.15  +GNIBCC;              6000 N !
  PARAMETER G(BCC_A2,VA;0)        298.15  +300;                 6000 N ! 
  PARAMETER G(BCC_A2,AL,VA;0)     298.15  +ALPHALVA+LAMBALVA;   6000 N ! 
  PARAMETER G(BCC_A2,CR,VA;0)     298.15  +100000;              6000 N ! 
  PARAMETER G(BCC_A2,NI,VA;0)     298.15  +ALPHNIVA+LAMBNIVA;   6000 N ! 
  PARAMETER G(BCC_A2,AL,CR;0)     298.15  -54900+10*T;          6000 N ! 
  PARAMETER G(BCC_A2,AL,NI;0)     298.15  +ALPHALNI+LAMBALNI;   6000 N ! 
  PARAMETER G(BCC_A2,CR,NI;0)     298.15  +17170-11.82*T;       6000 N ! 
  PARAMETER G(BCC_A2,CR,NI;1)     298.15  +34418-11.858*T;      6000 N ! 
  PARAMETER TC(BCC_A2,CR;0)       298.15  -311.50;              6000 N ! 
  PARAMETER BMAGN(BCC_A2,CR;0)    298.15  -0.008;               6000 N ! 
  PARAMETER TC(BCC_A2,NI;0)       298.15   +575;                6000 N ! 
  PARAMETER BMAGN(BCC_A2,NI;0)    298.15   +0.85;               6000 N ! 
  PARAMETER TC(BCC_A2,CR,NI;0)       298.15   +2373;            6000 N ! 
  PARAMETER TC(BCC_A2,CR,NI;1)       298.15   +617;             6000 N ! 
  PARAMETER BMAGN(BCC_A2,CR,NI;0)    298.15   +4;               6000 N ! 
  PARAMETER G(BCC_A2,AL,CR,NI;0)     298.15   +42500;           6000 N ! 
  
 PHASE B2 %) 2 0.5 0.5 !
   CONSTITUENT B2 : AL,CR,NI,VA : AL,CR,NI,VA: !
  PARAMETER G(B2,AL:AL;0)  298.15  +ZERO;                       6000 N !
  PARAMETER G(B2,CR:CR;0)  298.15  +ZERO;                       6000 N !
  PARAMETER G(B2,NI:NI;0)  298.15  +ZERO;                       6000 N !
  PARAMETER G(B2,VA:VA;0)  298.15  +300;                        6000 N ! 
  PARAMETER G(B2,AL:VA;0)  298.15  +0.5*ALPHALVA-0.5*LAMBALVA;  6000 N ! 
  PARAMETER G(B2,VA:AL;0)  298.15  +0.5*ALPHALVA-0.5*LAMBALVA;  6000 N ! 
  PARAMETER G(B2,CR:VA;0)  298.15  +ZERO;                       6000 N !
  PARAMETER G(B2,VA:CR;0)  298.15  +ZERO;                       6000 N !
  PARAMETER G(B2,NI:VA;0)  298.15  +0.5*ALPHNIVA-0.5*LAMBNIVA;  6000 N ! 
  PARAMETER G(B2,VA:NI;0)  298.15  +0.5*ALPHNIVA-0.5*LAMBNIVA;  6000 N ! 
  PARAMETER G(B2,CR:AL;0)  298.15  -2000;                       6000 N !
  PARAMETER G(B2,AL:CR;0)  298.15  -2000;                       6000 N !
  PARAMETER G(B2,NI:AL;0)  298.15  +0.5*ALPHALNI-0.5*LAMBALNI;  6000 N ! 
  PARAMETER G(B2,AL:NI;0)  298.15  +0.5*ALPHALNI-0.5*LAMBALNI;  6000 N ! 
  PARAMETER G(B2,NI:CR;0)  298.15  +4000;                       6000 N !
  PARAMETER G(B2,CR:NI;0)  298.15  +4000;                       6000 N !





$------------------------------------------------------------------------------
$ PARAMETERS FOR FCC PHASE
$------------------------------------------------------------------------------
 PHASE FCC_A1 %( 1 1 !
   CONSTITUENT FCC_A1 :AL,CR,NI: !
  PARAMETER G(FCC_A1,AL;0)         298.15  +GHSERAL;            6000 N !
  PARAMETER G(FCC_A1,CR;0)         298.15  +GCRFCC;             6000 N !
  PARAMETER TC(FCC_A1,CR;0)        298.15  -1109.00;            6000 N !
  PARAMETER BMAGN(FCC_A1,CR;0)     298.15  -2.46;               6000 N !
  PARAMETER G(FCC_A1,NI;0)         298.15  +GHSERNI;            6000 N !
  PARAMETER TC(FCC_A1,NI;0)        298.15  +633;                6000 N ! 
  PARAMETER BMAGN(FCC_A1,NI;0)     298.15  +0.52;               6000 N ! 
  PARAMETER G(FCC_A1,AL,CR;0)      298.15  -45900+6*T;          6000 N ! 
  PARAMETER G(FCC_A1,AL,NI;0)      298.15  -162408+16.213*T;    6000 N ! 
  PARAMETER G(FCC_A1,AL,NI;1)      298.15  +73418-34.914*T;     6000 N ! 
  PARAMETER G(FCC_A1,AL,NI;2)      298.15  +33471-9.837*T;      6000 N ! 
  PARAMETER G(FCC_A1,AL,NI;3)      298.15  -30758+10.253*T;     6000 N ! 
  PARAMETER TC(FCC_A1,AL,NI;0)     298.15  -1112;               6000 N ! 
  PARAMETER TC(FCC_A1,AL,NI;1)     298.15  +1745;               6000 N ! 
  PARAMETER G(FCC_A1,CR,NI;0)      298.15  +8030-12.880*T;      6000 N ! 
  PARAMETER G(FCC_A1,CR,NI;1)      298.15  +33080-16.036*T;     6000 N ! 
  PARAMETER TC(FCC_A1,CR,NI;0)     298.15  -3605;               6000 N ! 
  PARAMETER BMAGN(FCC_A1,CR,NI;0)  298.15  -1.91;               6000 N ! 
  PARAMETER G(FCC_A1,AL,CR,NI;0)   298.15  +30300;              6000 N !

 PHASE  L12_FCC %' 2  0.75  0.25 !
   CONSTITUENT L12_FCC :AL,CR,NI: AL,CR,NI : !
  PARAMETER G(L12_FCC,AL:AL;0)      298.15  +ZERO;              6000 N !
  PARAMETER G(L12_FCC,CR:CR;0)      298.15  +ZERO;              6000 N !
  PARAMETER G(L12_FCC,NI:NI;0)      298.15  +ZERO;              6000 N !
  PARAMETER G(L12_FCC,AL:NI;0)      298.15  +3*U1ALNI;          6000 N !
  PARAMETER G(L12_FCC,NI:AL;0)      298.15  +3*U1ALNI;          6000 N !
  PARAMETER G(L12_FCC,AL:CR;0)      298.15  +3*U1ALCR;          6000 N !
  PARAMETER G(L12_FCC,CR:AL;0)      298.15  +3*U1ALCR;          6000 N !
  PARAMETER G(L12_FCC,CR:NI;0)      298.15  +3*U1CRNI;          6000 N !
  PARAMETER G(L12_FCC,NI:CR;0)      298.15  +3*U1CRNI;          6000 N !

  PARAMETER G(L12_FCC,AL,NI:AL;0)  298.15   +6*U1ALNI;          6000 N !
  PARAMETER G(L12_FCC,AL,NI:NI;0)  298.15   +6*U1ALNI;          6000 N !
  PARAMETER G(L12_FCC,AL,NI:AL;1)  298.15   +3*L1ALNI;          6000 N !
  PARAMETER G(L12_FCC,AL,NI:NI;1)  298.15   +3*L1ALNI;          6000 N !
  PARAMETER G(L12_FCC,AL:AL,NI;0)  298.15   +L0ALNI;            6000 N ! 
  PARAMETER G(L12_FCC,NI:AL,NI;0)  298.15   +L0ALNI;            6000 N ! 
  PARAMETER G(L12_FCC,CR:AL,NI;0)  298.15   +L0ALNI;            6000 N ! 
  PARAMETER G(L12_FCC,AL:AL,NI;1)  298.15   +L1ALNI;            6000 N !
  PARAMETER G(L12_FCC,NI:AL,NI;1)  298.15   +L1ALNI;            6000 N !
  PARAMETER G(L12_FCC,CR:AL,NI;1)  298.15   +L1ALNI;            6000 N !
$
  PARAMETER G(L12_FCC,AL,CR:AL;0)  298.15   +6*U1ALCR;          6000 N !
  PARAMETER G(L12_FCC,AL,CR:CR;0)  298.15   +6*U1ALCR;          6000 N !
  PARAMETER G(L12_FCC,AL,CR:AL;1)  298.15   +3*L1ALCR;          6000 N !
  PARAMETER G(L12_FCC,AL,CR:CR;1)  298.15   +3*L1ALCR;          6000 N !
$
  PARAMETER G(L12_FCC,CR,NI:CR;0)  298.15   +6*U1CRNI;          6000 N !
  PARAMETER G(L12_FCC,CR,NI:NI;0)  298.15   +6*U1CRNI;          6000 N !
  PARAMETER G(L12_FCC,CR,NI:CR;1)  298.15   +3*L1CRNI;          6000 N !
  PARAMETER G(L12_FCC,CR,NI:NI;1)  298.15   +3*L1CRNI;          6000 N !
  

  PARAMETER G(L12_FCC,AL,CR:NI;0)  298.15   +6*U1ALCR;          6000 N !
  PARAMETER G(L12_FCC,AL,NI:CR;0)  298.15   +6*U1ALNI+9975;     6000 N !
  PARAMETER G(L12_FCC,CR,NI:AL;0)  298.15   +6*U1CRNI+9975;     6000 N !
  
  PARAMETER G(L12_FCC,AL,CR:NI;1)  298.15   0;                  6000 N !
  PARAMETER G(L12_FCC,AL,NI:CR;1)  298.15   -9975+3*L1ALNI;     6000 N !
  PARAMETER G(L12_FCC,CR,NI:AL;1)  298.15   -9975;              6000 N !
  
  PARAMETER G(L12_FCC,AL,CR,NI:AL;0)  298.15   -9975;           6000 N !
  PARAMETER G(L12_FCC,AL,CR,NI:CR;0)  298.15   -9975;           6000 N !
  PARAMETER G(L12_FCC,AL,CR,NI:NI;0)  298.15   +39900;          6000 N !



"""

DBF = Database(TDB_TEST_STRING)

def test_surface():
    "Bare minimum: calculation produces a result."
    energy_surf(DBF, ['AL', 'CR', 'NI'], ['L12_FCC'],
                T=1273, pdens=10, mode='numpy')

@nose.tools.raises(AttributeError)
def test_unknown_model_attribute():
    "Sampling an unknown model attribute raises exception."
    energy_surf(DBF, ['AL', 'CR', 'NI'], ['L12_FCC'],
                T=1400.0, output='_fail_')
