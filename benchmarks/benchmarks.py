""" Benchmarks for use with the airspeed velocity (asv) package. 
(http://asv.readthedocs.io/en/latest/)

The calculate and equilibrium benchmarks were chosen from the examples for 
simplicity. They may not be completely representative, but the purpose of the
benchmarks is to show how the performance changes over time. Testing the limits
of the code is for the tests.
"""

from pycalphad import Database, Model, calculate, equilibrium, v
from pycalphad.core.halton import halton
import timeit

__author__ = "Brandon Bocklund"

class BenchmarkSetups:
    """ Benchmarks the creation of databases and models.
    
    Note that multiple databases, files, etc. could be tested
    using the 'params' attribute. 
    """

    def setup(self):
        # create a file to read from
        with open('alfe_sei.TDB', 'w') as f:
            f.write(ALFE_TDB)
        # setup TDBs
        self.tdb_file =  open('alfe_sei.TDB', 'r')
        self.tdb_string = ALFE_TDB
        # setup databases
        self.db = Database(self.tdb_string)

    def teardown(self):
        self.tdb_file.close()

    def time_database_string(self):
        Database(self.tdb_string)
    
    def time_database_file(self):
        Database(self.tdb_file)
    
    def time_model_magnetic(self):
        Model(self.db, ['AL', 'FE', 'VA'], 'B2_BCC')

    def time_model_non_magnetic(self):
        Model(self.db, ['AL', 'FE', 'VA'], 'LIQUID')

    def time_calculate_magnetic(self):
        calculate(self.db, ['AL', 'FE', 'VA'], 'B2_BCC', T=(300, 2000, 10))

    def time_calculate_non_magnetic(self):
        calculate(self.db, ['AL', 'FE', 'VA'], 'LIQUID', T=(300, 2000, 10))

class BenchmarkEquilibrium:
    """ Benchmarks the running of equilibrium calculations. """
    # timer is for multiprocessing, but it may need to be more granular.
    # default_timer is walltime, where normally process time is used.
    timer = timeit.default_timer
    number = 5 # set run times manually, because they are longer running

    def setup(self):

        # setup databases
        self.db_alfe = Database(ALFE_TDB)
        self.db_alni = Database(ALNI_TDB)

    def time_equilibrium_al_fe(self):
        equilibrium(self.db_alni, ['AL', 'NI', 'VA'], ['LIQUID', 'FCC_L12'], {v.X('AL'): 0.10, v.T: (300, 2500, 20), v.P: 101325})

    def time_equilibrium_al_ni(self):
        equilibrium(self.db_alfe , ['AL', 'FE', 'VA'], ['LIQUID', 'B2_BCC'], {v.X('AL'): 0.25, v.T: (300, 2000, 50), v.P: 101325})

def time_halton(dim, pts):
    halton(dim, pts)

time_halton.params = ([1, 2, 5, 10, 50, 100, 300], [1000, 5000, 10000, 50000, 100000])
time_halton.param_names = ['dimensions', 'number of points']

ALFE_TDB = """
$ ALFE
$
$ -----------------------------------------------------------------------------
$ 2006.12.21
$ 2007.02.20 mod  ( 386.15 --> 368.15 )
$
$ TDB file created by K.Hashimoto and T.Abe,
$
$ Particle Simulation and Thermodynamics Group, National Institute for
$ Materials Science. 1-2-1 Sengen, Tsukuba, Ibaraki 305-0047, Japan
$
$ e-mail: abe.taichi@nims.go.jp
$
$ Copyright (C) NIMS 2007
$ -----------------------------------------------------------------------------
$
$ The parameter set is taken from
$ COST 507, Thermochemical database for light metal alloys, vol.2
$ Ed. I.Ansara, A.T.Dinsdale, M.H.Rand, (1998)
$ ISBN: 92-828-3902-8
$
$ -----------------------------------------------------------------------------
$
 ELEMENT /-   ELECTRON_GAS              0.0000E+00  0.0000E+00  0.0000E+00!
 ELEMENT VA   VACUUM                    0.0000E+00  0.0000E+00  0.0000E+00!
 ELEMENT AL   FCC_A1                    26.981539   4577.296    28.3215!
 ELEMENT FE   BCC_A2                    55.847      4489        27.28  !

$
$--------1---------2---------3---------4---------5---------6---------7---------8
$
 FUNCTION UN_ASS     298.15   0;                                 300.00 N !
 FUNCTION GHSERAL    298.15  -7976.15+137.093038*T-24.3671976*T*LN(T)
   -.001884662*T**2-8.77664E-07*T**3+74092*T**(-1);              700  Y
   -11276.24+223.048446*T-38.5844296*T*LN(T)+.018531982*T**2
   -5.764227E-06*T**3+74092*T**(-1);                             933.47 Y
   -11278.378+188.684153*T-31.748192*T*LN(T)-1.230524E+28*T**(-9); 2900 N !

 FUNCTION GHSERFE    298.15  +1225.7+124.134*T-23.5143*T*LN(T)
     -.00439752*T**2-5.8927E-08*T**3+77359*T**(-1);              1811  Y
      -25383.581+299.31255*T-46*T*LN(T)+2.29603E+31*T**(-9);     6000  N !
$
 FUNCTION GALBCC     298.15  +10083-4.813*T+GHSERAL#;            6000  N !
 FUNCTION GBCCAL     298.15  +10083-4.813*T+GHSERAL#;            6000  N !
 FUNCTION GALLIQ     298.15  +11005.029-11.841867*T
                  +7.934E-20*T**7+GHSERAL#;                      933.47  Y
       +10482.382-11.253974*T+1.231E+28*T**(-9)+GHSERAL#;        2900 N !
 FUNCTION GFEFCC     298.15  -1462.4+8.282*T-1.15*T*LN(T)
     +6.4E-04*T**2+GHSERFE#;                                     1811  Y
      -1713.815+0.94001*T+0.49251E+31*T**(-9)+GHSERFE#;          6000  N !
 FUNCTION GFELIQ     298.15  +12040.17-6.55843*T
     -3.67516E-21*T**7+GHSERFE#;                                 1811  Y
     +14544.751-8.01055*T-2.29603E+31*T**(-9)+GHSERFE#;          6000  N !
$
 TYPE_DEFINITION % SEQ *!
 DEFINE_SYSTEM_DEFAULT ELEMENT 2 !
 DEFAULT_COMMAND DEF_SYS_ELEMENT VA !
$
PHASE LIQUID %  1  1.0  !
   CONSTITUENT LIQUID :AL,FE: !
   PARAMETER G(LIQUID,AL;0)  298.15  +GALLIQ#;            6000 N !
   PARAMETER G(LIQUID,FE;0)  298.15  +GFELIQ#;            6000 N !
   PARAMETER G(LIQUID,AL,FE;0) 298.15 -91976.5+22.1314*T; 6000 N !
   PARAMETER G(LIQUID,AL,FE;1) 298.15 -5672.58+4.8728*T;  6000 N !
   PARAMETER G(LIQUID,AL,FE;2) 298.15 +121.9;             6000 N !


TYPE_DEFINITION & GES A_P_D B2_BCC DIS_PART BCC_A2 ,,,!
TYPE_DEFINITION - GES A_P_D BCC_A2 MAGNETIC  -1.0    0.4 !
PHASE BCC_A2  %&-  2 1   3 !
   CONSTITUENT BCC_A2  :AL,FE: VA :  !
   PARAMETER G(BCC_A2,AL:VA;0)      298.15  +GALBCC#;      2900 N !
   PARAMETER G(BCC_A2,FE:VA;0)      298.15  +GHSERFE#;     6000 N !
   PARAMETER TC(BCC_A2,FE:VA;0)     298.15  1043;          6000 N !
   PARAMETER BMAGN(BCC_A2,FE:VA;0)  298.15  2.22;          6000 N !
$  PARAMETER G(BCC_A2,AL,FE:VA;0)   298.15  +4.0*(-30740+7.9972*T);
   PARAMETER G(BCC_A2,AL,FE:VA;0)   298.15  -122960.+31.9888*T; 6000 N !
$  PARAMETER G(BCC_A2,AL,FE:VA;1)   298.15  +8.0*(368.15);
   PARAMETER G(BCC_A2,AL,FE:VA;1)   298.15  +2945.2;       6000 N !
   PARAMETER TC(BCC_A2,AL,FE:VA;1)  298.15   +504;         6000 N !

$ separate the order and disordered phases
$ -2*860*8.31451
$
PHASE B2_BCC %&  3 0.5  0.5  3  !
   CONSTITUENT B2_BCC  :AL,FE:AL,FE:VA:  !
   PARAMETER G(B2_BCC,AL:AL:VA;0)  298.15  0.0;         6000 N !
   PARAMETER G(B2_BCC,FE:AL:VA;0)  298.15  -14300.9572; 6000 N !
   PARAMETER G(B2_BCC,AL:FE:VA;0)  298.15  -14300.9572; 6000 N !
   PARAMETER G(B2_BCC,FE:FE:VA;0)  298.15  0.0;         6000 N !


$ includes the parameters of order + disordered in a single description
$TYPE_DEFINITION + GES A_P_D BCC_B2 MAGNETIC  -1.0    0.4 !
$PHASE BCC_B2 %+  3 0.5  0.5  3  !
$   CONSTITUENT BCC_B2  :AL,FE:AL,FE:VA:  !
$   PARAMETER G(BCC_B2,AL:AL:VA;0)  298.15  +GBCCAL#;       6000 N !
$   PARAMETER G(BCC_B2,FE:AL:VA;0)  298.15
$             -37890.478+7.9972*T+0.5*GALBCC#+0.5*GHSERFE#; 6000 N !
$   PARAMETER G(BCC_B2,AL:FE:VA;0)  298.15
$             -37890.478+7.9972*T+0.5*GALBCC#+0.5*GHSERFE#; 6000 N !
$   PARAMETER G(BCC_B2,FE:FE:VA;0)  298.15  +GHSERFE#;      6000 N !
$   PARAMETER TC(BCC_B2,FE:AL:VA;0)     298.15  521.5;      6000 N !
$   PARAMETER BMAGN(BCC_B2,FE:AL:VA;0)  298.15  1.11;       6000 N !
$   PARAMETER TC(BCC_B2,AL:FE:VA;0)     298.15  521.5;      6000 N !
$   PARAMETER BMAGN(BCC_B2,AL:FE:VA;0)  298.15  1.11;       6000 N !
$   PARAMETER TC(BCC_B2,FE:FE:VA;0)     298.15  1043.0;     6000 N !
$   PARAMETER BMAGN(BCC_B2,FE:FE:VA;0)  298.15  2.22;       6000 N !
$   PARAMETER G(BCC_B2,AL,FE:AL:VA;0)
$                               298.15 -22485.072+7.9772*T; 6000 N !
$   PARAMETER G(BCC_B2,AL:AL,FE:VA;0)
$                               298.15 -22485.072+7.9772*T; 6000 N !
$   PARAMETER G(BCC_B2,AL,FE:AL:VA;1)     298.15 +368.15;   6000 N !
$   PARAMETER G(BCC_B2,AL:AL,FE:VA;1)     298.15 +368.15;   6000 N !
$   PARAMETER TC(BCC_B2,AL,FE:AL:VA;0)    298.15  189.0;    6000 N !
$   PARAMETER TC(BCC_B2,AL:AL,FE:VA;0)    298.15  189.0;    6000 N !
$   PARAMETER TC(BCC_B2,AL,FE:AL:VA;1)    298.15   63.0;    6000 N !
$   PARAMETER TC(BCC_B2,AL:AL,FE:VA;1)    298.15   63.0;    6000 N !
$   PARAMETER BMAGN(BCC_B2,AL,FE:AL:VA;0) 298.15    0.0;    6000 N !
$   PARAMETER BMAGN(BCC_B2,AL:AL,FE:VA;0) 298.15    0.0;    6000 N !
$   PARAMETER TC(BCC_B2,AL,FE:FE:VA;0)    289.15 -189.0;    6000 N !
$   PARAMETER TC(BCC_B2,FE:AL,FE:VA;0)    298.15 -189.0;    6000 N !
$   PARAMETER TC(BCC_B2,AL,FE:FE:VA;1)    298.15   63.0;    6000 N !
$   PARAMETER TC(BCC_B2,FE:AL,FE:VA;1)    298.15   63.0;    6000 N !
$   PARAMETER BMAGN(BCC_B2,AL,FE:FE:VA;0) 298.15    0.0;    6000 N !
$   PARAMETER BMAGN(BCC_B2,FE:AL,FE:VA;0) 298.15    0.0;    6000 N !
$   PARAMETER G(BCC_B2,AL,FE:FE:VA;0)
$                               298.15 -24693.972+7.9772*T; 6000 N !
$   PARAMETER G(BCC_B2,FE:AL,FE:VA;0)
$                               298.15 -24693.972+7.9772*T; 6000 N !
$   PARAMETER G(BCC_B2,AL,FE:FE:VA;1)     298.15 +368.15;   6000 N !
$   PARAMETER G(BCC_B2,FE:AL,FE:VA;1)     298.15 +368.15;   6000 N !

TYPE_DEFINITION / GES A_P_D FCC_A1 MAGNETIC  -3.0    2.80000E-01 !
PHASE FCC_A1  %/  2 1   1 !
   CONSTITUENT FCC_A1  :AL,FE : VA :  !
   PARAMETER G(FCC_A1,AL:VA;0)      298.15   +GHSERAL#;       2900 N !
   PARAMETER G(FCC_A1,FE:VA;0)      298.15   +GFEFCC#;        6000 N !
   PARAMETER TC(FCC_A1,FE:VA;0)     298.15   +67;             6000 N !
   PARAMETER BMAGN(FCC_A1,FE:VA;0)  298.15   +0.7;            6000 N !
   PARAMETER G(FCC_A1,AL,FE:VA;0)   298.15  -76066.1+18.6758*T; 6000 N !
   PARAMETER G(FCC_A1,AL,FE:VA;1)   298.15  +21167.4+1.3398*T;  6000 N !

TYPE_DEFINITION ) GES A_P_D HCP_A3 MAGNETIC  -3.0    2.80000E-01 !
PHASE HCP_A3 %) 2 1 .5 !
   CONSTITUENT HCP_A3  :AL,FE: VA :  !
   PARAMETER G(HCP_A3,AL:VA;0)  298.15  +5481-1.8*T+GHSERAL#;  2900 N !
   PARAMETER G(HCP_A3,FE:VA;0)  298.15
                  -3705.78+12.591*T-1.15*T*LN(T)
                  +6.4E-04*T**2+GHSERFE#;  1.81100E+03  Y
                  -3957.199+5.24951*T+4.9251E+30*T**(-9)+GHSERFE#;  6000 N !
   PARAMETER G(HCP_A3,AL,FE:VA;0)  298.15  -106903.0+20.0*T;        6000 N !

PHASE AL2FE % 2  2  1   !
   CONSTITUENT AL2FE  :AL:FE:  !
   PARAMETER G(AL2FE,AL:FE;0)  298.15
                              -98097.0+18.7503*T+2*GHSERAL#+GHSERFE#; 6000 N !

PHASE AL13FE4 %  3  0.6275  0.235  0.1375  !
   CONSTITUENT AL13FE4  :AL:FE:AL,VA:  !
   PARAMETER G(AL13FE4,AL:FE:AL;0)  298.15
                   -30714.4+7.44*T+0.765*GHSERAL#+0.235*GHSERFE#;     6000 N !
   PARAMETER G(AL13FE4,AL:FE:VA;0)  298.15
                   -27781.3+7.2566*T+0.6275*GHSERAL#+0.235*GHSERFE#;  6000 N !

PHASE AL5FE2 % 2  5  2  !
   CONSTITUENT AL5FE2  :AL:FE:  !
   PARAMETER G(AL5FE2,AL:FE;0)  298.15
                            -228576+48.99503*T+5*GHSERAL#+2*GHSERFE#; 6000 N !

PHASE AL5FE4 % 1 1.0 !
   CONSTITUENT AL5FE4  :AL,FE :  !
   PARAMETER G(AL5FE4,AL;0)     298.15  +12178.90-4.813*T+GHSERAL#;   6000 N !
   PARAMETER G(AL5FE4,FE;0)     298.15  +5009.03+GHSERFE#;            6000 N !
   PARAMETER G(AL5FE4,AL,FE;0)  298.15  -131649+29.4833*T;            6000 N !
   PARAMETER G(AL5FE4,AL,FE;1)  298.15  -18619.5;                     6000 N !

$ALFE-NIMS"""

ALNI_TDB = """
$ Database file written 2014-10-21
$ From database: USER                    
 ELEMENT /-   ELECTRON_GAS              0.0000E+00  0.0000E+00  0.0000E+00!
 ELEMENT VA   VACUUM                    0.0000E+00  0.0000E+00  0.0000E+00!
 ELEMENT AL   FCC_A1                    2.6982E+01  4.5773E+03  2.8322E+01!
 ELEMENT NI   FCC_A1                    5.8690E+01  4.7870E+03  2.9796E+01!
 
 FUNCTION F154T      2.98150E+02  +323947.58-25.1480943*T-20.859*T*LN(T)
     +4.5665E-05*T**2-3.942E-09*T**3-24275.5*T**(-1);  4.30000E+03  Y
      +342017.233-54.0526109*T-17.7891*T*LN(T)+6.822E-05*T**2
     -1.91111667E-08*T**3-14782200*T**(-1);  8.20000E+03  Y
      +542396.07-411.214335*T+22.2419*T*LN(T)-.00349619*T**2+4.0491E-08*T**3
     -2.0366965E+08*T**(-1);  1.00000E+04  N !
 FUNCTION F625T      2.98150E+02  +496408.232+35.479739*T-41.6397*T*LN(T)
     +.00249636*T**2-4.90507333E-07*T**3+85390.3*T**(-1);  9.00000E+02  Y
      +497613.221+17.368131*T-38.85476*T*LN(T)-2.249805E-04*T**2
     -9.49003167E-09*T**3-5287.23*T**(-1);  2.80000E+03  N !
 FUNCTION GHSERAL    2.98150E+02  -7976.15+137.093038*T-24.3671976*T*LN(T)
     -.001884662*T**2-8.77664E-07*T**3+74092*T**(-1);  7.00000E+02  Y
      -11276.24+223.048446*T-38.5844296*T*LN(T)+.018531982*T**2
     -5.764227E-06*T**3+74092*T**(-1);  9.33600E+02  Y
      -11278.378+188.684153*T-31.748192*T*LN(T)-1.231E+28*T**(-9);  
     6.00000E+03  N !
 FUNCTION GBCCAL     2.98150E+02  +10083-4.813*T+GHSERAL#;   6.00000E+03   N 
     !
 FUNCTION LB2ALVA    2.98150E+02  150000;   6.00000E+03   N !
 FUNCTION B2ALVA     2.98150E+02  +10000-T;   6.00000E+03   N !
 FUNCTION F13191T    2.98150E+02  +417658.868-44.7777921*T-20.056*T*LN(T)
     -.0060415*T**2+1.24774E-06*T**3-16320*T**(-1);  8.00000E+02  Y
      +413885.448+9.41787679*T-28.332*T*LN(T)+.00173115*T**2-8.399E-08*T**3
     +289050*T**(-1);  3.90000E+03  Y
      +440866.732-62.5810038*T-19.819*T*LN(T)+5.067E-04*T**2
     -4.93233333E-08*T**3-15879735*T**(-1);  7.60000E+03  Y
      +848806.287-813.398164*T+64.69*T*LN(T)-.00731865*T**2
     +8.71833333E-08*T**3-3.875846E+08*T**(-1);  1.00000E+04  N !
 FUNCTION F13265T    2.98150E+02  +638073.279-68.1901928*T-24.897*T*LN(T)
     -.0313584*T**2+5.93355333E-06*T**3-14215*T**(-1);  8.00000E+02  Y
      +611401.772+268.084821*T-75.25401*T*LN(T)+.01088525*T**2
     -7.08741667E-07*T**3+2633835*T**(-1);  2.10000E+03  Y
      +637459.339+72.0712678*T-48.587*T*LN(T)-9.09E-05*T**2
     +9.12933333E-08*T**3-1191755*T**(-1);  4.50000E+03  Y
      +564540.781+329.599011*T-80.11301*T*LN(T)+.00578085*T**2
     -1.08841667E-07*T**3+29137900*T**(-1);  6.00000E+03  N !
 FUNCTION GHSERNI    2.98140E+02  -5179.159+117.854*T-22.096*T*LN(T)
     -.0048407*T**2;  1.72800E+03  Y
      -27840.655+279.135*T-43.1*T*LN(T)+1.12754E+31*T**(-9);  6.00000E+03  N 
     !
 FUNCTION GBCCNI     2.98150E+02  +8715.084-3.556*T+GHSERNI#;   6.00000E+03  
      N !
 FUNCTION LB2NIVA    2.98150E+02  -64024.38+26.49419*T;   6.00000E+03   N !
 FUNCTION B2NIVA     2.98150E+02  +162397.3-27.40575*T;   6.00000E+03   N !
 FUNCTION LB2ALNI    2.98150E+02  -52440.88+11.30117*T;   6.00000E+03   N !
 FUNCTION B2ALNI     2.98150E+02  -152397.3+26.40575*T;   6.00000E+03   N !
 FUNCTION ALNI3      2.98150E+02  +3*U1ALNI#;   6.00000E+03   N !
 FUNCTION AL3NI      2.98150E+02  +3*U1ALNI#;   6.00000E+03   N !
 FUNCTION AL2NI2     2.98150E+02  +4*U1ALNI#;   6.00000E+03   N !
 FUNCTION L04ALNI    2.98150E+02  +U3ALNI#;   6.00000E+03   N !
 FUNCTION L14ALNI    2.98150E+02  +U4ALNI#;   6.00000E+03   N !
 FUNCTION U3ALNI     298.15 0.0; 6000.00  N !
 FUNCTION U4ALNI     2.98150E+02  +7203.60609-3.7427303*T;   6.00000E+03   N 
     !
 FUNCTION U1ALNI     2.98150E+02  +2*UNTIER#*UALNI#;   6.00000E+03   N !
 FUNCTION UNTIER     2.98150E+02  +TROIS#**(-1);   6.00000E+03   N !
 FUNCTION UALNI      2.98150E+02  -22212.8931+4.39570389*T;   6.00000E+03   
     N !
 FUNCTION TROIS      2.98150E+02  3;   6.00000E+03   N !
 FUNCTION UN_ASS 298.15 0; 300 N !
 
 TYPE_DEFINITION % SEQ *!
 DEFINE_SYSTEM_DEFAULT ELEMENT 2 !
 DEFAULT_COMMAND DEF_SYS_ELEMENT VA /- !


 PHASE LIQUID %  1  1.0  !
 CONSTITUENT LIQUID :AL,NI :  !

   PARAMETER G(LIQUID,AL;0)  2.98130E+02  +11005.029-11.841867*T
  +7.934E-20*T**7+GHSERAL#;  9.33600E+02  Y
   +10482.382-11.253974*T+1.231E+28*T**(-9)+GHSERAL#;  6.00000E+03  N REF3 !
   PARAMETER G(LIQUID,NI;0)  2.98130E+02  +16414.686-9.397*T
  -3.82318E-21*T**7+GHSERNI#;  1.72800E+03  Y
   +18290.88-10.537*T-1.12754E+31*T**(-9)+GHSERNI#;  6.00000E+03  N REF3 !
   PARAMETER G(LIQUID,AL,NI;0)  2.98150E+02  -207109.28+41.31501*T;   
  6.00000E+03   N REF7 !
   PARAMETER G(LIQUID,AL,NI;1)  2.98150E+02  -10185.79+5.8714*T;   
  6.00000E+03   N REF7 !
   PARAMETER G(LIQUID,AL,NI;2)  2.98150E+02  +81204.81-31.95713*T;   
  6.00000E+03   N REF7 !
   PARAMETER G(LIQUID,AL,NI;3)  2.98150E+02  +4365.35-2.51632*T;   
  6.00000E+03   N REF7 !
   PARAMETER G(LIQUID,AL,NI;4)  2.98150E+02  -22101.64+13.16341*T;   
  6.00000E+03   N REF7 !


 PHASE AL3NI1  %  2 .75   .25 !
    CONSTITUENT AL3NI1  :AL : NI :  !

   PARAMETER G(AL3NI1,AL:NI;0)  2.98150E+02  -48483.73+12.29913*T
  +.75*GHSERAL#+.25*GHSERNI#;   6.00000E+03   N REF7 !


 PHASE AL3NI2  %  3 3   2   1 !
    CONSTITUENT AL3NI2  :AL : AL,NI : NI,VA :  !

   PARAMETER G(AL3NI2,AL:AL:NI;0)  2.98150E+02  +5*GBCCAL#+GBCCNI#-39465.978
  +7.89525*T;   6.00000E+03   N REF7 !
   PARAMETER G(AL3NI2,AL:NI:NI;0)  2.98150E+02  +3*GBCCAL#+3*GBCCNI#
  -427191.9+79.21725*T;   6.00000E+03   N REF7 !
   PARAMETER G(AL3NI2,AL:AL:VA;0)  2.98150E+02  +5*GBCCAL#+30000-3*T;   
  6.00000E+03   N REF7 !
   PARAMETER G(AL3NI2,AL:NI:VA;0)  2.98150E+02  +3*GBCCAL#+2*GBCCNI#
  -357725.92+68.322*T;   6.00000E+03   N REF7 !
   PARAMETER G(AL3NI2,AL:AL,NI:*;0)  2.98150E+02  -193484.18+131.79*T;   
  6.00000E+03   N REF7 !
$   PARAMETER G(AL3NI2,AL:AL,NI:NI;0)  2.98150E+02  -193484.18+131.79*T;   
$  6.00000E+03   N REF7 !
$   PARAMETER G(AL3NI2,AL:AL,NI:VA;0)  2.98150E+02  -193484.18+131.79*T;   
$  6.00000E+03   N REF7 !
   PARAMETER G(AL3NI2,AL:*:NI,VA;0)  2.98150E+02  -22001.7+7.0332*T;   
  6.00000E+03   N REF7 !
$   PARAMETER G(AL3NI2,AL:AL:NI,VA;0)  2.98150E+02  -22001.7+7.0332*T;   
$  6.00000E+03   N REF7 !
$   PARAMETER G(AL3NI2,AL:NI:NI,VA;0)  2.98150E+02  -22001.7+7.0332*T;   
$  6.00000E+03   N REF7 !


 PHASE AL3NI5  %  2 .375   .625 !
    CONSTITUENT AL3NI5  :AL : NI :  !

   PARAMETER G(AL3NI5,AL:NI;0)  2.98150E+02  +.375*GHSERAL#+.625*GHSERNI#
  -55507.7594+7.2648103*T;   6.00000E+03   N REF7 !


 TYPE_DEFINITION & GES A_P_D BCC_A2 MAGNETIC  -1.0    4.00000E-01 !
 PHASE BCC_A2  %&  2 1   3 !
    CONSTITUENT BCC_A2  :AL,NI,VA : VA :  !

   PARAMETER G(BCC_A2,AL:VA;0)  2.98150E+02  +GBCCAL#;   6.00000E+03   N 
  REF3 !
   PARAMETER G(BCC_A2,NI:VA;0)  2.98150E+02  +GBCCNI#;   6.00000E+03   N 
  REF3 !
   PARAMETER TC(BCC_A2,NI:VA;0)  2.98150E+02  575;   6.00000E+03   N REF2 !
   PARAMETER BMAGN(BCC_A2,NI:VA;0)  2.98150E+02  .85;   6.00000E+03   N REF2 !
   PARAMETER G(BCC_A2,VA:VA;0) 298.15 0; 6000 N!
   PARAMETER G(BCC_A2,AL,VA:VA;0)  2.98150E+02  +B2ALVA#+LB2ALVA#;   
  6.00000E+03   N REF8 !
   PARAMETER G(BCC_A2,AL,NI:VA;0)  2.98150E+02  +B2ALNI#+LB2ALNI#;   
  6.00000E+03   N REF8 !
   PARAMETER G(BCC_A2,NI,VA:VA;0)  2.98150E+02  +B2NIVA#+LB2NIVA#;   
  6.00000E+03   N REF8 !


$ THIS PHASE HAS A DISORDERED CONTRIBUTION FROM BCC_A2                  
 TYPE_DEFINITION ' GES A_P_D BCC_B2 DIS_PART BCC_A2 ,,,!
 TYPE_DEFINITION ( GES A_P_D BCC_B2 MAGNETIC  -1.0    4.00000E-01 !
 PHASE BCC_B2  %('  3 .5   .5   3 !
    CONSTITUENT BCC_B2  :AL,NI,VA : AL,NI,VA : VA :  !

   PARAMETER G(BCC_B2,AL:AL:VA;0) 298.15 0; 6000 N!
   PARAMETER G(BCC_B2,NI:AL:VA;0)  2.98150E+02  +.5*B2ALNI#-.5*LB2ALNI#;   
  6.00000E+03   N REF8 !
   PARAMETER G(BCC_B2,VA:AL:VA;0)  2.98150E+02  +.5*B2ALVA#-.5*LB2ALVA#;   
  6.00000E+03   N REF8 !
   PARAMETER G(BCC_B2,AL:NI:VA;0)  2.98150E+02  +.5*B2ALNI#-.5*LB2ALNI#;   
  6.00000E+03   N REF8 !
   PARAMETER G(BCC_B2,NI:NI:VA;0) 298.15 0; 6000 N!
   PARAMETER G(BCC_B2,VA:NI:VA;0)  2.98150E+02  +.5*B2NIVA#-.5*LB2NIVA#;   
  6.00000E+03   N REF8 !
   PARAMETER G(BCC_B2,AL:VA:VA;0)  2.98150E+02  +.5*B2ALVA#-.5*LB2ALVA#;   
  6.00000E+03   N REF8 !
   PARAMETER G(BCC_B2,NI:VA:VA;0)  2.98150E+02  +.5*B2NIVA#-.5*LB2NIVA#;   
  6.00000E+03   N REF8 !
   PARAMETER G(BCC_B2,VA:VA:VA;0) 298.15 0; 6000 N!


 TYPE_DEFINITION ) GES A_P_D FCC_A1 MAGNETIC  -3.0    2.80000E-01 !
 PHASE FCC_A1  %)  2 1   1 !
    CONSTITUENT FCC_A1  :AL,NI : VA :  !

   PARAMETER G(FCC_A1,AL:VA;0)  2.98150E+02  +GHSERAL#;   6.00000E+03   N 
  REF3 !
   PARAMETER G(FCC_A1,NI:VA;0)  2.98150E+02  +GHSERNI#;   6.00000E+03   N 
  REF3 !
   PARAMETER TC(FCC_A1,NI:VA;0)  2.98150E+02  633;   6.00000E+03   N REF2 !
   PARAMETER BMAGN(FCC_A1,NI:VA;0)  2.98150E+02  .52;   6.00000E+03   N REF2 !
   PARAMETER TC(FCC_A1,AL,NI:VA;0)  2.98150E+02  -1112;   6.00000E+03   N 
  REF7 !
   PARAMETER TC(FCC_A1,AL,NI:VA;1)  2.98150E+02  1745;   6.00000E+03   N 
  REF7 !
   PARAMETER G(FCC_A1,AL,NI:VA;0)  2.98150E+02  -162407.75+16.212965*T;   
  6.00000E+03   N REF7 !
   PARAMETER G(FCC_A1,AL,NI:VA;1)  2.98150E+02  +73417.798-34.914168*T;   
  6.00000E+03   N REF7 !
   PARAMETER G(FCC_A1,AL,NI:VA;2)  2.98150E+02  +33471.014-9.8373558*T;   
  6.00000E+03   N REF7 !
   PARAMETER G(FCC_A1,AL,NI:VA;3)  2.98150E+02  -30758.01+10.25267*T;   
  6.00000E+03   N REF7 !


$ THIS PHASE HAS A DISORDERED CONTRIBUTION FROM FCC_A1                  
 TYPE_DEFINITION * GES A_P_D FCC_L12 DIS_PART FCC_A1 ,,,!
 TYPE_DEFINITION + GES A_P_D FCC_L12 MAGNETIC  -3.0    2.80000E-01 !
 PHASE FCC_L12  %*+  3 .75   .25   1 !
    CONSTITUENT FCC_L12  :AL,NI : AL,NI : VA :  !

   PARAMETER G(FCC_L12,AL:AL:VA;0) 298.15 0; 6000 N!
   PARAMETER G(FCC_L12,NI:AL:VA;0)  2.98150E+02  +ALNI3#;   6.00000E+03   N 
  REF9 !
   PARAMETER G(FCC_L12,AL:NI:VA;0)  2.98150E+02  +AL3NI#;   6.00000E+03   N 
  REF9 !
   PARAMETER G(FCC_L12,NI:NI:VA;0) 298.15 0; 6000 N!
   PARAMETER G(FCC_L12,AL,NI:AL:VA;0)  2.98150E+02  -1.5*ALNI3#+1.5*AL2NI2#
  +1.5*AL3NI#;   6.00000E+03   N REF9 !
   PARAMETER G(FCC_L12,AL,NI:AL:VA;1)  2.98150E+02  +.5*ALNI3#-1.5*AL2NI2#
  +1.5*AL3NI#;   6.00000E+03   N REF9 !
   PARAMETER G(FCC_L12,AL,NI:NI:VA;0)  2.98150E+02  +1.5*ALNI3#+1.5*AL2NI2#
  -1.5*AL3NI#;   6.00000E+03   N REF9 !
   PARAMETER G(FCC_L12,AL,NI:NI:VA;1)  2.98150E+02  -1.5*ALNI3#+1.5*AL2NI2#
  -.5*AL3NI#;   6.00000E+03   N REF9 !
   PARAMETER G(FCC_L12,*:AL,NI:VA;0)  2.98150E+02  +L04ALNI#;   6.00000E+03  
   N REF9 !
   PARAMETER G(FCC_L12,*:AL,NI:VA;1)  2.98150E+02  +L14ALNI#;   6.00000E+03  
   N REF9 !
   PARAMETER G(FCC_L12,AL,NI:*:VA;0)  2.98150E+02  +3*L04ALNI#;   
  6.00000E+03   N REF9 !
   PARAMETER G(FCC_L12,AL,NI:*:VA;1)  2.98150E+02  +3*L14ALNI#;   
  6.00000E+03   N REF9 !

 LIST_OF_REFERENCES
 NUMBER  SOURCE
   REF10  'AL1<G> CODATA KEY VALUES SGTE ** ALUMINIUM <GAS> Cp values 
         similar in Codata Key Values and IVTAN Vol. 3'
   REF11  'AL2<G> CHATILLON(1992) Enthalpy of formation for Al1<g> taken 
         from Codata Key Values. Enthalpy of form. from TPIS dissociation 
         energy mean Value corrected with new fef from Sunil K.K. and Jordan 
         K.D. (J.Phys. Chem. 92(1988)2774) ab initio calculations.'
   REF3   'Alan Dinsdale, SGTE Data for Pure Elements, NPL Report DMA(A)195 
         Rev. August 1990'
   REF8   'N. Dupin, I. Ansara, Z. metallkd., Vol 90 (1999) p 76-85; Al-Ni'
   REF14  'NI1<G> T.C.R.A.S Class: 1 Data provided by T.C.R.A.S. October 1996'
   REF15  'NI2<G> T.C.R.A.S Class: 5 Data provided by T.C.R.A.S. October 1996'
   REF2   'Alan Dinsdale, SGTE Data for Pure Elements, NPL Report DMA(A)195 
         September 1989'
   REF7   'N. Dupin, Thesis, LTPCM, France, 1995; Al-Ni, also in I. Ansara, 
         N. Dupin, H.L. Lukas, B. SUndman J. Alloys Compds, 247 (1-2), 20-30 
         (1997)'
   REF9   ' N. Dupin, I. Ansara, B. Sundman Thermodynamic Re-Assessment of 
         the Ternary System Al-Cr-Ni, Calphad, 25 (2), 279-298 (2001); Al-Cr
         -Ni'
  ! """
