"""The tdb module provides support for reading and writing databases in
Thermo-Calc TDB format.
"""

from pyparsing import CaselessKeyword, CharsNotIn, Group, Empty
from pyparsing import LineEnd, OneOrMore, Regex, SkipTo
from pyparsing import Suppress, White, Word, alphanums, alphas, nums
from pyparsing import delimitedList, ParseException
import re
from sympy import sympify, And, Piecewise
import pycalphad.variables as v

def _make_piecewise_ast(toks):
    """
    Convenience function for converting tokens into a piecewise sympy AST.
    """
    cur_tok = 0
    expr_cond_pairs = []
    variable_fixes = {
        'T': v.T,
        'P': v.P
    }
    # sympify doesn't recognize LN as ln()
    while cur_tok < len(toks)-1:
        low_temp = toks[cur_tok]
        high_temp = toks[cur_tok+2]
        expr_string = toks[cur_tok+1].replace('#', '')
        expr_string = \
            re.sub(r'(?<!\w)LN(?!\w)', 'ln', expr_string, flags=re.IGNORECASE)
        expr_string = \
            re.sub(r'(?<!\w)EXP(?!\w)', 'exp', expr_string,
                   flags=re.IGNORECASE)
        # TODO: sympify uses eval. Don't use it on unsanitized input.
        expr_cond_pairs.append(
            (
                sympify(expr_string).subs(variable_fixes),
                And(low_temp <= v.T, v.T < high_temp)
            )
        )
        cur_tok = cur_tok + 2
    # not sure about having zero as implicit default value
    #expr_cond_pairs.append((0, True))
    return Piecewise(*expr_cond_pairs) #pylint: disable=W0142

def _tdb_grammar(): #pylint: disable=R0914
    """
    Convenience function for getting the pyparsing grammar of a TDB file.
    """
    int_number = Word(nums).setParseAction(lambda t: [int(t[0])])
    # matching float w/ regex is ugly but is recommended by pyparsing
    float_number = Regex(r'[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?') \
        .setParseAction(lambda t: [float(t[0])])
    # symbol name, e.g., phase name, function name
    symbol_name = Word(alphanums+'_', min=1)
    # species name, e.g., CO2, AL, FE3+
    species_name = Word(alphanums+'+-', min=1)
    constituent_array = Group(
        delimitedList(Group(delimitedList(species_name, ',')), ':')
        )
    param_types = CaselessKeyword('G') | CaselessKeyword('L') | \
                  CaselessKeyword('TC') | CaselessKeyword('BMAGN')
    # Let sympy do heavy arithmetic / algebra parsing for us
    # a convenience function will handle the piecewise details
    func_expr = float_number + OneOrMore(SkipTo(';') \
        + Suppress(';') + float_number + Suppress(Word('YNyn', exact=1)))
    # ELEMENT
    cmd_element = CaselessKeyword('ELEMENT') + Word(alphas+'/-', min=1, max=2)
    # TYPE_DEFINITION
    cmd_typedef = CaselessKeyword('TYPE_DEFINITION') + \
        Suppress(White()) + CharsNotIn(' !', exact=1) + SkipTo(LineEnd())
    # FUNCTION
    cmd_function = CaselessKeyword('FUNCTION') + symbol_name + \
        func_expr.setParseAction(_make_piecewise_ast)
    # DEFINE_SYSTEM_DEFAULT
    cmd_defsysdef = CaselessKeyword('DEFINE_SYSTEM_DEFAULT')
    # DEFAULT_COMMAND
    cmd_defcmd = CaselessKeyword('DEFAULT_COMMAND')
    # PHASE
    cmd_phase = CaselessKeyword('PHASE') + symbol_name + \
        Suppress(White()) + CharsNotIn(' !', min=1) + Suppress(White()) + \
        Suppress(int_number) + Group(OneOrMore(float_number)) + LineEnd()
    # CONSTITUENT
    cmd_constituent = CaselessKeyword('CONSTITUENT') + symbol_name + \
        Suppress(White()) + Suppress(':') + constituent_array + \
        Suppress(':') + LineEnd()
    # PARAMETER
    cmd_parameter = CaselessKeyword('PARAMETER') + param_types + \
        Suppress('(') + symbol_name + Suppress(',') + constituent_array + \
        Suppress(';') + int_number + Suppress(')') + \
        func_expr.setParseAction(_make_piecewise_ast)
    # Now combine the grammar together
    all_commands = cmd_element | \
                    cmd_typedef | \
                    cmd_function | \
                    cmd_defsysdef | \
                    cmd_defcmd | \
                    cmd_phase | \
                    cmd_constituent | \
                    cmd_parameter | \
                    Empty()
    return all_commands

def _process_typedef(targetdb, typechar, line):
    """
    Process the TYPE_DEFINITION command.
    """
    # ' GES A_P_D BCC_A2 MAGNETIC  -1    0.4
    tokens = line.split()
    if len(tokens) < 6:
        return
    if tokens[3].upper() == 'MAGNETIC':
        # magnetic model (IHJ model assumed by default)
        targetdb.typedefs[typechar] = {
            'ihj_magnetic':[float(tokens[4]), float(tokens[5])]
        }
    # GES A_P_D L12_FCC DIS_PART FCC_A1
    if tokens[3].upper() == 'DIS_PART':
        # order-disorder model
        targetdb.typedefs[typechar] = {
            'disordered_phase': tokens[4],
            'ordered_phase': tokens[2]
        }
        if tokens[2] in targetdb.phases:
            # Since TDB files do not enforce any kind of ordering
            # on the specification of ordered and disordered phases,
            # we need to handle the case of when either phase is specified
            # first. In this case, we imagine the ordered phase is
            # specified first. If the disordered phase is specified
            # first, we will have to catch it in _process_phase().
            targetdb.phases[tokens[2]].model_hints.update(
                targetdb.typedefs[typechar]
            )

def _process_phase(targetdb, name, typedefs, subls):
    """
    Process the PHASE command.
    """
    splitname = name.split(':')
    phase_name = splitname[0]
    options = None
    if len(splitname) > 1:
        options = splitname[1]
        print(options)
    targetdb.add_structure_entry(phase_name, phase_name)
    model_hints = {}
    for typedef in list(typedefs):
        if typedef in targetdb.typedefs.keys():
            if 'ihj_magnetic' in targetdb.typedefs[typedef].keys():
                model_hints['ihj_magnetic_afm_factor'] = \
                    targetdb.typedefs[typedef]['ihj_magnetic'][0]
                model_hints['ihj_magnetic_structure_factor'] = \
                    targetdb.typedefs[typedef]['ihj_magnetic'][1]
            if 'ordered_phase' in targetdb.typedefs[typedef].keys():
                model_hints['ordered_phase'] = \
                    targetdb.typedefs[typedef]['ordered_phase']
                model_hints['disordered_phase'] = \
                    targetdb.typedefs[typedef]['disordered_phase']
    targetdb.add_phase(phase_name, model_hints, subls)

def _process_parameter(targetdb, param_type, phase_name, #pylint: disable=R0913
                       constituent_array, param_order, param, ref=None):
    """
    Process the PARAMETER command.
    """
    targetdb.add_parameter(param_type, phase_name, constituent_array.asList(),
                           param_order, param, ref)

def _unimplemented(*args, **kwargs): #pylint: disable=W0613
    """
    Null function.
    """
    pass

_TDB_PROCESSOR = {
    'ELEMENT': lambda db, el: db.elements.add(el),
    'TYPE_DEFINITION': _process_typedef,
    'FUNCTION': lambda db, name, sym: db.symbols.__setitem__(name, sym),
    'DEFINE_SYSTEM_DEFAULT': _unimplemented,
    'DEFAULT_COMMAND': _unimplemented,
    'PHASE': _process_phase,
    'CONSTITUENT': \
        lambda db, name, c: db.add_phase_constituents(name.split(':')[0], c),
    'PARAMETER': _process_parameter
}
def tdbread(targetdb, lines):
    """
    Parse a TDB file into a pycalphad Database object.

    Parameters
    ----------
    targetdb : Database
        A pypycalphad Database.
    lines : string
        A raw TDB file.
    """
    lines = lines.replace('\t', ' ')
    lines = lines.strip()
    # Split the string by newlines
    splitlines = lines.split('\n')
    # Remove extra whitespace inside line
    splitlines = [' '.join(k.split()) for k in splitlines]
    # Remove comments
    splitlines = [k.strip().split('$', 1)[0] for k in splitlines]
    # Combine everything back together
    lines = ' '.join(splitlines)
    # Now split by the command delimeter
    commands = lines.split('!')
    # Filter out comments one more time
    # It's possible they were at the end of a command
    commands = [k.strip() for k in commands if not k.startswith("$")]

    for command in commands:
        try:
            tokens = None
            tokens = _tdb_grammar().parseString(command)
            if len(tokens) == 0:
                continue
            _TDB_PROCESSOR[tokens[0]](targetdb, *tokens[1:])
        except:
            print("Failed while parsing: " + command)
            print("Tokens: " + str(tokens))
            raise

if __name__ == "__main__":
    MYTDB = '''
$ CRFENI
$
$ TDB-file for the thermodynamic assessment of the Cr-Fe-Ni system
$
$-------------------------------------------------------------------------------
$ 2012.5.11
$ 
$ TDB file created by T.Abe, K.Hashimoto and Y.Sawada
$
$ Particle Simulation and Thermodynamics Group, National Institute for 
$ Materials Science. 1-2-1 Sengen, Tsukuba, Ibaraki 305-0047, Japan
$ e-mail: abe.taichi @ nims.go.jp
$ Copyright (C) NIMS 2012
$
$-------------------------------------------------------------------------------
$ PARAMETERS ARE TAKEN FROM
$ The parameter set is taken from
$ [1999Mie] Thermodynamic reassessment of Fe-Cr-Ni system with emphasis on the 
$ iron-rich corner, J.Miettinen, pycalphad, 23 (1999) 231-248.
$
$ [1987And] Thermodynamic properties of teh Cr-Fe system,
$ J.-O.Andersson and B.Sundman, pycalphad, 11 (1987), 83-92.
$
$ [1993Lee] Revision of thermodynamic descriptions of the Fe-Cr and Fe-Ni 
$ liquid phases, B.-J.Lee, pycalphad, 17 (1993), 251-268.
$
$ [1992Lee] On the stability of Cr-Carbides, B.-J.Lee, 
$ pycalphad, 16 (1992), 121-149.
$
$ [1990Hil] M.Hillert, C.Qiu, Metall. Trans.A, 21A (1990) 1673.
$
$ Unpublished works
$ [1985Xin] Magnetic parameters in the Cr-Fe system,
$ Z.S.Xing, D.D.Gohil, A.T.Dinsdale, T.Chart, NPL, DMA (a) 103, London, 1985.
$
$ [chart] Magnetic parameters in the Cr-Ni system are tha same as in T.Chart 
$ unpublished work referred in several papers, e.g. M.Kajihara and M.Hillert, 
$ Metall.Trans.A, 21A (1990) 2777-2787.
$
$ [1987Gus] P.Gustafson, Internal report, No.74, KTH, Sweden, Mar. 1987.
$
$-------------------------------------------------------------------------------
$ COMMENTS 
$ HCP is added in this file since it is included in 1992Lee. The sigma phase
$ is modeld with 8-4-18 type taken from [1987And]. 
$                                                                 T.A.
$ ------------------------------------------------------------------------------
Element /-          ELECTRON_GAS         0         0         0  !
Element VA                VACUUM         0         0         0  !
ELEMENT CR                BCC_A2    51.996      4050    23.560  !
ELEMENT FE                BCC_A2    55.847      4489    27.28   !
Element NI                FCC_A1    58.69       4787    29.7955 !
$--------1---------2---------3---------4---------5---------6---------7---------8 
$
FUNCTION GLIQCR  300 +15483.015+146.059775*T-26.908*T*LN(T)
   +.00189435*T**2-1.47721E-06*T**3+139250*T**(-1)+2.37615E-21*T**7;  2180 Y
   -16459.984+335.616316*T-50*T*LN(T);                                6000 N !
FUNCTION GBCCCR  300 -8856.94+157.48*T
 -26.908*T*LN(T)+.00189435*T**2-1.47721E-06*T**3+139250*T**(-1);      2180 Y
  -34869.344+344.18*T-50*T*LN(T)-2.885261E+32*T**(-9);                6000 N !
FUNCTION GFCCCR  300 -1572.94+157.643*T  
 -26.908*T*LN(T)+.00189435*T**2-1.47721E-06*T**3+139250*T**(-1);      2180 Y
  -27585.344+344.343*T-50*T*LN(T)-2.885261E+32*T**(-9);               6000 N !
Function GHCPCR  300  +4438+GBCCCR;                                   6000 N !
$
FUNCTION GBCCFE  300  +1225.7+124.134*T-23.5143*T*LN(T)
     -.00439752*T**2-5.8927E-08*T**3+77359*T**(-1);                    1811 Y
      -25383.581+299.31255*T-46*T*LN(T)+2.29603E+31*T**(-9);           6000 N !
FUNCTION GFCCFE 300 -1462.4+8.282*T-1.15*T*LN(T)+6.4E-4*T**2+GBCCFE;   1811 Y
      -1713.815+0.94001*T+4.9251E+30*T**(-9)+GBCCFE;                   6000 N !
FUNCTION GHCPFE 300 -3705.78+12.591*T-1.15*T*LN(T)+6.4E-4*T**2+GBCCFE; 1811 Y
      -3957.199+5.24951*T+4.9251E+30*T**(-9)+GBCCFE;                   6000 N !
FUNCTION GLIQFE  300  +13265.87+117.57557*T-23.5143*T*LN(T)
      -0.00439752*T**2-5.8927E-08*T**3+77359*T**(-1)-3.67516E-21*T**7; 1811 Y
      -10838.83+291.302*T-46*T*LN(T);                                  6000 N !
$
FUNCTION GFCCNI  300 -5179.159+117.854*T
      -22.096*T*ln(T)-0.0048407*T**2;                                 1728 Y
      -27840.655+279.135*T-43.1*T*ln(T)+1.12754e+031*T**(-9);         3000 N !
FUNCTION GLIQNI  300  11235.527+108.457*T-22.096*T*ln(T)
      -0.0048407*T**2-3.82318e-021*T**7;                              1728 Y
      -9549.775+268.598*T-43.1*T*ln(T);                               3000 N !
Function GHCPNI     300  +1046+1.255*T+GFCCNI;                        6000 N !
FUNCTION GBCCNI     300  +8715.084-3.556*T+GFCCNI;                    6000 N !
$
FUNCTION ZERO       300  +0;                                          6000 N !
FUNCTION UN_ASS     300  +0;                                          6000 N !
$
$ ------------------------------------------------------------------------------
TYPE_DEFINITION % SEQ * !
DEFINE_SYSTEM_DEFAULT ELEMENT 3 !
DEFAULT_COMMAND DEFINE_SYS_ELEMENT VA /- !
$
TYPE_DEFINITION ' GES A_P_D BCC_A2 MAGNETIC  -1    0.4  !
Type_Definition ( GES A_P_D FCC_A1 Magnetic  -3    0.28 !
Type_Definition ) GES A_P_D HCP_A3 Magnetic  -3    0.28 !
$
$ ------------------------------------------------------------------------------
 Phase LIQUID % 1 1 !
 Constituent LIQUID : CR,FE,NI : !

 PARAMETER G(LIQUID,CR;0)     300  +GLIQCR;               6000 N !
 Parameter G(LIQUID,FE;0)     300  +GLIQFE;               6000 N !
 Parameter G(LIQUID,NI;0)     300  +GLIQNI;               6000 N !
$
 PARAMETER G(LIQUID,CR,FE;0)  300  -17737+7.996546*T;     6000 N ! $1993Lee  
 PARAMETER G(LIQUID,CR,FE;1)  300  -1331;                 6000 N ! $1993Lee 
 Parameter G(LIQUID,CR,NI;0)  300  +318-7.3318*T;         6000 N ! $1992Lee 
 Parameter G(LIQUID,CR,NI;1)  300  +16941-6.3696*T;       6000 N ! $1992Lee 
 PARAMETER G(LIQUID,FE,NI;0)  300  -16911+5.1622*T;       6000 N ! $1993Lee 
 PARAMETER G(LIQUID,FE,NI;1)  300  +10180-4.146656*T;     6000 N ! $1993Lee 
$
  PARAMETER G(LIQUID,CR,FE,NI;0) 300 +130000-50*T;         6000 N ! $1999Mie 
  PARAMETER G(LIQUID,CR,FE,NI;1) 300 +80000-50*T;          6000 N ! $1999Mie 
  PARAMETER G(LIQUID,CR,FE,NI;2) 300 +60000-50*T;          6000 N ! $1999Mie 
$
$ ------------------------------------------------------------------------------
PHASE BCC_A2  %'  2 1   3 !
CONSTITUENT BCC_A2  : CR,FE,NI : VA :  !
$
 PARAMETER G(BCC_A2,CR:VA;0)      300 +GBCCCR;             6000 N !
 PARAMETER G(BCC_A2,FE:VA;0)      300 +GBCCFE;             6000 N !
 PARAMETER G(BCC_A2,NI:VA;0)      300 +GBCCNI;             3000 N ! 
 Parameter TC(BCC_A2,CR:VA;0)     300 -311.5;              6000 N !
 PARAMETER TC(BCC_A2,FE:VA;0)     300 +1043;               6000 N !
 PARAMETER TC(BCC_A2,NI:VA;0)     300 +575;                6000 N ! 
 Parameter BMAGN(BCC_A2,CR:VA;0)  300 -0.008;              6000 N !
 PARAMETER BMAGN(BCC_A2,FE:VA;0)  300 +2.22;               6000 N !
 PARAMETER BMAGN(BCC_A2,NI:VA;0)  300 +0.85;               6000 N !
$ 
 PARAMETER TC(BCC_A2,CR,FE:VA;0)  300 +1650;               6000 N ! $1987And
 PARAMETER TC(BCC_A2,CR,FE:VA;1)  300 +550;                6000 N ! $1987And
 Parameter TC(BCC_A2,CR,NI:VA;0)  300 +2373;               6000 N ! $chart
 Parameter TC(BCC_A2,CR,NI:VA;1)  300 +617;                6000 N ! $chart
 PARAMETER TC(BCC_A2,FE,NI:VA;0)    300 +ZERO;             6000 N ! $1985Xing
 PARAMETER BMAGN(BCC_A2,CR,FE:VA;0) 300 -0.85;             6000 N ! $1987And
 Parameter BMAGN(BCC_A2,CR,NI:VA;0) 300 +4;                6000 N ! $chart
 PARAMETER BMAGN(BCC_A2,FE,NI:VA;0) 300 +ZERO;             6000 N ! $1985Xing 
$
 PARAMETER G(BCC_A2,CR,FE:VA;0)   300 +20500-9.68*T;       6000 N ! $1987And
 Parameter G(BCC_A2,CR,NI:VA;0)   300 +17170-11.8199*T;    6000 N ! $1992Lee
 Parameter G(BCC_A2,CR,NI:VA;1)   300 +34418-11.8577*T;    6000 N ! $1992Lee
 PARAMETER G(BCC_A2,FE,NI:VA;0)   300 -956.63-1.28726*T;   6000 N ! $1985Xin
 PARAMETER G(BCC_A2,FE,NI:VA;1)   300 +1789.03-1.92912*T;  6000 N ! $1985Xin 
$
 PARAMETER G(BCC_A2,CR,FE,NI:VA;0) 300 +6000.+10*T;        6000 N ! $1999Mie 
 PARAMETER G(BCC_A2,CR,FE,NI:VA;1) 300 -18500+10*T;        6000 N ! $1999Mie 
 PARAMETER G(BCC_A2,CR,FE,NI:VA;2) 300 -27000+10*T;        6000 N ! $1999Mie 
$ ------------------------------------------------------------------------------
Phase FCC_A1 %( 2  1  1  !
Constituent FCC_A1 : CR,FE,NI : VA : !
$
 PARAMETER G(FCC_A1,CR:VA;0)      300  +GFCCCR;             6000 N !
 PARAMETER G(FCC_A1,FE:VA;0)      300  +GFCCFE;             6000 N !
 Parameter G(FCC_A1,NI:VA;0)      300  +GFCCNI;             3000 N !
 PARAMETER TC(FCC_A1,CR:VA;0)     300  -1109;               6000 N !
 PARAMETER TC(FCC_A1,FE:VA;0)     300  -201;                6000 N !
 PARAMETER TC(FCC_A1,NI:VA;0)     300  +633;                6000 N !
 PARAMETER BMAGN(FCC_A1,CR:VA;0)  300  -2.46;               6000 N !
 PARAMETER BMAGN(FCC_A1,FE:VA;0)  300  -2.1;                6000 N !
 PARAMETER BMAGN(FCC_A1,NI:VA;0)  300  +0.52;               6000 N !
$
 Parameter TC(FCC_A1,CR,FE:VA;0)    300  +UN_ASS;           6000 N !
 Parameter TC(FCC_A1,CR,NI:VA;0)    300  -3605;             6000 N ! $UPW-cha
 PARAMETER TC(FCC_A1,FE,NI:VA;0)    300  +2133;             6000 N ! $1985Xing
 PARAMETER TC(FCC_A1,FE,NI:VA;1)    300  -682;              6000 N ! $1985Xing 
 Parameter BMAGN(FCC_A1,CR,FE:VA;0) 300  +UN_ASS;           6000 N !
 Parameter BMAGN(FCC_A1,CR,NI:VA;0) 300  -1.91;             6000 N ! $UPW-cha
 PARAMETER BMAGN(FCC_A1,FE,NI:VA;0) 300  +9.55;             6000 N ! $1985Xing 
 PARAMETER BMAGN(FCC_A1,FE,NI:VA;1) 300  +7.23;             6000 N ! $1985Xing 
 PARAMETER BMAGN(FCC_A1,FE,NI:VA;2) 300  +5.93;             6000 N ! $1985Xing 
 PARAMETER BMAGN(FCC_A1,FE,NI:VA;3) 300  +6.18;             6000 N ! $1985Xing 
$
 PARAMETER G(FCC_A1,CR,FE:VA;0)  300 +10833-7.477*T;        6000 N ! $1987And
 PARAMETER G(FCC_A1,CR,FE:VA;1)  300 +1410;                 6000 N ! $1987And
 Parameter G(FCC_A1,CR,NI:VA;0)  300 +8030-12.8801*T;       6000 N ! $1992Lee
 Parameter G(FCC_A1,CR,NI:VA;1)  300 +33080-16.0362*T;      6000 N ! $1992Lee
 PARAMETER G(FCC_A1,FE,NI:VA;0)  300 -12054.355+3.27413*T;  6000 N ! $1985Xing 
 PARAMETER G(FCC_A1,FE,NI:VA;1)  300 +11082.1315-4.4507*T;  6000 N ! $1985Xing 
 PARAMETER G(FCC_A1,FE,NI:VA;2)  300 -725.805174;           6000 N ! $1985Xing 
$
 PARAMETER G(FCC_A1,CR,FE,NI:VA;0) 300 +10000+10*T;         6000 N ! $1999Mie 
 PARAMETER G(FCC_A1,CR,FE,NI:VA;1) 300 -6500;               6000 N ! $1999Mie  
 PARAMETER G(FCC_A1,CR,FE,NI:VA;2) 300 +48000;              6000 N ! $1999Mie 
$
$ ------------------------------------------------------------------------------
Phase HCP_A3 %) 2  1  0.5  !
Constituent HCP_A3 : CR,FE,NI : VA : !
$
 PARAMETER G(HCP_A3,CR:VA;0)      300  +GHCPCR;             6000 N ! $1992Lee
 PARAMETER G(HCP_A3,FE:VA;0)      300  +GHCPFE;             6000 N ! $1992Lee
 Parameter G(HCP_A3,NI:VA;0)      300  +GHCPNI;             3000 N ! $1992Lee
 PARAMETER TC(HCP_A3,CR:VA;0)     300  -1109;               6000 N ! $1992Lee
 PARAMETER TC(HCP_A3,FE:VA;0)     300  +ZERO;               6000 N ! $1992Lee
 PARAMETER TC(HCP_A3,NI:VA;0)     300  +633;                6000 N ! $1992Lee
 PARAMETER BMAGN(HCP_A3,CR:VA;0)  300  -2.46;               6000 N ! $1992Lee
 PARAMETER BMAGN(HCP_A3,FE:VA;0)  300  +ZERO;               6000 N ! $1992Lee
 PARAMETER BMAGN(HCP_A3,NI:VA;0)  300  +0.52;               6000 N ! $1992Lee
$
 PARAMETER G(HCP_A3,CR,FE:VA;0)   300  +10833-7.477*T;      6000 N ! $1992Lee
$
$ ------------------------------------------------------------------------------
 PHASE SIGMA % 3  8  4  18 !
  CONSTITUENT SIGMA : FE,NI : CR : CR,FE,NI : !

  PARAMETER G(SIGMA,FE:CR:CR;0) 300 
              +92300.-95.96*T+8*GFCCFE+4*GBCCCR+18*GBCCCR;   6000 N ! $1987And
  PARAMETER G(SIGMA,FE:CR:FE;0) 300 
              +117300-95.96*T+8*GFCCFE+4*GBCCCR+18*GBCCFE;   6000 N ! $1987And
  PARAMETER G(SIGMA,FE:CR:NI;0) 300 
                             +8*GFCCFE+4*GBCCCR+18*GBCCNI;   6000 N ! $1990Hil
  PARAMETER G(SIGMA,NI:CR:CR;0) 300 
              +180000-170*T  +8*GFCCNI+4*GBCCCR+18*GBCCCR;   6000 N ! $1999Mie
  PARAMETER G(SIGMA,NI:CR:FE;0) 300 
                             +8*GFCCNI+4*GBCCCR+18*GBCCFE;   6000 N ! $1990Hil
  PARAMETER G(SIGMA,NI:CR:NI;0) 300 
              +175400        +8*GFCCNI+4*GBCCCR+18*GBCCNI;   6000 N ! $1987Gus
$
$ ------------------------------------------------------------------------------
$CRFENI-NIMS















'''
    from pycalphad.io.database import Database
    TESTDB = Database()
    tdbread(TESTDB, MYTDB)
