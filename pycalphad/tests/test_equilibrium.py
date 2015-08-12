"""
The equilibrium test module verifies that the Equilibrium class produces the
correct solution for thermodynamic equilibrium.
"""

from unittest.case import SkipTest
from numpy.testing import assert_allclose
from pycalphad import Database, calculate, equilibrium
import pycalphad.variables as v

ROSE_TEST_STRING = """
ELEMENT H                 TEST    0      0    0  !
ELEMENT HE                TEST    0      0    0  !
ELEMENT LI                TEST    0      0    0  !
ELEMENT BE                TEST    0      0    0  !
ELEMENT B                 TEST    0      0    0  !
ELEMENT C                 TEST    0      0    0  !
ELEMENT N                 TEST    0      0    0  !
ELEMENT O                 TEST    0      0    0  !
ELEMENT F                 TEST    0      0    0  !
ELEMENT NE                TEST    0      0    0  !

TYPE_DEFINITION % SEQ * !
FUNCTION STR 300 200000; 6000 N !

 PHASE TEST % 1 1 !
 CONSTITUENT TEST : H,HE,LI,BE,B,C,N,O,F,NE : !

PARAMETER G(TEST,H,HE,LI;0)     300  STR#;               6000 N !
PARAMETER G(TEST,H,HE,LI;1)     300  STR#;               6000 N !
PARAMETER G(TEST,H,HE,LI;2)     300  STR#;               6000 N !
PARAMETER G(TEST,H,HE,BE;0)     300  STR#;               6000 N !
PARAMETER G(TEST,H,HE,BE;1)     300  STR#;               6000 N !
PARAMETER G(TEST,H,HE,BE;2)     300  STR#;               6000 N !
PARAMETER G(TEST,H,HE,B;0)     300  STR#;               6000 N !
PARAMETER G(TEST,H,HE,B;1)     300  STR#;               6000 N !
PARAMETER G(TEST,H,HE,B;2)     300  STR#;               6000 N !
PARAMETER G(TEST,H,HE,C;0)     300  STR#;               6000 N !
PARAMETER G(TEST,H,HE,C;1)     300  STR#;               6000 N !
PARAMETER G(TEST,H,HE,C;2)     300  STR#;               6000 N !
PARAMETER G(TEST,H,HE,N;0)     300  STR#;               6000 N !
PARAMETER G(TEST,H,HE,N;1)     300  STR#;               6000 N !
PARAMETER G(TEST,H,HE,N;2)     300  STR#;               6000 N !
PARAMETER G(TEST,H,HE,O;0)     300  STR#;               6000 N !
PARAMETER G(TEST,H,HE,O;1)     300  STR#;               6000 N !
PARAMETER G(TEST,H,HE,O;2)     300  STR#;               6000 N !
PARAMETER G(TEST,H,HE,F;0)     300  STR#;               6000 N !
PARAMETER G(TEST,H,HE,F;1)     300  STR#;               6000 N !
PARAMETER G(TEST,H,HE,F;2)     300  STR#;               6000 N !
PARAMETER G(TEST,H,HE,NE;0)     300  STR#;               6000 N !
PARAMETER G(TEST,H,HE,NE;1)     300  STR#;               6000 N !
PARAMETER G(TEST,H,HE,NE;2)     300  STR#;               6000 N !
PARAMETER G(TEST,H,LI,BE;0)     300  STR#;               6000 N !
PARAMETER G(TEST,H,LI,BE;1)     300  STR#;               6000 N !
PARAMETER G(TEST,H,LI,BE;2)     300  STR#;               6000 N !
PARAMETER G(TEST,H,LI,B;0)     300  STR#;               6000 N !
PARAMETER G(TEST,H,LI,B;1)     300  STR#;               6000 N !
PARAMETER G(TEST,H,LI,B;2)     300  STR#;               6000 N !
PARAMETER G(TEST,H,LI,C;0)     300  STR#;               6000 N !
PARAMETER G(TEST,H,LI,C;1)     300  STR#;               6000 N !
PARAMETER G(TEST,H,LI,C;2)     300  STR#;               6000 N !
PARAMETER G(TEST,H,LI,N;0)     300  STR#;               6000 N !
PARAMETER G(TEST,H,LI,N;1)     300  STR#;               6000 N !
PARAMETER G(TEST,H,LI,N;2)     300  STR#;               6000 N !
PARAMETER G(TEST,H,LI,O;0)     300  STR#;               6000 N !
PARAMETER G(TEST,H,LI,O;1)     300  STR#;               6000 N !
PARAMETER G(TEST,H,LI,O;2)     300  STR#;               6000 N !
PARAMETER G(TEST,H,LI,F;0)     300  STR#;               6000 N !
PARAMETER G(TEST,H,LI,F;1)     300  STR#;               6000 N !
PARAMETER G(TEST,H,LI,F;2)     300  STR#;               6000 N !
PARAMETER G(TEST,H,LI,NE;0)     300  STR#;               6000 N !
PARAMETER G(TEST,H,LI,NE;1)     300  STR#;               6000 N !
PARAMETER G(TEST,H,LI,NE;2)     300  STR#;               6000 N !
PARAMETER G(TEST,H,BE,B;0)     300  STR#;               6000 N !
PARAMETER G(TEST,H,BE,B;1)     300  STR#;               6000 N !
PARAMETER G(TEST,H,BE,B;2)     300  STR#;               6000 N !
PARAMETER G(TEST,H,BE,C;0)     300  STR#;               6000 N !
PARAMETER G(TEST,H,BE,C;1)     300  STR#;               6000 N !
PARAMETER G(TEST,H,BE,C;2)     300  STR#;               6000 N !
PARAMETER G(TEST,H,BE,N;0)     300  STR#;               6000 N !
PARAMETER G(TEST,H,BE,N;1)     300  STR#;               6000 N !
PARAMETER G(TEST,H,BE,N;2)     300  STR#;               6000 N !
PARAMETER G(TEST,H,BE,O;0)     300  STR#;               6000 N !
PARAMETER G(TEST,H,BE,O;1)     300  STR#;               6000 N !
PARAMETER G(TEST,H,BE,O;2)     300  STR#;               6000 N !
PARAMETER G(TEST,H,BE,F;0)     300  STR#;               6000 N !
PARAMETER G(TEST,H,BE,F;1)     300  STR#;               6000 N !
PARAMETER G(TEST,H,BE,F;2)     300  STR#;               6000 N !
PARAMETER G(TEST,H,BE,NE;0)     300  STR#;               6000 N !
PARAMETER G(TEST,H,BE,NE;1)     300  STR#;               6000 N !
PARAMETER G(TEST,H,BE,NE;2)     300  STR#;               6000 N !
PARAMETER G(TEST,H,B,C;0)     300  STR#;               6000 N !
PARAMETER G(TEST,H,B,C;1)     300  STR#;               6000 N !
PARAMETER G(TEST,H,B,C;2)     300  STR#;               6000 N !
PARAMETER G(TEST,H,B,N;0)     300  STR#;               6000 N !
PARAMETER G(TEST,H,B,N;1)     300  STR#;               6000 N !
PARAMETER G(TEST,H,B,N;2)     300  STR#;               6000 N !
PARAMETER G(TEST,H,B,O;0)     300  STR#;               6000 N !
PARAMETER G(TEST,H,B,O;1)     300  STR#;               6000 N !
PARAMETER G(TEST,H,B,O;2)     300  STR#;               6000 N !
PARAMETER G(TEST,H,B,F;0)     300  STR#;               6000 N !
PARAMETER G(TEST,H,B,F;1)     300  STR#;               6000 N !
PARAMETER G(TEST,H,B,F;2)     300  STR#;               6000 N !
PARAMETER G(TEST,H,B,NE;0)     300  STR#;               6000 N !
PARAMETER G(TEST,H,B,NE;1)     300  STR#;               6000 N !
PARAMETER G(TEST,H,B,NE;2)     300  STR#;               6000 N !
PARAMETER G(TEST,H,C,N;0)     300  STR#;               6000 N !
PARAMETER G(TEST,H,C,N;1)     300  STR#;               6000 N !
PARAMETER G(TEST,H,C,N;2)     300  STR#;               6000 N !
PARAMETER G(TEST,H,C,O;0)     300  STR#;               6000 N !
PARAMETER G(TEST,H,C,O;1)     300  STR#;               6000 N !
PARAMETER G(TEST,H,C,O;2)     300  STR#;               6000 N !
PARAMETER G(TEST,H,C,F;0)     300  STR#;               6000 N !
PARAMETER G(TEST,H,C,F;1)     300  STR#;               6000 N !
PARAMETER G(TEST,H,C,F;2)     300  STR#;               6000 N !
PARAMETER G(TEST,H,C,NE;0)     300  STR#;               6000 N !
PARAMETER G(TEST,H,C,NE;1)     300  STR#;               6000 N !
PARAMETER G(TEST,H,C,NE;2)     300  STR#;               6000 N !
PARAMETER G(TEST,H,N,O;0)     300  STR#;               6000 N !
PARAMETER G(TEST,H,N,O;1)     300  STR#;               6000 N !
PARAMETER G(TEST,H,N,O;2)     300  STR#;               6000 N !
PARAMETER G(TEST,H,N,F;0)     300  STR#;               6000 N !
PARAMETER G(TEST,H,N,F;1)     300  STR#;               6000 N !
PARAMETER G(TEST,H,N,F;2)     300  STR#;               6000 N !
PARAMETER G(TEST,H,N,NE;0)     300  STR#;               6000 N !
PARAMETER G(TEST,H,N,NE;1)     300  STR#;               6000 N !
PARAMETER G(TEST,H,N,NE;2)     300  STR#;               6000 N !
PARAMETER G(TEST,H,O,F;0)     300  STR#;               6000 N !
PARAMETER G(TEST,H,O,F;1)     300  STR#;               6000 N !
PARAMETER G(TEST,H,O,F;2)     300  STR#;               6000 N !
PARAMETER G(TEST,H,O,NE;0)     300  STR#;               6000 N !
PARAMETER G(TEST,H,O,NE;1)     300  STR#;               6000 N !
PARAMETER G(TEST,H,O,NE;2)     300  STR#;               6000 N !
PARAMETER G(TEST,H,F,NE;0)     300  STR#;               6000 N !
PARAMETER G(TEST,H,F,NE;1)     300  STR#;               6000 N !
PARAMETER G(TEST,H,F,NE;2)     300  STR#;               6000 N !
PARAMETER G(TEST,HE,LI,BE;0)     300  STR#;               6000 N !
PARAMETER G(TEST,HE,LI,BE;1)     300  STR#;               6000 N !
PARAMETER G(TEST,HE,LI,BE;2)     300  STR#;               6000 N !
PARAMETER G(TEST,HE,LI,B;0)     300  STR#;               6000 N !
PARAMETER G(TEST,HE,LI,B;1)     300  STR#;               6000 N !
PARAMETER G(TEST,HE,LI,B;2)     300  STR#;               6000 N !
PARAMETER G(TEST,HE,LI,C;0)     300  STR#;               6000 N !
PARAMETER G(TEST,HE,LI,C;1)     300  STR#;               6000 N !
PARAMETER G(TEST,HE,LI,C;2)     300  STR#;               6000 N !
PARAMETER G(TEST,HE,LI,N;0)     300  STR#;               6000 N !
PARAMETER G(TEST,HE,LI,N;1)     300  STR#;               6000 N !
PARAMETER G(TEST,HE,LI,N;2)     300  STR#;               6000 N !
PARAMETER G(TEST,HE,LI,O;0)     300  STR#;               6000 N !
PARAMETER G(TEST,HE,LI,O;1)     300  STR#;               6000 N !
PARAMETER G(TEST,HE,LI,O;2)     300  STR#;               6000 N !
PARAMETER G(TEST,HE,LI,F;0)     300  STR#;               6000 N !
PARAMETER G(TEST,HE,LI,F;1)     300  STR#;               6000 N !
PARAMETER G(TEST,HE,LI,F;2)     300  STR#;               6000 N !
PARAMETER G(TEST,HE,LI,NE;0)     300  STR#;               6000 N !
PARAMETER G(TEST,HE,LI,NE;1)     300  STR#;               6000 N !
PARAMETER G(TEST,HE,LI,NE;2)     300  STR#;               6000 N !
PARAMETER G(TEST,HE,BE,B;0)     300  STR#;               6000 N !
PARAMETER G(TEST,HE,BE,B;1)     300  STR#;               6000 N !
PARAMETER G(TEST,HE,BE,B;2)     300  STR#;               6000 N !
PARAMETER G(TEST,HE,BE,C;0)     300  STR#;               6000 N !
PARAMETER G(TEST,HE,BE,C;1)     300  STR#;               6000 N !
PARAMETER G(TEST,HE,BE,C;2)     300  STR#;               6000 N !
PARAMETER G(TEST,HE,BE,N;0)     300  STR#;               6000 N !
PARAMETER G(TEST,HE,BE,N;1)     300  STR#;               6000 N !
PARAMETER G(TEST,HE,BE,N;2)     300  STR#;               6000 N !
PARAMETER G(TEST,HE,BE,O;0)     300  STR#;               6000 N !
PARAMETER G(TEST,HE,BE,O;1)     300  STR#;               6000 N !
PARAMETER G(TEST,HE,BE,O;2)     300  STR#;               6000 N !
PARAMETER G(TEST,HE,BE,F;0)     300  STR#;               6000 N !
PARAMETER G(TEST,HE,BE,F;1)     300  STR#;               6000 N !
PARAMETER G(TEST,HE,BE,F;2)     300  STR#;               6000 N !
PARAMETER G(TEST,HE,BE,NE;0)     300  STR#;               6000 N !
PARAMETER G(TEST,HE,BE,NE;1)     300  STR#;               6000 N !
PARAMETER G(TEST,HE,BE,NE;2)     300  STR#;               6000 N !
PARAMETER G(TEST,HE,B,C;0)     300  STR#;               6000 N !
PARAMETER G(TEST,HE,B,C;1)     300  STR#;               6000 N !
PARAMETER G(TEST,HE,B,C;2)     300  STR#;               6000 N !
PARAMETER G(TEST,HE,B,N;0)     300  STR#;               6000 N !
PARAMETER G(TEST,HE,B,N;1)     300  STR#;               6000 N !
PARAMETER G(TEST,HE,B,N;2)     300  STR#;               6000 N !
PARAMETER G(TEST,HE,B,O;0)     300  STR#;               6000 N !
PARAMETER G(TEST,HE,B,O;1)     300  STR#;               6000 N !
PARAMETER G(TEST,HE,B,O;2)     300  STR#;               6000 N !
PARAMETER G(TEST,HE,B,F;0)     300  STR#;               6000 N !
PARAMETER G(TEST,HE,B,F;1)     300  STR#;               6000 N !
PARAMETER G(TEST,HE,B,F;2)     300  STR#;               6000 N !
PARAMETER G(TEST,HE,B,NE;0)     300  STR#;               6000 N !
PARAMETER G(TEST,HE,B,NE;1)     300  STR#;               6000 N !
PARAMETER G(TEST,HE,B,NE;2)     300  STR#;               6000 N !
PARAMETER G(TEST,HE,C,N;0)     300  STR#;               6000 N !
PARAMETER G(TEST,HE,C,N;1)     300  STR#;               6000 N !
PARAMETER G(TEST,HE,C,N;2)     300  STR#;               6000 N !
PARAMETER G(TEST,HE,C,O;0)     300  STR#;               6000 N !
PARAMETER G(TEST,HE,C,O;1)     300  STR#;               6000 N !
PARAMETER G(TEST,HE,C,O;2)     300  STR#;               6000 N !
PARAMETER G(TEST,HE,C,F;0)     300  STR#;               6000 N !
PARAMETER G(TEST,HE,C,F;1)     300  STR#;               6000 N !
PARAMETER G(TEST,HE,C,F;2)     300  STR#;               6000 N !
PARAMETER G(TEST,HE,C,NE;0)     300  STR#;               6000 N !
PARAMETER G(TEST,HE,C,NE;1)     300  STR#;               6000 N !
PARAMETER G(TEST,HE,C,NE;2)     300  STR#;               6000 N !
PARAMETER G(TEST,HE,N,O;0)     300  STR#;               6000 N !
PARAMETER G(TEST,HE,N,O;1)     300  STR#;               6000 N !
PARAMETER G(TEST,HE,N,O;2)     300  STR#;               6000 N !
PARAMETER G(TEST,HE,N,F;0)     300  STR#;               6000 N !
PARAMETER G(TEST,HE,N,F;1)     300  STR#;               6000 N !
PARAMETER G(TEST,HE,N,F;2)     300  STR#;               6000 N !
PARAMETER G(TEST,HE,N,NE;0)     300  STR#;               6000 N !
PARAMETER G(TEST,HE,N,NE;1)     300  STR#;               6000 N !
PARAMETER G(TEST,HE,N,NE;2)     300  STR#;               6000 N !
PARAMETER G(TEST,HE,O,F;0)     300  STR#;               6000 N !
PARAMETER G(TEST,HE,O,F;1)     300  STR#;               6000 N !
PARAMETER G(TEST,HE,O,F;2)     300  STR#;               6000 N !
PARAMETER G(TEST,HE,O,NE;0)     300  STR#;               6000 N !
PARAMETER G(TEST,HE,O,NE;1)     300  STR#;               6000 N !
PARAMETER G(TEST,HE,O,NE;2)     300  STR#;               6000 N !
PARAMETER G(TEST,HE,F,NE;0)     300  STR#;               6000 N !
PARAMETER G(TEST,HE,F,NE;1)     300  STR#;               6000 N !
PARAMETER G(TEST,HE,F,NE;2)     300  STR#;               6000 N !
PARAMETER G(TEST,LI,BE,B;0)     300  STR#;               6000 N !
PARAMETER G(TEST,LI,BE,B;1)     300  STR#;               6000 N !
PARAMETER G(TEST,LI,BE,B;2)     300  STR#;               6000 N !
PARAMETER G(TEST,LI,BE,C;0)     300  STR#;               6000 N !
PARAMETER G(TEST,LI,BE,C;1)     300  STR#;               6000 N !
PARAMETER G(TEST,LI,BE,C;2)     300  STR#;               6000 N !
PARAMETER G(TEST,LI,BE,N;0)     300  STR#;               6000 N !
PARAMETER G(TEST,LI,BE,N;1)     300  STR#;               6000 N !
PARAMETER G(TEST,LI,BE,N;2)     300  STR#;               6000 N !
PARAMETER G(TEST,LI,BE,O;0)     300  STR#;               6000 N !
PARAMETER G(TEST,LI,BE,O;1)     300  STR#;               6000 N !
PARAMETER G(TEST,LI,BE,O;2)     300  STR#;               6000 N !
PARAMETER G(TEST,LI,BE,F;0)     300  STR#;               6000 N !
PARAMETER G(TEST,LI,BE,F;1)     300  STR#;               6000 N !
PARAMETER G(TEST,LI,BE,F;2)     300  STR#;               6000 N !
PARAMETER G(TEST,LI,BE,NE;0)     300  STR#;               6000 N !
PARAMETER G(TEST,LI,BE,NE;1)     300  STR#;               6000 N !
PARAMETER G(TEST,LI,BE,NE;2)     300  STR#;               6000 N !
PARAMETER G(TEST,LI,B,C;0)     300  STR#;               6000 N !
PARAMETER G(TEST,LI,B,C;1)     300  STR#;               6000 N !
PARAMETER G(TEST,LI,B,C;2)     300  STR#;               6000 N !
PARAMETER G(TEST,LI,B,N;0)     300  STR#;               6000 N !
PARAMETER G(TEST,LI,B,N;1)     300  STR#;               6000 N !
PARAMETER G(TEST,LI,B,N;2)     300  STR#;               6000 N !
PARAMETER G(TEST,LI,B,O;0)     300  STR#;               6000 N !
PARAMETER G(TEST,LI,B,O;1)     300  STR#;               6000 N !
PARAMETER G(TEST,LI,B,O;2)     300  STR#;               6000 N !
PARAMETER G(TEST,LI,B,F;0)     300  STR#;               6000 N !
PARAMETER G(TEST,LI,B,F;1)     300  STR#;               6000 N !
PARAMETER G(TEST,LI,B,F;2)     300  STR#;               6000 N !
PARAMETER G(TEST,LI,B,NE;0)     300  STR#;               6000 N !
PARAMETER G(TEST,LI,B,NE;1)     300  STR#;               6000 N !
PARAMETER G(TEST,LI,B,NE;2)     300  STR#;               6000 N !
PARAMETER G(TEST,LI,C,N;0)     300  STR#;               6000 N !
PARAMETER G(TEST,LI,C,N;1)     300  STR#;               6000 N !
PARAMETER G(TEST,LI,C,N;2)     300  STR#;               6000 N !
PARAMETER G(TEST,LI,C,O;0)     300  STR#;               6000 N !
PARAMETER G(TEST,LI,C,O;1)     300  STR#;               6000 N !
PARAMETER G(TEST,LI,C,O;2)     300  STR#;               6000 N !
PARAMETER G(TEST,LI,C,F;0)     300  STR#;               6000 N !
PARAMETER G(TEST,LI,C,F;1)     300  STR#;               6000 N !
PARAMETER G(TEST,LI,C,F;2)     300  STR#;               6000 N !
PARAMETER G(TEST,LI,C,NE;0)     300  STR#;               6000 N !
PARAMETER G(TEST,LI,C,NE;1)     300  STR#;               6000 N !
PARAMETER G(TEST,LI,C,NE;2)     300  STR#;               6000 N !
PARAMETER G(TEST,LI,N,O;0)     300  STR#;               6000 N !
PARAMETER G(TEST,LI,N,O;1)     300  STR#;               6000 N !
PARAMETER G(TEST,LI,N,O;2)     300  STR#;               6000 N !
PARAMETER G(TEST,LI,N,F;0)     300  STR#;               6000 N !
PARAMETER G(TEST,LI,N,F;1)     300  STR#;               6000 N !
PARAMETER G(TEST,LI,N,F;2)     300  STR#;               6000 N !
PARAMETER G(TEST,LI,N,NE;0)     300  STR#;               6000 N !
PARAMETER G(TEST,LI,N,NE;1)     300  STR#;               6000 N !
PARAMETER G(TEST,LI,N,NE;2)     300  STR#;               6000 N !
PARAMETER G(TEST,LI,O,F;0)     300  STR#;               6000 N !
PARAMETER G(TEST,LI,O,F;1)     300  STR#;               6000 N !
PARAMETER G(TEST,LI,O,F;2)     300  STR#;               6000 N !
PARAMETER G(TEST,LI,O,NE;0)     300  STR#;               6000 N !
PARAMETER G(TEST,LI,O,NE;1)     300  STR#;               6000 N !
PARAMETER G(TEST,LI,O,NE;2)     300  STR#;               6000 N !
PARAMETER G(TEST,LI,F,NE;0)     300  STR#;               6000 N !
PARAMETER G(TEST,LI,F,NE;1)     300  STR#;               6000 N !
PARAMETER G(TEST,LI,F,NE;2)     300  STR#;               6000 N !
PARAMETER G(TEST,BE,B,C;0)     300  STR#;               6000 N !
PARAMETER G(TEST,BE,B,C;1)     300  STR#;               6000 N !
PARAMETER G(TEST,BE,B,C;2)     300  STR#;               6000 N !
PARAMETER G(TEST,BE,B,N;0)     300  STR#;               6000 N !
PARAMETER G(TEST,BE,B,N;1)     300  STR#;               6000 N !
PARAMETER G(TEST,BE,B,N;2)     300  STR#;               6000 N !
PARAMETER G(TEST,BE,B,O;0)     300  STR#;               6000 N !
PARAMETER G(TEST,BE,B,O;1)     300  STR#;               6000 N !
PARAMETER G(TEST,BE,B,O;2)     300  STR#;               6000 N !
PARAMETER G(TEST,BE,B,F;0)     300  STR#;               6000 N !
PARAMETER G(TEST,BE,B,F;1)     300  STR#;               6000 N !
PARAMETER G(TEST,BE,B,F;2)     300  STR#;               6000 N !
PARAMETER G(TEST,BE,B,NE;0)     300  STR#;               6000 N !
PARAMETER G(TEST,BE,B,NE;1)     300  STR#;               6000 N !
PARAMETER G(TEST,BE,B,NE;2)     300  STR#;               6000 N !
PARAMETER G(TEST,BE,C,N;0)     300  STR#;               6000 N !
PARAMETER G(TEST,BE,C,N;1)     300  STR#;               6000 N !
PARAMETER G(TEST,BE,C,N;2)     300  STR#;               6000 N !
PARAMETER G(TEST,BE,C,O;0)     300  STR#;               6000 N !
PARAMETER G(TEST,BE,C,O;1)     300  STR#;               6000 N !
PARAMETER G(TEST,BE,C,O;2)     300  STR#;               6000 N !
PARAMETER G(TEST,BE,C,F;0)     300  STR#;               6000 N !
PARAMETER G(TEST,BE,C,F;1)     300  STR#;               6000 N !
PARAMETER G(TEST,BE,C,F;2)     300  STR#;               6000 N !
PARAMETER G(TEST,BE,C,NE;0)     300  STR#;               6000 N !
PARAMETER G(TEST,BE,C,NE;1)     300  STR#;               6000 N !
PARAMETER G(TEST,BE,C,NE;2)     300  STR#;               6000 N !
PARAMETER G(TEST,BE,N,O;0)     300  STR#;               6000 N !
PARAMETER G(TEST,BE,N,O;1)     300  STR#;               6000 N !
PARAMETER G(TEST,BE,N,O;2)     300  STR#;               6000 N !
PARAMETER G(TEST,BE,N,F;0)     300  STR#;               6000 N !
PARAMETER G(TEST,BE,N,F;1)     300  STR#;               6000 N !
PARAMETER G(TEST,BE,N,F;2)     300  STR#;               6000 N !
PARAMETER G(TEST,BE,N,NE;0)     300  STR#;               6000 N !
PARAMETER G(TEST,BE,N,NE;1)     300  STR#;               6000 N !
PARAMETER G(TEST,BE,N,NE;2)     300  STR#;               6000 N !
PARAMETER G(TEST,BE,O,F;0)     300  STR#;               6000 N !
PARAMETER G(TEST,BE,O,F;1)     300  STR#;               6000 N !
PARAMETER G(TEST,BE,O,F;2)     300  STR#;               6000 N !
PARAMETER G(TEST,BE,O,NE;0)     300  STR#;               6000 N !
PARAMETER G(TEST,BE,O,NE;1)     300  STR#;               6000 N !
PARAMETER G(TEST,BE,O,NE;2)     300  STR#;               6000 N !
PARAMETER G(TEST,BE,F,NE;0)     300  STR#;               6000 N !
PARAMETER G(TEST,BE,F,NE;1)     300  STR#;               6000 N !
PARAMETER G(TEST,BE,F,NE;2)     300  STR#;               6000 N !
PARAMETER G(TEST,B,C,N;0)     300  STR#;               6000 N !
PARAMETER G(TEST,B,C,N;1)     300  STR#;               6000 N !
PARAMETER G(TEST,B,C,N;2)     300  STR#;               6000 N !
PARAMETER G(TEST,B,C,O;0)     300  STR#;               6000 N !
PARAMETER G(TEST,B,C,O;1)     300  STR#;               6000 N !
PARAMETER G(TEST,B,C,O;2)     300  STR#;               6000 N !
PARAMETER G(TEST,B,C,F;0)     300  STR#;               6000 N !
PARAMETER G(TEST,B,C,F;1)     300  STR#;               6000 N !
PARAMETER G(TEST,B,C,F;2)     300  STR#;               6000 N !
PARAMETER G(TEST,B,C,NE;0)     300  STR#;               6000 N !
PARAMETER G(TEST,B,C,NE;1)     300  STR#;               6000 N !
PARAMETER G(TEST,B,C,NE;2)     300  STR#;               6000 N !
PARAMETER G(TEST,B,N,O;0)     300  STR#;               6000 N !
PARAMETER G(TEST,B,N,O;1)     300  STR#;               6000 N !
PARAMETER G(TEST,B,N,O;2)     300  STR#;               6000 N !
PARAMETER G(TEST,B,N,F;0)     300  STR#;               6000 N !
PARAMETER G(TEST,B,N,F;1)     300  STR#;               6000 N !
PARAMETER G(TEST,B,N,F;2)     300  STR#;               6000 N !
PARAMETER G(TEST,B,N,NE;0)     300  STR#;               6000 N !
PARAMETER G(TEST,B,N,NE;1)     300  STR#;               6000 N !
PARAMETER G(TEST,B,N,NE;2)     300  STR#;               6000 N !
PARAMETER G(TEST,B,O,F;0)     300  STR#;               6000 N !
PARAMETER G(TEST,B,O,F;1)     300  STR#;               6000 N !
PARAMETER G(TEST,B,O,F;2)     300  STR#;               6000 N !
PARAMETER G(TEST,B,O,NE;0)     300  STR#;               6000 N !
PARAMETER G(TEST,B,O,NE;1)     300  STR#;               6000 N !
PARAMETER G(TEST,B,O,NE;2)     300  STR#;               6000 N !
PARAMETER G(TEST,B,F,NE;0)     300  STR#;               6000 N !
PARAMETER G(TEST,B,F,NE;1)     300  STR#;               6000 N !
PARAMETER G(TEST,B,F,NE;2)     300  STR#;               6000 N !
PARAMETER G(TEST,C,N,O;0)     300  STR#;               6000 N !
PARAMETER G(TEST,C,N,O;1)     300  STR#;               6000 N !
PARAMETER G(TEST,C,N,O;2)     300  STR#;               6000 N !
PARAMETER G(TEST,C,N,F;0)     300  STR#;               6000 N !
PARAMETER G(TEST,C,N,F;1)     300  STR#;               6000 N !
PARAMETER G(TEST,C,N,F;2)     300  STR#;               6000 N !
PARAMETER G(TEST,C,N,NE;0)     300  STR#;               6000 N !
PARAMETER G(TEST,C,N,NE;1)     300  STR#;               6000 N !
PARAMETER G(TEST,C,N,NE;2)     300  STR#;               6000 N !
PARAMETER G(TEST,C,O,F;0)     300  STR#;               6000 N !
PARAMETER G(TEST,C,O,F;1)     300  STR#;               6000 N !
PARAMETER G(TEST,C,O,F;2)     300  STR#;               6000 N !
PARAMETER G(TEST,C,O,NE;0)     300  STR#;               6000 N !
PARAMETER G(TEST,C,O,NE;1)     300  STR#;               6000 N !
PARAMETER G(TEST,C,O,NE;2)     300  STR#;               6000 N !
PARAMETER G(TEST,C,F,NE;0)     300  STR#;               6000 N !
PARAMETER G(TEST,C,F,NE;1)     300  STR#;               6000 N !
PARAMETER G(TEST,C,F,NE;2)     300  STR#;               6000 N !
PARAMETER G(TEST,N,O,F;0)     300  STR#;               6000 N !
PARAMETER G(TEST,N,O,F;1)     300  STR#;               6000 N !
PARAMETER G(TEST,N,O,F;2)     300  STR#;               6000 N !
PARAMETER G(TEST,N,O,NE;0)     300  STR#;               6000 N !
PARAMETER G(TEST,N,O,NE;1)     300  STR#;               6000 N !
PARAMETER G(TEST,N,O,NE;2)     300  STR#;               6000 N !
PARAMETER G(TEST,N,F,NE;0)     300  STR#;               6000 N !
PARAMETER G(TEST,N,F,NE;1)     300  STR#;               6000 N !
PARAMETER G(TEST,N,F,NE;2)     300  STR#;               6000 N !
PARAMETER G(TEST,O,F,NE;0)     300  STR#;               6000 N !
PARAMETER G(TEST,O,F,NE;1)     300  STR#;               6000 N !
PARAMETER G(TEST,O,F,NE;2)     300  STR#;               6000 N !
"""

ROSE_DBF = Database(ROSE_TEST_STRING)
ALFE_DBF = Database('examples/alfe_sei.TDB')

# ROSE DIAGRAM TESTS
# This will fail until the equilibrium engine is switched from Newton-Raphson
@SkipTest
def test_rose_nine():
    "Nine-component rose diagram point equilibrium calculation."
    my_phases_rose = ['TEST']
    comps = ['H', 'HE', 'LI', 'BE', 'B', 'C', 'N', 'O', 'F']
    conds = dict({v.T: 1000})
    for comp in comps[:-1]:
        conds[v.X(comp)] = 1.0/float(len(comps))
    eqx = equilibrium(ROSE_DBF, comps, my_phases_rose, conds)
    assert_allclose(eqx.GM.values, -5.8351e3)

# OTHER TESTS
def test_eq_binary():
    "Binary phase diagram point equilibrium calculation with magnetism."
    my_phases = ['LIQUID', 'FCC_A1', 'HCP_A3', 'AL5FE2',
                 'AL2FE', 'AL13FE4', 'AL5FE4']
    comps = ['AL', 'FE', 'VA']
    conds = {v.T: 1400, v.X('AL'): 0.55}
    eqx = equilibrium(ALFE_DBF, comps, my_phases, conds)
    assert_allclose(eqx.GM.values, -9.608807e4)

def test_eq_single_phase():
    "Equilibrium energy should be the same as for a single phase with no miscibility gaps."
    res = calculate(ALFE_DBF, ['AL', 'FE'], 'LIQUID', T=[1400, 2500],
                    points={'LIQUID': [[0.1, 0.9], [0.2, 0.8], [0.3, 0.7],
                                       [0.4, 0.6], [0.5, 0.5], [0.6, 0.4],
                                       [0.7, 0.3], [0.8, 0.2]]})
    eq = equilibrium(ALFE_DBF, ['AL', 'FE'], 'LIQUID',
                     {v.T: [1400, 2500], v.X('AL'): [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]})
    assert_allclose(eq.GM, res.GM, atol=1e-4)

if __name__ == '__main__':
    import nose
    nose.run(defaultTest=__name__)
