"""
This module tests the functionality of the TDB file parser.
"""
import nose.tools
from pycalphad import Database
from sympy import SympifyError

@nose.tools.raises(SympifyError)
def test_tdb_popen_exploit():
    "Prevent execution of arbitrary code using Popen."
    tdb_exploit_string = \
        """
        PARAMETER G(L12_FCC,AL,CR,NI:NI;0)
        298.15 [].__class__.__base__.__subclasses__()[158]('/bin/ls'); 6000 N !
        """
    Database(tdb_exploit_string)
