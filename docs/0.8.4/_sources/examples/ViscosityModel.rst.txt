
Custom Models in pycalphad: Viscosity
=====================================

Viscosity Model Background
--------------------------

We are going to take a CALPHAD-based property model from the literature
and use it to predict the viscosity of Al-Cu-Zr liquids.

For a binary alloy liquid under small undercooling, Gąsior suggested an
entropy model of the form

.. math:: \eta = (\sum_i x_i \eta_i ) (1 - 2\frac{S_{ex}}{R})

where :math:`\eta_i` is the viscosity of the element :math:`i`,
:math:`x_i` is the mole fraction, :math:`S_{ex}` is the excess entropy,
and :math:`R` is the gas constant.

For more details on this model, see

1. M.E. Trybula, T. Gancarz, W. Gąsior, *Density, surface tension and
   viscosity of liquid binary Al-Zn and ternary Al-Li-Zn alloys*, Fluid
   Phase Equilibria 421 (2016) 39-48,
   `doi:10.1016/j.fluid.2016.03.013 <http://dx.doi.org/10.1016/j.fluid.2016.03.013>`__.

2. Władysław Gąsior, *Viscosity modeling of binary alloys: Comparative
   studies*, Calphad 44 (2014) 119-128,
   `doi:10.1016/j.calphad.2013.10.007 <http://dx.doi.org/10.1016/j.calphad.2013.10.007>`__.

3. Chenyang Zhou, Cuiping Guo, Changrong Li, Zhenmin Du, *Thermodynamic
   assessment of the phase equilibria and prediction of glass-forming
   ability of the Al–Cu–Zr system*, Journal of Non-Crystalline Solids
   461 (2017) 47-60,
   `doi:10.1016/j.jnoncrysol.2016.09.031 <https://doi.org/10.1016/j.jnoncrysol.2016.09.031>`__.

.. code:: ipython3

    from pycalphad import Database

TDB Parameters
--------------

We can calculate the excess entropy of the liquid using the Al-Cu-Zr
thermodynamic database from Zhou et al.

We add three new parameters to describe the viscosity (in Pa-s) of the
pure elements Al, Cu, and Zr:

::

      $ Viscosity test parameters
      PARAMETER ETA(LIQUID,AL;0) 2.98150E+02  +0.000281*EXP(12300/(8.3145*T));   6.00000E+03   
     N REF:0 !
      PARAMETER ETA(LIQUID,CU;0) 2.98150E+02  +0.000657*EXP(21500/(8.3145*T));   6.00000E+03   
     N REF:0 !
     PARAMETER ETA(LIQUID,ZR;0) 2.98150E+02  +4.74E-3 - 4.97E-6*(T-2128) ;   6.00000E+03   
       N REF:0 !

Great! However, if we try to load the database now, we will get an
error. This is because ``ETA`` parameters are not supported by default
in pycalphad, so we need to tell pycalphad’s TDB parser that “ETA”
should be on the list of supported parameter types.

.. code:: ipython3

    dbf = Database('alcuzr-viscosity.tdb')


.. parsed-literal::

    Failed while parsing:     PARAMETER ETA(LIQUID,AL;0) 2.98150E+02 +0.000281*EXP(12300/(8.3145*T)); 6.00000E+03 N REF:0 
    Tokens: None


::


    ---------------------------------------------------------------------------

    ParseException                            Traceback (most recent call last)

    <ipython-input-2-d711f4128286> in <module>
    ----> 1 dbf = Database('alcuzr-viscosity.tdb')
    

    ~/Projects/pycalphad/pycalphad/io/database.py in __new__(cls, *args)
        117             elif fname.find('\n') == -1:
        118                 # Single-line string; it's probably a filename
    --> 119                 return cls.from_file(fname, fmt=fmt)
        120             else:
        121                 # Newlines found: probably a full database string


    ~/Projects/pycalphad/pycalphad/io/database.py in from_file(fname, fmt)
        211         try:
        212             dbf = Database()
    --> 213             format_registry[fmt.lower()].read(dbf, fd)
        214         finally:
        215             # Close file descriptors created in this routine


    ~/Projects/pycalphad/pycalphad/io/tdb.py in read_tdb(dbf, fd)
        950         tokens = None
        951         try:
    --> 952             tokens = grammar.parseString(command)
        953             _TDB_PROCESSOR[tokens[0]](dbf, *tokens[1:])
        954         except:


    ~/anaconda3/envs/calphad-dev-2/lib/python3.7/site-packages/pyparsing.py in parseString(self, instring, parseAll)
       1943             else:
       1944                 # catch and re-raise exception from here, clears out pyparsing internal stack trace
    -> 1945                 raise exc
       1946         else:
       1947             return tokens


    ~/anaconda3/envs/calphad-dev-2/lib/python3.7/site-packages/pyparsing.py in parseString(self, instring, parseAll)
       1933             instring = instring.expandtabs()
       1934         try:
    -> 1935             loc, tokens = self._parse(instring, 0)
       1936             if parseAll:
       1937                 loc = self.preParse(instring, loc)


    ~/anaconda3/envs/calphad-dev-2/lib/python3.7/site-packages/pyparsing.py in _parseCache(self, instring, loc, doActions, callPreParse)
       1834                 ParserElement.packrat_cache_stats[MISS] += 1
       1835                 try:
    -> 1836                     value = self._parseNoCache(instring, loc, doActions, callPreParse)
       1837                 except ParseBaseException as pe:
       1838                     # cache a copy of the exception, without the traceback


    ~/anaconda3/envs/calphad-dev-2/lib/python3.7/site-packages/pyparsing.py in _parseNoCache(self, instring, loc, doActions, callPreParse)
       1673             if self.mayIndexError or preloc >= len(instring):
       1674                 try:
    -> 1675                     loc, tokens = self.parseImpl(instring, preloc, doActions)
       1676                 except IndexError:
       1677                     raise ParseException(instring, len(instring), self.errmsg, self)


    ~/anaconda3/envs/calphad-dev-2/lib/python3.7/site-packages/pyparsing.py in parseImpl(self, instring, loc, doActions)
       4248             if maxException is not None:
       4249                 maxException.msg = self.errmsg
    -> 4250                 raise maxException
       4251             else:
       4252                 raise ParseException(instring, loc, "no defined alternatives to match", self)


    ~/anaconda3/envs/calphad-dev-2/lib/python3.7/site-packages/pyparsing.py in parseImpl(self, instring, loc, doActions)
       4233         for e in self.exprs:
       4234             try:
    -> 4235                 ret = e._parse(instring, loc, doActions)
       4236                 return ret
       4237             except ParseException as err:


    ~/anaconda3/envs/calphad-dev-2/lib/python3.7/site-packages/pyparsing.py in _parseCache(self, instring, loc, doActions, callPreParse)
       1834                 ParserElement.packrat_cache_stats[MISS] += 1
       1835                 try:
    -> 1836                     value = self._parseNoCache(instring, loc, doActions, callPreParse)
       1837                 except ParseBaseException as pe:
       1838                     # cache a copy of the exception, without the traceback


    ~/anaconda3/envs/calphad-dev-2/lib/python3.7/site-packages/pyparsing.py in _parseNoCache(self, instring, loc, doActions, callPreParse)
       1673             if self.mayIndexError or preloc >= len(instring):
       1674                 try:
    -> 1675                     loc, tokens = self.parseImpl(instring, preloc, doActions)
       1676                 except IndexError:
       1677                     raise ParseException(instring, len(instring), self.errmsg, self)


    ~/anaconda3/envs/calphad-dev-2/lib/python3.7/site-packages/pyparsing.py in parseImpl(self, instring, loc, doActions)
       4048                     raise ParseSyntaxException(instring, len(instring), self.errmsg, self)
       4049             else:
    -> 4050                 loc, exprtokens = e._parse(instring, loc, doActions)
       4051             if exprtokens or exprtokens.haskeys():
       4052                 resultlist += exprtokens


    ~/anaconda3/envs/calphad-dev-2/lib/python3.7/site-packages/pyparsing.py in _parseCache(self, instring, loc, doActions, callPreParse)
       1834                 ParserElement.packrat_cache_stats[MISS] += 1
       1835                 try:
    -> 1836                     value = self._parseNoCache(instring, loc, doActions, callPreParse)
       1837                 except ParseBaseException as pe:
       1838                     # cache a copy of the exception, without the traceback


    ~/anaconda3/envs/calphad-dev-2/lib/python3.7/site-packages/pyparsing.py in _parseNoCache(self, instring, loc, doActions, callPreParse)
       1673             if self.mayIndexError or preloc >= len(instring):
       1674                 try:
    -> 1675                     loc, tokens = self.parseImpl(instring, preloc, doActions)
       1676                 except IndexError:
       1677                     raise ParseException(instring, len(instring), self.errmsg, self)


    ~/anaconda3/envs/calphad-dev-2/lib/python3.7/site-packages/pyparsing.py in parseImpl(self, instring, loc, doActions)
       4248             if maxException is not None:
       4249                 maxException.msg = self.errmsg
    -> 4250                 raise maxException
       4251             else:
       4252                 raise ParseException(instring, loc, "no defined alternatives to match", self)


    ~/anaconda3/envs/calphad-dev-2/lib/python3.7/site-packages/pyparsing.py in parseImpl(self, instring, loc, doActions)
       4233         for e in self.exprs:
       4234             try:
    -> 4235                 ret = e._parse(instring, loc, doActions)
       4236                 return ret
       4237             except ParseException as err:


    ~/anaconda3/envs/calphad-dev-2/lib/python3.7/site-packages/pyparsing.py in _parseCache(self, instring, loc, doActions, callPreParse)
       1834                 ParserElement.packrat_cache_stats[MISS] += 1
       1835                 try:
    -> 1836                     value = self._parseNoCache(instring, loc, doActions, callPreParse)
       1837                 except ParseBaseException as pe:
       1838                     # cache a copy of the exception, without the traceback


    ~/anaconda3/envs/calphad-dev-2/lib/python3.7/site-packages/pyparsing.py in _parseNoCache(self, instring, loc, doActions, callPreParse)
       1677                     raise ParseException(instring, len(instring), self.errmsg, self)
       1678             else:
    -> 1679                 loc, tokens = self.parseImpl(instring, preloc, doActions)
       1680 
       1681         tokens = self.postParse(instring, loc, tokens)


    ~/Projects/pycalphad/pycalphad/io/tdb.py in parseImpl(self, instring, loc, doActions)
        186         except ValueError:
        187             pass
    --> 188         raise ParseException(instring, loc, self.errmsg, self)
        189 
        190 def _tdb_grammar(): #pylint: disable=R0914


    ParseException: Expected {{"ELEMENT" W:(ABCD...) W:(ABCD...) Re:('[-+]?([0-9]+\\.(?!([0-9]|[eE])))|([0-9]*\\.?[0-9]+([eE][-+]?[0-9]+)?)') Re:('[-+]?([0-9]+\\.(?!([0-9]|[eE])))|([0-9]*\\.?[0-9]+([eE][-+]?[0-9]+)?)') Re:('[-+]?([0-9]+\\.(?!([0-9]|[eE])))|([0-9]*\\.?[0-9]+([eE][-+]?[0-9]+)?)') LineEnd} | {"SPECIES" W:(ABCD...) [Suppress:("%")] Group:({{W:(ABCD...) [Re:('[-+]?([0-9]+\\.(?!([0-9]|[eE])))|([0-9]*\\.?[0-9]+([eE][-+]?[0-9]+)?)')]}}...) [{Suppress:("/") W:(+-01...)}] LineEnd} | {"TYPE_DEFINITION" Suppress:(<SP><TAB><CR><LF>) !W:( !) SkipTo:(LineEnd)} | {"FUNCTION" W:(ABCD...) {{Re:('[-+]?([0-9]+\\.(?!([0-9]|[eE])))|([0-9]*\\.?[0-9]+([eE][-+]?[0-9]+)?)') | [","]...} {{SkipTo:(";") Suppress:(";") [Suppress:(",")]... [Re:('[-+]?([0-9]+\\.(?!([0-9]|[eE])))|([0-9]*\\.?[0-9]+([eE][-+]?[0-9]+)?)')] Suppress:({W:(YNyn) | <SP><TAB><CR><LF>})}}...}} | {"ASSESSED_SYSTEMS" SkipTo:(LineEnd)} | {"DEFINE_SYSTEM_DEFAULT" SkipTo:(LineEnd)} | {"DEFAULT_COMMAND" SkipTo:(LineEnd)} | {"DATABASE_INFO" SkipTo:(LineEnd)} | {"VERSION_DATE" SkipTo:(LineEnd)} | {"REFERENCE_FILE" SkipTo:(LineEnd)} | {"ADD_REFERENCES" SkipTo:(LineEnd)} | {"LIST_OF_REFERENCES" SkipTo:(LineEnd)} | {"TEMPERATURE_LIMITS" SkipTo:(LineEnd)} | {"PHASE" W:(ABCD...) Suppress:(<SP><TAB><CR><LF>) !W:( !) Suppress:(<SP><TAB><CR><LF>) Suppress:(W:(0123...)) Group:({Re:('[-+]?([0-9]+\\.(?!([0-9]|[eE])))|([0-9]*\\.?[0-9]+([eE][-+]?[0-9]+)?)')}...) Suppress:(SkipTo:(LineEnd))} | {"CONSTITUENT" W:(ABCD...) Suppress:(<SP><TAB><CR><LF>) Suppress:(":") Group:(Group:({{[Suppress:(",")] {W:(ABCD...) [Suppress:("%")]}}}...) [: Group:({{[Suppress:(",")] {W:(ABCD...) [Suppress:("%")]}}}...)]...) Suppress:(":") LineEnd} | {"PARAMETER" {"BMAGN" | "DF" | "DQ" | "G" | "GD" | "L" | "MF" | "MQ" | "NT" | "TC" | "THETA" | "V0" | "VS"} Suppress:("(") W:(ABCD...) [{Suppress:("&") W:(ABCD...)}] Suppress:(",") Group:(Group:({{[Suppress:(",")] {W:(ABCD...) [Suppress:("%")]}}}...) [: Group:({{[Suppress:(",")] {W:(ABCD...) [Suppress:("%")]}}}...)]...) [{Suppress:(";") W:(0123...)}] Suppress:(")") {{Re:('[-+]?([0-9]+\\.(?!([0-9]|[eE])))|([0-9]*\\.?[0-9]+([eE][-+]?[0-9]+)?)') | [","]...} {{SkipTo:(";") Suppress:(";") [Suppress:(",")]... [Re:('[-+]?([0-9]+\\.(?!([0-9]|[eE])))|([0-9]*\\.?[0-9]+([eE][-+]?[0-9]+)?)')] Suppress:({W:(YNyn) | <SP><TAB><CR><LF>})}}...}}}, found '('  (at char 17), (line:1, col:18)


Adding the ``ETA`` parameter to the TDB parser
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    import pycalphad.io.tdb_keywords
    pycalphad.io.tdb_keywords.TDB_PARAM_TYPES.append('ETA')

Now the database will load:

.. code:: ipython3

    dbf = Database('alcuzr-viscosity.tdb')

Writing the Custom Viscosity Model
----------------------------------

Now that we have our ``ETA`` parameters in the database, we need to
write a ``Model`` class to tell pycalphad how to compute viscosity. All
custom models are subclasses of the pycalphad ``Model`` class.

When the ``ViscosityModel`` is constructed, the ``build_phase`` method
is run and we need to construct the viscosity model after doing all the
other initialization using a new method ``build_viscosity``. The
implementation of ``build_viscosity`` needs to do four things: 1. Query
the Database for all the ``ETA`` parameters 2. Compute their weighted
sum 3. Compute the excess entropy of the liquid 4. Plug all the values
into the Gąsior equation and return the result

Since the ``build_phase`` method sets the attribute ``viscosity`` to the
``ViscosityModel``, we can access the property using ``viscosity`` as
the output in pycalphad caluclations.

.. code:: ipython3

    from tinydb import where
    import sympy
    from pycalphad import Model, variables as v
    
    class ViscosityModel(Model):
        def build_phase(self, dbe):
            super(ViscosityModel, self).build_phase(dbe)
            self.viscosity = self.build_viscosity(dbe)
    
        def build_viscosity(self, dbe):
            if self.phase_name != 'LIQUID':
                raise ValueError('Viscosity is only defined for LIQUID phase')
            phase = dbe.phases[self.phase_name]
            param_search = dbe.search
            # STEP 1
            eta_param_query = (
                (where('phase_name') == phase.name) & \
                (where('parameter_type') == 'ETA') & \
                (where('constituent_array').test(self._array_validity))
            )
            # STEP 2
            eta = self.redlich_kister_sum(phase, param_search, eta_param_query)
            # STEP 3
            excess_energy = self.GM - self.models['ref'] - self.models['idmix']
            #liquid_mod = Model(dbe, self.components, self.phase_name)
            ## we only want the excess contributions to the entropy
            #del liquid_mod.models['ref']
            #del liquid_mod.models['idmix']
            excess_entropy = -excess_energy.diff(v.T)
            ks = 2
            # STEP 4
            result = eta * (1 - ks * excess_entropy / v.R)
            self.eta = eta
            return result

Performing Calculations
-----------------------

Now we can create an instance of ``ViscosityModel`` for the liquid phase
using the ``Database`` object we created earlier. We can verify this
model has a ``viscosity`` attribute containing a symbolic expression for
the viscosity.

.. code:: ipython3

    mod = ViscosityModel(dbf, ['CU', 'ZR'], 'LIQUID')
    print(mod.viscosity)


.. parsed-literal::

    (1 + 0.240543628600637*(LIQUID0CU*LIQUID0ZR*(75.3798 - 9.6125*log(T))*(LIQUID0CU - LIQUID0ZR) + LIQUID0CU*LIQUID0ZR*(105.895 - 13.6488*log(T))*(LIQUID0CU - LIQUID0ZR)**3 + LIQUID0CU*LIQUID0ZR*(392.8485 - 51.3121*log(T)) + LIQUID0CU*LIQUID0ZR*(LIQUID0CU - LIQUID0ZR)**2*(36.8512*log(T) - 270.5305))/(1.0*LIQUID0CU + 1.0*LIQUID0ZR))*(0.000657*LIQUID0CU*exp(2585.84400745685/T) + LIQUID0ZR*(0.01531616 - 4.97e-6*T))


Finally we calculate and plot the viscosity.

.. code:: ipython3

    %matplotlib inline
    import matplotlib.pyplot as plt
    import numpy as np
    from pycalphad import calculate
    
    mod = ViscosityModel(dbf, ['CU', 'ZR'], 'LIQUID')
    
    temp = 2100
    # NOTICE: we need to tell pycalphad about our model for this phase
    models = {'LIQUID': mod}
    res = calculate(dbf, ['CU', 'ZR'], 'LIQUID', P=101325, T=temp, model=models, output='viscosity') 
    
    fig = plt.figure(figsize=(6,6))
    ax = fig.gca()
    ax.scatter(res.X.sel(component='ZR'), 1000 * res.viscosity.values)
    ax.set_xlabel('X(ZR)')
    ax.set_ylabel('Viscosity (mPa-s)')
    ax.set_xlim((0,1))
    ax.set_title('Viscosity at {}K'.format(temp));



.. image:: ViscosityModel_files/ViscosityModel_14_0.png


We repeat the calculation for Al-Cu.

.. code:: ipython3

    %matplotlib inline
    import matplotlib.pyplot as plt
    import numpy as np
    from pycalphad import calculate
    
    temp = 1300
    models = {'LIQUID': ViscosityModel}  # we can also use Model class
    res = calculate(dbf, ['CU', 'AL'], 'LIQUID', P=101325, T=temp, model=models, output='viscosity')
    
    fig = plt.figure(figsize=(6,6))
    ax = fig.gca()
    ax.scatter(res.X.sel(component='CU'), 1000 * res.viscosity.values)
    ax.set_xlabel('X(CU)')
    ax.set_ylabel('Viscosity (mPa-s)')
    ax.set_xlim((0,1))
    ax.set_title('Viscosity at {}K'.format(temp));



.. image:: ViscosityModel_files/ViscosityModel_16_0.png


