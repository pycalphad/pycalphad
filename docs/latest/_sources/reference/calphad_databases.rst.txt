Calphad Databases
=================

Calphad database files contain model parameters for Gibbs energy and thermophysical properties of indvididual phases.
They are commonly stored and distributed in the following formats:

- Thermodynamic DataBase (``.tdb``) format (most common for open databases)
- ChemSage (``.dat``) format
- Commercial encrypted databases (not able to be read by PyCalphad)
- Emerging XML-based formats (`PyCalphad-XML <https://github.com/pycalphad/pycalphad-xml>`_, `XTDB <https://doi.org/10.1016/j.calphad.2025.102849>`_, `ThermML <https://github.com/flotang-gtt/ThermML>`_)

PyCalphad natively has ability parse databases saved with the file extension ``.tdb`` or ``.dat``.
It can be extended via a plugin system to understand other formats (see `PyCalphad-XML <https://github.com/pycalphad/pycalphad-xml>`_, which is deployed as a plugin that allows PyCalphad to understand the ``.xml`` extension).
These files are read by PyCalphad by importing and creating an instance of the ``Database`` class:

.. code-block:: python

   from pycalphad import Database
   db = Database("path/to/database.tdb")


Obtaining Calphad Databases
---------------------------

It is increasingly common (and required by some journals) for authors to publish their database files in Calphad assessment papers.
In the case they are not published, they can usually be constructed by hand from tables of model parameters in the assessment papers.

Many database files have been indexed and made availale through various community efforts (in no particular order):

- The `TDBDB <https://avdwgroup.engin.brown.edu/>`_ can be used to search for TDB included with publications
- `NIMS <https://cpddb.nims.go.jp>`_ (registration required) provides a collection of binary and ternary databases
- `NIST Materials Data <https://materialsdata.nist.gov/handle/11256/8>`_ repository contains databases deposited by authors
- `NIST solder database <https://www.metallurgy.nist.gov/phase/solder/solder.html>`_
- `Innomat AB <https://www.innomat.se/thermodynamic-databases/>`_ provide several multi-component databases
- `MatCalc <https://www.matcalc.at/index.php/databases/open-databases>`_ provide several multi-component databases
- `LibreCalphad <https://github.com/mfrichtl/librecalphad>`_ maintained by Matt Frichtl contains a multi-component steel database
- `PhaseDiagrams.org <https://phasediagrams.org>`_ provides a browsable set of binary assessments, mostly populated by the `SGTE binary collection by Bengt Hallstedt <https://doi.org/10.1016/j.calphad.2025.102833>`_.


If you would like to perform your own Calphad assessments using PyCalphad, check out `ESPEI <https://espei.org/>`_!