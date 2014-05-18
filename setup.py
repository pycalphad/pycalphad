from setuptools import setup, find_packages, Extension
import os, sys

major_version = sys.version_info[0]
minor_version = sys.version_info[1]

libtdbcpp =  Extension('calphad.io.libtdbcpp', # C++ extension for TDB parsing
                  [os.path.join(dirpath, f)
    for dirpath, dirnames, files in os.walk('calphad/io/libtdb/source')
    for f in files if f.endswith('.cpp')],
                  include_dirs=['calphad/io'],
                  library_dirs=[''],
                  libraries=['boost_python-'+str(major_version)+'.'+str(minor_version), # find python ver
			     'boost_log',
			     'boost_log_setup',
			     'boost_thread',
			     'boost_system',
			    ],
		  define_macros=[
			  ('BOOST_ALL_DYN_LINK',None)
			  ],
                  extra_compile_args=['-g','-std=c++0x'] # -g is debugging info for gcc
                 )

setup(
    name='Calphad',
    version='0.0.1',
    author='Richard Otis',
    author_email='richard.otis@outlook.com',
    packages=['calphad','calphad.io'],
    license='',
    install_requires=[''],
    classifiers=['Development Status :: 3 - Alpha'],
    ext_modules=[libtdbcpp],
)