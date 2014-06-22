from setuptools import setup, find_packages, Extension
import os, sys

libtdbcpp_sources = [os.path.join(dirpath, f)
    for dirpath, dirnames, files in os.walk('calphad/cpp/libtdb/source')
    for f in files if f.endswith('.cpp')]

libgibbscpp_sources = [os.path.join(dirpath, f)
    for dirpath, dirnames, files in os.walk('calphad/cpp/libgibbs/source')
    for f in files if f.endswith('.cpp')]

libqhullcpp_sources = [os.path.join(dirpath, f)
    for dirpath, dirnames, files in os.walk('calphad/cpp/libqhullcpp')
    for f in files if (f.endswith('.c') or f.endswith('.cpp'))]

libcalphadcpp_sources = libtdbcpp_sources + libgibbscpp_sources + libqhullcpp_sources
libcalphadcpp_sources = libcalphadcpp_sources + ['calphad/cpp/python.cxx']

major_version = sys.version_info[0]
minor_version = sys.version_info[1]
boost_python_version = 'boost_python-'+str(major_version)+'.'+str(minor_version)
		  
libcalphadcpp =  Extension('calphad.libcalphadcpp',
                  libcalphadcpp_sources,
                  include_dirs=['calphad/cpp','calphad/cpp/libqhullcpp','calphad/cpp/libqhull'],
                  libraries=['ipopt', # ipopt (NLP solver)
			     boost_python_version,
			     'boost_timer',
			     'boost_chrono',
			     'boost_log',
			     'boost_log_setup',
			     'boost_thread',
			     'boost_system',
			    ],
		  define_macros=[
			  ('BOOST_ALL_DYN_LINK',None),
			  ('qh_QHpointer',1)
			  ],
                  extra_compile_args=['-g','-std=c++0x'], # -g is debug for gcc
                 )

setup(
    name='Calphad',
    version='0.0.1',
    author='Richard Otis',
    author_email='richard.otis@outlook.com',
    packages=['calphad','calphad.io', 'calphad.minimize', 'calphad.plot','calphad.plot.projections'],
    license='',
    install_requires=['numpy','matplotlib','pandas'],
    classifiers=['Development Status :: 3 - Alpha'],
    ext_modules=[libcalphadcpp],
)
