from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

try:
    from Cython.Distutils import build_ext
except ImportError:
    use_cython = False
else:
    use_cython = True
    from Cython.Build import cythonize

cmdclass = {}
ext_modules = []

if use_cython:
    ext_modules += [
        Extension("networkqit.matfun", [ "utils/matfun.pyx" ]),
    ]
    cmdclass.update({ 'build_ext': build_ext })
else:
    ext_modules += [
        Extension("networkqit.matfun", [ "utils/matfun.c" ]),
    ]


matfuncython = [Extension('matfun', ["utils/cpp/pymatfun.cpp","utils/matfun.pyx"], language="c++", include_dirs=["utils", "utils/cpp/", "utils/eigen"] + eigency.get_includes(include_eigen=False) )]

setup(
    name='networkqit',
    cmdclass = cmdclass,
    version='0.1',
    description='''A package for fitting spectral entropies of complex networks''',
    include_package_data = True,
    install_requires=['numpy, matplotlib, eigency, cython, drawnow, seaborn'],
    url='carlonicolini.github.io/networkqit',
    author='Carlo Nicolini',
    author_email='carlo.nicolini@iit.it',
    ext_modules=[ext_modules, cythonize(matfuncython,include_path=[numpy.get_include()])]
    )
