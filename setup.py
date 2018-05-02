from distutils.core import setup
from distutils.extension import Extension

cmdclass = {}
ext_modules = []


setup(
    name='networkqit',
    cmdclass = cmdclass,
    version='0.1',
    description='''A package for fitting spectral entropies of complex networks''',
    include_package_data = True,
    keywords = 'networks complex graph theory entropy physics',
    install_requires=['numpy, matplotlib, pandas, numdifftools'],
    packages=find_packages(exclude=['contrib','docs','tests*']),
    url='carlonicolini.github.io/networkqit',
    python_requires='>=3',
    author='Carlo Nicolini',
    author_email='carlo.nicolini@iit.it'
    )
