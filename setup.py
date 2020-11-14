# Always prefer setuptools over distutils
from setuptools import setup, find_packages

# To use a consistent encoding
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

# Arguments marked as "Required" below must be included for upload to PyPI.
# Fields marked as "Optional" may be commented out.

setup(
    name="networkqit",
    version="1.00.0",
    description="A package for fitting spectral entropies of complex networks",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/carlonicolini/networkqit",
    author="Carlo Nicolini",
    author_email="carlo.nicolini@iit.it",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Programming Language :: Python :: 3.8",
    ],
    keywords="networks complex graph theory entropy physics",
    packages=find_packages(
        exclude=["contrib", "dist", "docs", "examples", "tests"]
    ),
    install_requires=[
        "autograd==1.3",
        "numpy==1.19.4",
        "scipy==1.5.4",
        "networkx==2.5",
        "numdifftools==0.9.39",
        "matplotlib==3.3.3",
        "pandas==1.1.4",
        "bctpy==0.5.2",
        "sympy==1.6.2",
        "mpmath==1.1.0",
        "tqdm==4.51.0",
        "drawnow==0.72.5",
        "seaborn==0.11.0",
    ],

    extras_require={  # Optional
        "dev": [
            "check-manifest==0.4.5",
        ],
        "test": [
            "coverage==5.3",
            "pytest==6.1.2"
        ],
        "doc": [
            "sphinx",
            "sphinx_rtd_theme==0.5.0",
            "nb2plots==0.6"
        ]
    },

    package_data={  # Optional
        "sample": ["package_data.dat"],
    },

    project_urls={  # Optional
        'Bug Reports': 'https://github.com/carlonicolini/networkqit/issues',
        'Source': 'https://github.com/carlonicolini/networkqit/',
    },
)
