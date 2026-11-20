# Sphinx configuration of the jVMC documentation.
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import shutil
import sys
from pathlib import Path

# Document the package of this repository, not an installed copy.
sys.path.insert(0, os.path.abspath("../.."))

# Sphinx only reads files inside this folder, so copy the example notebooks of the repository
# into ``examples/`` (ignored by git) at every build.
_here = Path(__file__).parent
(_here / "examples").mkdir(exist_ok=True)
for _nb in (_here / "../../examples").glob("*.ipynb"):
    shutil.copy2(_nb, _here / "examples" / _nb.name)

from jVMC.version import __version__  # noqa: E402

project = "jVMC"
author = "Markus Schmitt and the jVMC contributors"
copyright = "2020-2026, Markus Schmitt and the jVMC contributors"
release = __version__
version = ".".join(__version__.split(".")[:2])

extensions = [
    "sphinx.ext.autodoc",   # API pages from docstrings
    "sphinx.ext.mathjax",   # :math: in docstrings
    "sphinx.ext.viewcode",  # "[source]" links
    "myst_nb",              # Markdown pages and Jupyter notebooks
    "sphinx_rtd_theme",
]

# Optional dependencies that are not needed to import the package for the docs.
autodoc_mock_imports = ["jaxmg"]
autodoc_member_order = "bysource"

# Notebooks are shown with their saved outputs; they are not run during the build.
nb_execution_mode = "off"
myst_enable_extensions = ["dollarmath", "amsmath"]

templates_path = ["_templates"]
exclude_patterns = ["_build"]

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
