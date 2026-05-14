# Configuration file for Sphinx documentation builder.
# See https://www.sphinx-doc.org/en/master/usage/configuration.html

from __future__ import annotations

import sys
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# -- Project information -----------------------------------------------
project = "pso-segmentation"
copyright = "2026, Léo Colin"
author = "Léo Colin"
release = "0.1.0"

# -- General configuration -----------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "myst_parser",
]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
master_doc = "index"

# -- Autodoc configuration -----------------------------------------------
autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "undoc-members": False,
    "inherited-members": False,
    "show-inheritance": True,
}

autosummary_generate = True
autosummary_generate_overwrite = False

# -- Napoleon configuration (Google/NumPy docstring support) -----------
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_references = False
napoleon_use_ivar = True
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_type_annotations = True
napoleon_attr_annotations = True

# -- HTML output configuration ----------------------------------------
html_theme = "sphinx_rtd_theme"
html_theme_options = {
    "collapse_navigation": False,
}

html_static_path = []
html_title = "PSO Segmentation - Robust Particle Swarm Optimization Segmentation"

# -- Intersphinx configuration ----------------------------------------
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "pandas": ("https://pandas.pydata.org/docs", None),
}

# -- MyST parser configuration ----------------------------------------
myst_enable_extensions = [
    "colon_fence",
    "smartquotes",
    "linkify",
]

# Suppress warnings
suppress_warnings = ["myst.header"]
