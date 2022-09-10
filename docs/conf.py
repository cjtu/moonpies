# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
from moonpies import __version__
sys.path.insert(0, os.path.abspath('..'))


# -- Project information -----------------------------------------------------

project = 'MoonPIES'
copyright = '2021, Christian J. Tai Udovicic'
author = 'Katelyn R. Frizzell, Kristen M. Luchsinger, Alissa Madera, Tyler G. Paladino and Christian J. Tai Udovicic'

# The full version, including alpha/beta/rc tags
release = __version__


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "numpydoc",
    "sphinx.ext.todo",
    "sphinx.ext.viewcode",
    "sphinx.ext.autodoc",
    "sphinx_rtd_theme",
    'myst_parser'
    ]


# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# Point to index.rst
master_doc = 'index'

# Don't expand constants in doc params
autodoc_preserve_defaults = True

# Don't alphabetize functions
autodoc_member_order = "bysource"

# Latex options
latex_documents = [
    (master_doc, 'moonpiesapi.tex', project,
     author.replace(', ', '\\and ').replace(' and ', '\\and and '),
     'manual'),
]

# latex_elements = {
#     "papersize": "letterpaper",
#     "pointsize": "10pt",
#     "figure_align": "htbp",
#     "preamble": r"""
#         \usepackage{listings}
#         \lstset{ 
#             language=Python,                 % the language of the code
#             title=\lstname                   % show the filename of files included with \lstinputlisting; also try caption instead of title
#         }
#     """,
# }