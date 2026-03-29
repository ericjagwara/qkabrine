# Configuration file for the Sphinx documentation builder.

import os
import sys

# So Sphinx can find the package source
sys.path.insert(0, os.path.abspath('..'))

# ── Project information ───────────────────────────────────────
project = 'Qkabrine AutoML'
copyright = '2026, Eric Jagwara — Solid Elf Labs'
author = 'Eric Jagwara'
release = '2.1.1'
version = '2.1.1'

# ── Extensions ───────────────────────────────────────────────
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.intersphinx',
    'sphinx_autodoc_typehints',
    'sphinx_copybutton',
    'myst_parser',
]

# ── Theme ─────────────────────────────────────────────────────
html_theme = 'furo'
html_title = 'Qkabrine AutoML'
html_theme_options = {
    "sidebar_hide_name": False,
    "navigation_with_keys": True,
    "light_css_variables": {
        "color-brand-primary": "#6d4aff",
        "color-brand-content": "#6d4aff",
    },
    "dark_css_variables": {
        "color-brand-primary": "#a78bfa",
        "color-brand-content": "#a78bfa",
    },
    "footer_icons": [
        {
            "name": "GitHub",
            "url": "https://github.com/ericjagwara/qkabrine",
            "html": """
                <svg stroke="currentColor" fill="currentColor" viewBox="0 0 16 16"
                     height="1em" width="1em">
                  <path fill-rule="evenodd"
                    d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38
                       0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13
                       -.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87
                       2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95
                       0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21
                       2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04
                       2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82
                       2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48
                       0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0 0 16 8
                       c0-4.42-3.58-8-8-8z"/>
                </svg>
            """,
            "class": "",
        },
    ],
}

# ── Napoleon settings (for Google-style docstrings) ──────────
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_use_param = True
napoleon_use_rtype = True

# ── Autodoc settings ─────────────────────────────────────────
autodoc_member_order = 'bysource'
autodoc_typehints = 'description'
add_module_names = False

# ── Intersphinx ──────────────────────────────────────────────
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable', None),
    'sklearn': ('https://scikit-learn.org/stable', None),
}

# ── General ──────────────────────────────────────────────────
templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']
html_static_path = ['_static']
source_suffix = {'.rst': 'restructuredtext', '.md': 'markdown'}
master_doc = 'index'
