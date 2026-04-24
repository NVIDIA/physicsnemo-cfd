# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import sys

sys.path.insert(0, os.path.abspath(".."))

project = "NVIDIA PhysicsNeMo CFD"
copyright = "2026, NVIDIA PhysicsNeMo Team"
author = "NVIDIA PhysicsNeMo Team"

try:
    from physicsnemo.cfd import __version__ as version

    release = version
except Exception:
    release = "0.0.2a0"
    version = "0.0"

autodoc_mock_imports = [
    "torch_geometric",
    "warp",
    "cuml",
    "cupy",
    "physicsnemo.models",
    "physicsnemo.distributed",
    "physicsnemo.nn",
    "physicsnemo.utils",
    "physicsnemo.datapipes",
    "physicsnemo.experimental",
]

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx.ext.todo",
    "myst_parser",
]

source_suffix = {".rst": "restructuredtext", ".md": "markdown"}
templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

autodoc_typehints = "description"
autodoc_member_order = "bysource"
autosummary_generate = True

napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_custom_sections = [
    ("Variable Shape", "notes"),
    ("Forward", "params_style"),
    ("Outputs", "returns_style"),
]

todo_include_todos = True

html_theme = "nvidia_sphinx_theme"
html_static_path = ["_static"]
html_css_files = [
    "css/custom.css",
]
html_theme_options = {
    "navbar_align": "content",
    "navbar_center": ["navbar-external-links"],
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/NVIDIA/physicsnemo-cfd",
            "icon": "fa-brands fa-square-github",
            "type": "fontawesome",
        }
    ],
    "external_links": [
        {
            "name": "PhysicsNeMo",
            "url": "https://github.com/NVIDIA/physicsnemo",
        },
    ],
}

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pyvista": ("https://docs.pyvista.org/version/stable/", None),
}
