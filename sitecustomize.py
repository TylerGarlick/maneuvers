"""Project site customization: ensure `src/` is on sys.path when running locally.

This helps running modules and small code snippets in development without installing the package.
"""
import os
import sys

ROOT = os.path.abspath(os.path.dirname(__file__))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    # put src at front so it shadows a globally installed package of the same name
    sys.path.insert(0, SRC)
