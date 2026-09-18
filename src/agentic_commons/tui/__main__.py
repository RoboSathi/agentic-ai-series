"""`python -m agentic_commons.tui` entry point.

A dedicated `__main__` rather than running `app.py` as a module: doing
the latter imports the package first and then re-executes the module,
which Python warns about (`found in sys.modules after import`) and which
would give the app two module identities.
"""

from agentic_commons.tui.app import main

main()
