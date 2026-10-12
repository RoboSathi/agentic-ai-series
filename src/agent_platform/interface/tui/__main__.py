"""`python -m agent_platform.interface.tui` entry point.

A dedicated `__main__` rather than running `app.py` as a module: doing
the latter imports the package first and then re-executes the module,
which Python warns about (`found in sys.modules after import`) and which
would give the app two module identities.
"""

from agent_platform.interface.tui.app import main

main()
