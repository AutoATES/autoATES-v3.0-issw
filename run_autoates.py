#!/usr/bin/env python3
"""Run autoATES v3.0 (ISSW 2026 defaults).

    python run_autoates.py                  # uses autoates/comAutoATES/autoATESCfg.ini
    python run_autoates.py path/to/my.ini
"""
from autoates.runScripts.runComAutoATES import main

if __name__ == "__main__":
    main()
