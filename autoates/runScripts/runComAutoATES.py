"""
runComAutoATES.py
Main executable script for autoATES v3.0
"""

from pathlib import Path
import sys
import logging

# Add repository root to path
script_dir = Path(__file__).resolve().parent
repo_root = script_dir.parent.parent
sys.path.insert(0, str(repo_root))

from autoates.comAutoATES.comAutoATES import runAutoATES, load_config


def main():
    """Entry point for the command line tool."""
    
    print("=" * 70)
    print("🚀 Starting autoATES v3.0")
    print("=" * 70)

    try:
        # === Config path: optional CLI arg, else the default template ===
        if len(sys.argv) > 1:
            config_path = Path(sys.argv[1]).expanduser().resolve()
        else:
            config_path = script_dir.parent / "comAutoATES" / "autoATESCfg.ini"

        print(f"Looking for config at: {config_path}")

        cfg = load_config(str(config_path))

        if not cfg.sections():
            print(f"❌ ERROR: Could not load config from {config_path}")
            print("Please make sure autoATESCfg.ini exists in autoates/comAutoATES/")
            sys.exit(1)

        print(f"✅ Loaded config sections: {cfg.sections()}")

        # Run the full workflow
        results = runAutoATES(config=cfg)

        working_dir = cfg.get("General", "working_dir", fallback="./Outputs")

        print("\n" + "=" * 70)
        print("✅ autoATES v3.0 finished successfully!")
        print(f"Output directory: {Path(working_dir).absolute()}")
        print("=" * 70)

    except Exception as e:
        logging.error(f"Error running autoATES: {e}", exc_info=True)
        print(f"\n❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()