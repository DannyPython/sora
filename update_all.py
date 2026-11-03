"""Refresh all three data files, then check them.

    python update_all.py              # update SORA, SOFR and USD/SGD, then run check_data.py
    python update_all.py --dry-run    # fetch + validate only, write nothing

Exit code 1 if any updater or the final data check fails. Keys: see .env.example.
"""
import subprocess
import sys
from pathlib import Path

root = Path(__file__).resolve().parent
scripts = ["update_sora", "update_sofr", "update_fx"]
extra = [a for a in sys.argv[1:] if a == "--dry-run"]

failed = []
for name in scripts:
    print(f"--- {name} ---")
    result = subprocess.run([sys.executable, f"updaters/{name}.py", *extra], cwd=root)
    if result.returncode != 0:
        failed.append(name)

print("--- check_data ---")
if subprocess.run([sys.executable, "check_data.py"], cwd=root).returncode != 0:
    failed.append("check_data")

print("\nDone." if not failed else f"\nFailed: {', '.join(failed)}")
sys.exit(1 if failed else 0)
