import subprocess
import sys
from pathlib import Path

root = Path(__file__).resolve().parent
scripts = ["update_sora", "update_sofr", "update_fx"]

failed = []
for name in scripts:
    print(f"--- {name} ---")
    result = subprocess.run([sys.executable, f"updaters/{name}.py"], cwd=root)
    if result.returncode != 0:
        failed.append(name)

print("\nDone." if not failed else f"\nFailed: {', '.join(failed)}")