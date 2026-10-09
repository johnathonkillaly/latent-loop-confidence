"""Regenerate each public analysis in a separate clean checkout.

Historical preservation hashes include report manifests, so running multiple
report stages in one writable checkout is intentionally not supported.
Use an environment installed with .[test] for an accelerator-free check.
"""

import os
import subprocess
import sys
import tempfile
from pathlib import Path


COMMANDS = [
    ["latent_loop.cli", "report", "--config", "configs/pilot.yaml"],
    ["latent_loop.experiment03.run", "report"],
    ["latent_loop.experiment04a.pipeline", "report"],
    ["latent_loop.experiment04b.pipeline", "report"],
]


def main():
    root = Path(__file__).resolve().parents[1]
    for index, command in enumerate(COMMANDS):
        with tempfile.TemporaryDirectory(prefix="latent-loop-report-") as folder:
            subprocess.run(["git", "clone", "--shared", "--quiet", str(root), folder], check=True)
            env = dict(os.environ, PYTHONPATH=str(Path(folder) / "src"))
            subprocess.run([sys.executable, "-m", *command], cwd=folder, env=env, check=True)
            print(f"Verified report {index + 1}/{len(COMMANDS)}: {command[0]}", flush=True)


if __name__ == "__main__":
    main()
