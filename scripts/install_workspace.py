"""Install the reviewed scaffold into a NEW requested workspace; never merge or overwrite."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    source = Path(__file__).resolve().parents[1]
    destination = args.destination.expanduser().resolve()
    if destination.exists():
        raise SystemExit(f"Refusing to overwrite or merge existing destination: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    excluded = shutil.ignore_patterns(".venv", ".git", "tmp", "__pycache__", ".pytest_cache", "*.egg-info")
    shutil.copytree(source, destination, ignore=excluded)
    manifest = {}
    for path in destination.rglob("*"):
        if path.is_file():
            manifest[str(path.relative_to(destination))] = hashlib.sha256(path.read_bytes()).hexdigest()
    (destination / "INSTALL_MANIFEST.json").write_text(json.dumps({
        "source_workspace": str(source), "destination": str(destination),
        "policy": "new destination only; original work preserved; source run provenance retained",
        "installed_file_sha256": manifest}, indent=2) + "\n")
    subprocess.run([sys._base_executable, "-m", "venv", str(destination / ".venv")], check=True)
    python = destination / ".venv/bin/python"
    subprocess.run([str(python), "-m", "pip", "install", "-r", "requirements-lock.txt"], cwd=destination, check=True)
    subprocess.run([str(python), "-m", "pip", "install", "--no-deps", "--no-build-isolation", "-e", "."], cwd=destination, check=True)
    subprocess.run(["git", "init", "-b", "main"], cwd=destination, check=True)
    subprocess.run([str(python), "-m", "pytest", "-q"], cwd=destination, check=True)
    subprocess.run([str(python), "-m", "dns_exfiltration", "doctor"], cwd=destination, check=True)
    # Available outside the sandbox during the authorized install; never fabricate RAM.
    hardware = {"python": str(python), "destination": str(destination)}
    for key in ["hw.memsize", "hw.ncpu", "hw.model", "machdep.cpu.brand_string"]:
        result = subprocess.run(["sysctl", "-n", key], capture_output=True, text=True)
        hardware[key] = result.stdout.strip() if result.returncode == 0 else None
    (destination / "results/hardware.json").write_text(json.dumps(hardware, indent=2) + "\n")
    print(f"Installed and verified: {destination}")


if __name__ == "__main__":
    main()
