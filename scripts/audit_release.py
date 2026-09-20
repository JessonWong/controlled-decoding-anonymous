"""Audit tracked release files without printing potential secret values."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import re
import subprocess


ROOT = Path(__file__).resolve().parents[1]
SKIP = {".git", ".venv", "venv", "__pycache__", ".pytest_cache", "build", "dist"}
BINARY_SUFFIXES = {".pt", ".pth", ".bin", ".safetensors", ".ckpt", ".pyc", ".zip", ".pdf"}
PATTERNS = {
    "provider credential": re.compile(r"(?:sk-(?:or-v1-|proj-)?[A-Za-z0-9_-]{24,}|AIza[A-Za-z0-9_-]{25,}|hf_[A-Za-z0-9]{25,}|AKIA[A-Z0-9]{16})"),
    "private key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "personal absolute path": re.compile(r"/(?:home|Users|jet/home|ocean/projects|scratch)/[^\s\"'<>]+"),
    "email address": re.compile(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}"),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deny", action="append", default=[], help="Additional private identifier to reject; never printed.")
    parser.add_argument("--write-manifest", action="store_true", help="Write SHA-256 checksums after a successful audit.")
    args = parser.parse_args()
    problems = []
    files = []
    for path in sorted(ROOT.rglob("*")):
        relative = path.relative_to(ROOT)
        if any(part in SKIP or part.endswith(".egg-info") for part in relative.parts):
            continue
        if path.is_symlink():
            problems.append((str(relative), 0, "symlink"))
            continue
        if not path.is_file():
            continue
        files.append(path)
        if path.name == "api_key.py" or (path.name.startswith(".env") and path.name != ".env.example"):
            problems.append((str(relative), 0, "credential file"))
        if path.suffix in BINARY_SUFFIXES:
            problems.append((str(relative), 0, "binary artifact"))
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            problems.append((str(relative), 0, "non-text artifact"))
            continue
        for number, line in enumerate(text.splitlines(), 1):
            for label, pattern in PATTERNS.items():
                if pattern.search(line):
                    problems.append((str(relative), number, label))
            if any(value.casefold() in line.casefold() for value in args.deny):
                problems.append((str(relative), number, "private identifier"))
        if path.suffix == ".py":
            try:
                ast.parse(text, filename=str(relative), feature_version=(3, 10))
            except SyntaxError as exc:
                problems.append((str(relative), exc.lineno, "Python 3.10 syntax error"))
    if (ROOT / ".git").is_dir():
        remotes = subprocess.check_output(["git", "-C", str(ROOT), "remote"], text=True)
        if remotes.strip():
            problems.append((".git/config", 0, "remote configured; inspect its anonymity before distribution"))
    if problems:
        for path, line, label in problems:
            print(f"{path}:{line}: {label}")
        raise SystemExit(f"FAIL: {len(problems)} findings (values omitted).")
    manifest_path = ROOT / "MANIFEST.sha256"
    if args.write_manifest:
        lines = [f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(ROOT).as_posix()}"
                 for p in files if p != manifest_path]
        manifest_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    elif manifest_path.exists():
        recorded = {}
        for line in manifest_path.read_text().splitlines():
            digest, relative = line.split("  ", 1)
            recorded[relative] = digest
        actual = {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                  for p in files if p != manifest_path}
        if actual != recorded:
            raise SystemExit("FAIL: file list or checksums differ from MANIFEST.sha256.")
    print(json.dumps({"status": "pass", "files": len(files),
                      "checks": ["credential patterns", "personal paths", "private identifiers",
                                 "binary artifacts", "symlinks", "Python 3.10 syntax", "checksums"]}))


if __name__ == "__main__":
    main()

