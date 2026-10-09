"""Conservative publication scan of every reachable Git blob, including gzip JSON.

Print locations and categories only, never matched values. This is a heuristic
review aid, not a proof that arbitrary secrets or proprietary text are absent.
"""

import gzip
import json
import re
import subprocess
from pathlib import Path


def git(*args):
    return subprocess.check_output(["git", *args])


def main():
    objects = git("rev-list", "--objects", "--all").decode().splitlines()
    names = dict(line.split(" ", 1) for line in objects if " " in line)
    batch = subprocess.check_output(
                    ["git", "cat-file", "--batch-check=%(objectname) %(objecttype) %(objectsize)"],
                    input=("\n".join(line.split()[0] for line in objects) + "\n").encode())
    patterns = {
        "token": r"(?:gh[pousr]_[A-Za-z0-9]{25,}|github_pat_[A-Za-z0-9_]{30,}|sk-[A-Za-z0-9_-]{20,}|AKIA[A-Z0-9]{16})",
        "private_key": r"-----BEGIN (?:RSA |OPENSSH |EC |DSA )?PRIVATE KEY-----",
        "private_path": r"(?:/Users/[^/\s\"<>]+/|/home/[^/\s\"<>]+/|[A-Z]:\\\\Users\\\\)",
        "internal_name": r"\b(?:xtium|boothgpt|couplergpt)\b|\.ghe\.|\.internal\b|https?://localhost",
        "private_network": r"\b(?:10\.\d{1,3}\.\d{1,3}\.\d{1,3}|192\.168\.\d{1,3}\.\d{1,3}|172\.(?:1[6-9]|2\d|3[01])\.\d{1,3}\.\d{1,3})\b",
        "credential_assignment": r"(?i)(?:api_key|password|access_token|secret_key)\s*[=:]\s*[\"'][^\"'\s]{8,}[\"']",
        "email": r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}",
    }
    issues, sizes = [], []
    blobs = 0
    gzip_blobs = 0
    for line in batch.decode().splitlines():
        oid, kind, size = line.split()
        if kind != "blob":
            continue
        blobs += 1
        name = names.get(oid, "<historical blob>")
        if int(size) > 10 * 1024**2:
            sizes.append({"path": name, "bytes": int(size)})
        if re.search(r"(?:\.safetensors|\.npz|\.pt|\.pth|\.pem|\.env)$", name):
            issues.append({"path": name, "category": "sensitive_artifact_extension", "blob": oid})
        if name == "scripts/audit_publication.py":
            # Scan rules contain intentional example patterns; reviewed manually.
            continue
        raw = git("cat-file", "blob", oid)
        if raw.startswith(b"\x1f\x8b"):
            raw = gzip.decompress(raw)
            gzip_blobs += 1
        if raw.startswith(b"\x89PNG"):
            # PNG bytes do not contain repository text; figures are regenerated
            # separately from measured JSON and visually reviewed.
            continue
        text = raw.decode("utf-8", errors="replace")
        text = re.sub(r"data:image/[^;]+;base64,[A-Za-z0-9+/=]+", "<image>", text)
        for category, pattern in patterns.items():
            for match in re.finditer(pattern, text):
                issues.append({"path": name, "blob": oid, "category": category,
                               "line": text.count("\n", 0, match.start()) + 1})
    commits = git("log", "--all", "--format=%H%x00%an%x00%ae%x00%cn%x00%ce").decode().splitlines()
    report = {"reachable_blobs": blobs, "gzip_blobs_decompressed": gzip_blobs,
              "commits": len(commits), "issues": issues, "files_over_10_mib": sizes,
              "author_email_domains": sorted({row.split("\x00")[2].split("@")[-1] for row in commits}),
              "working_tree_clean": not bool(git("status", "--porcelain")),
              "limitations": "Heuristic scan; manual provenance, names, logs and licensing review is also required."}
    destination = Path("artifacts/publication/history_scan.json")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
