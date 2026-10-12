#!/usr/bin/env python3
"""Refuses to commit anything that looks like a live credential."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

#: Anthropic keys, and the object-id form the console displays beside them. Both are
#: worth refusing: one authenticates, and the other identifies an account's credential.
PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("an Anthropic API key", re.compile(r"sk-ant-[A-Za-z0-9_\-]{20,}")),
    ("an Anthropic key id", re.compile(r"\bapikey_[A-Za-z0-9]{20,}\b")),
    ("an OpenAI key", re.compile(r"\bsk-[A-Za-z0-9]{32,}\b")),
    ("an AWS access key id", re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b")),
    ("a GitHub token", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{36,}\b")),
    ("a Slack token", re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b")),
    ("a Google API key", re.compile(r"\bAIza[0-9A-Za-z_\-]{35}\b")),
)

#: Skipped wholesale: this file states the patterns, and binary or vendored trees are
#: not where a pasted key lands.
SKIP_PARTS = frozenset({".git", ".venv", "__pycache__", "node_modules", "data"})
SKIP_SUFFIXES = frozenset({".pdf", ".png", ".jpg", ".jpeg", ".gif", ".ico", ".woff", ".woff2"})


def looks_like_real_material(candidate: str) -> bool:
    """Whether the matched text is plausibly a live secret rather than a placeholder.

    A real key's random part has upper case, lower case and digits. A hand-written stand-in
    (``sk-ant-must-not-leak``, ``sk-ant-xxxxx``, ``sk-ant-REPLACE-ME``) has at most one of
    the three, which makes the distinction cheap and reliable.
    """
    tail = re.split(r"[-_]", candidate, maxsplit=2)[-1]
    classes = (
        any(c.islower() for c in tail),
        any(c.isupper() for c in tail),
        any(c.isdigit() for c in tail),
    )
    return sum(classes) >= 2


def findings_in(path: Path) -> list[str]:
    if path.resolve() == Path(__file__).resolve():
        return []
    if path.suffix.lower() in SKIP_SUFFIXES or SKIP_PARTS & set(path.parts):
        return []
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return []

    hits: list[str] = []
    for label, pattern in PATTERNS:
        for match in pattern.finditer(text):
            if not looks_like_real_material(match.group(0)):
                continue
            line = text.count("\n", 0, match.start()) + 1
            shown = f"{match.group(0)[:12]}...{match.group(0)[-4:]}"
            hits.append(f"{path}:{line}: {label} ({shown})")
    return hits


def tracked_files() -> list[Path]:
    out = subprocess.run(
        ["git", "ls-files", "-z"], capture_output=True, text=True, check=False
    ).stdout
    return [Path(name) for name in out.split("\0") if name]


def main(argv: list[str]) -> int:
    paths = [Path(a) for a in argv] or tracked_files()
    hits = [hit for path in paths if path.is_file() for hit in findings_in(path)]
    if not hits:
        return 0
    print("Refusing to commit: this looks like a live credential.\n", file=sys.stderr)
    for hit in hits:
        print(f"  {hit}", file=sys.stderr)
    print(
        "\nIf it's real, it's already on this disk in plain text: rotate it at "
        "console.anthropic.com, then keep the replacement out of the repo. The key is "
        "read from the environment only, never from a file in the tree.\n"
        "If it's a placeholder, make it obviously one (lower case and hyphens, no "
        "digits) so it reads as a stand-in to a person as well as to this script.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
