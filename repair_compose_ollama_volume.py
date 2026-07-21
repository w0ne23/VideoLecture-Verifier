#!/usr/bin/env python3
# Repair docker-compose.yml after an Ollama volume entry was accidentally inserted
# under a service-level volumes block such as services.db.volumes.
#
# Run from repository root:
#   python repair_compose_ollama_volume.py

from __future__ import annotations

from pathlib import Path
import sys


TARGET = Path("docker-compose.yml")


def indent_of(line: str) -> int:
    return len(line) - len(line.lstrip(" "))


def remove_bad_service_ollama_volume(lines: list[str]) -> tuple[list[str], int]:
    remove = [False] * len(lines)
    removed = 0

    for i, line in enumerate(lines):
        if line.strip() != "volumes:":
            continue

        vol_indent = indent_of(line)

        # root-level volumes is valid and should not be cleaned here.
        if vol_indent == 0:
            continue

        j = i + 1
        while j < len(lines):
            child = lines[j]
            if child.strip() and indent_of(child) <= vol_indent:
                break

            if child.strip() == "ollama:":
                bad_indent = indent_of(child)
                remove[j] = True
                removed += 1

                k = j + 1
                while k < len(lines):
                    if lines[k].strip() and indent_of(lines[k]) <= bad_indent:
                        break
                    remove[k] = True
                    k += 1
                j = k
                continue

            j += 1

    return [line for idx, line in enumerate(lines) if not remove[idx]], removed


def find_root_block(lines: list[str], name: str) -> tuple[int, int] | None:
    start = None
    for i, line in enumerate(lines):
        if indent_of(line) == 0 and line.strip() == f"{name}:":
            start = i
            break

    if start is None:
        return None

    end = len(lines)
    for j in range(start + 1, len(lines)):
        if lines[j].strip() and indent_of(lines[j]) == 0:
            end = j
            break
    return start, end


def ensure_root_ollama_volume(lines: list[str]) -> tuple[list[str], bool]:
    found = find_root_block(lines, "volumes")
    if found is None:
        if lines and lines[-1].strip():
            lines.append("")
        lines.extend(["volumes:", "  ollama:"])
        return lines, True

    start, end = found
    for i in range(start + 1, end):
        if indent_of(lines[i]) == 2 and lines[i].strip() == "ollama:":
            return lines, False

    lines.insert(start + 1, "  ollama:")
    return lines, True


def main() -> int:
    if not TARGET.exists():
        print(f"ERROR: {TARGET} not found. Run from repository root.", file=sys.stderr)
        return 1

    original = TARGET.read_text(encoding="utf-8")
    lines = original.splitlines()

    lines, removed = remove_bad_service_ollama_volume(lines)
    lines, added_root_volume = ensure_root_ollama_volume(lines)

    fixed = "\n".join(lines) + ("\n" if original.endswith("\n") else "")

    if fixed == original:
        print("No changes needed.")
        return 0

    backup = TARGET.with_suffix(TARGET.suffix + ".bak_repair_ollama_volume")
    backup.write_text(original, encoding="utf-8")
    TARGET.write_text(fixed, encoding="utf-8")

    print(f"Updated: {TARGET}")
    print(f"Backup : {backup}")
    print(f"Removed bad service-level ollama volume entries: {removed}")
    print(f"Added root-level volumes.ollama: {added_root_volume}")
    print()
    print("Check:")
    print("  docker compose config >/tmp/compose.check && echo OK")
    print("  grep -n \"^[[:space:]]*volumes:\\|^[[:space:]]*ollama:\" docker-compose.yml")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
