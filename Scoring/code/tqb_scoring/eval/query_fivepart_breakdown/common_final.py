from __future__ import annotations

import shutil
from pathlib import Path
from typing import Iterable, Mapping


def versioned_name(filename: str, version_tag: str) -> str:
    path = Path(str(filename))
    suffix = path.suffix
    stem = path.stem if suffix else path.name
    tagged = f"{stem}__{version_tag}"
    return f"{tagged}{suffix}" if suffix else tagged


def _copy_file(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        if src.resolve() == dst.resolve():
            return
    except FileNotFoundError:
        pass
    shutil.copy2(src, dst)


def sync_final_outputs(
    final_dir: Path,
    files: Iterable[Path],
    must_do_aliases: Mapping[str, Path] | None = None,
    *,
    version_tag: str | None = None,
    copy_plain_files: bool = True,
) -> None:
    final_dir.mkdir(parents=True, exist_ok=True)
    must_do_dir = final_dir / "must_do"
    must_do_dir.mkdir(parents=True, exist_ok=True)
    version_dir = final_dir / str(version_tag) if version_tag else None
    version_must_do_dir = (version_dir / "must_do") if version_dir is not None else None
    if version_dir is not None:
        version_dir.mkdir(parents=True, exist_ok=True)
    if version_must_do_dir is not None:
        version_must_do_dir.mkdir(parents=True, exist_ok=True)

    for src in files:
        if copy_plain_files:
            _copy_file(src, final_dir / src.name)
        if version_tag:
            tagged_name = versioned_name(src.name, str(version_tag))
            _copy_file(src, final_dir / tagged_name)
            if version_dir is not None:
                _copy_file(src, version_dir / tagged_name)

    for alias_name, src in (must_do_aliases or {}).items():
        if copy_plain_files:
            _copy_file(src, final_dir / alias_name)
            _copy_file(src, must_do_dir / alias_name)
        if version_tag:
            tagged_alias = versioned_name(alias_name, str(version_tag))
            _copy_file(src, final_dir / tagged_alias)
            _copy_file(src, must_do_dir / tagged_alias)
            if version_dir is not None:
                _copy_file(src, version_dir / tagged_alias)
            if version_must_do_dir is not None:
                _copy_file(src, version_must_do_dir / tagged_alias)


def render_final_readme(
    *,
    title: str,
    summary: str,
    primary_files: list[str],
    must_do_files: list[str],
    support_files: list[str] | None = None,
    notes: list[str] | None = None,
) -> str:
    lines = [
        f"# {title}",
        "",
        summary.strip(),
        "",
        "Primary paper-facing files:",
        "",
    ]
    lines.extend(f"- `{item}`" for item in primary_files)
    lines.extend(
        [
            "",
            "Must-do bundle (`must_do/`):",
            "",
        ]
    )
    lines.extend(f"- `must_do/{item}`" for item in must_do_files)
    if support_files:
        lines.extend(
            [
                "",
                "Support files:",
                "",
            ]
        )
        lines.extend(f"- `{item}`" for item in support_files)
    if notes:
        lines.extend(["", *notes])
    lines.append("")
    return "\n".join(lines)
