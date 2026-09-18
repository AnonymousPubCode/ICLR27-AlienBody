#!/usr/bin/env python3
"""Create/update the Hugging Face Space for AlienBody demo."""
from __future__ import annotations

from pathlib import Path

from huggingface_hub import HfApi, create_repo

ROOT = Path(__file__).resolve().parent
# Override with HF_SPACE_ID; keep a generic placeholder (no personal namespace).
REPO_ID = __import__("os").environ.get("HF_SPACE_ID", "AnonymousPubCode/AlienBody")


def main() -> None:
    api = HfApi()
    url = create_repo(
        repo_id=REPO_ID,
        repo_type="space",
        space_sdk="docker",
        private=False,
        exist_ok=True,
    )
    print("repo:", url)

    readme = (ROOT / "README_SPACE.md").read_text(encoding="utf-8")
    api.upload_file(
        path_or_fileobj=readme.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=REPO_ID,
        repo_type="space",
    )
    print("uploaded README.md")

    for src, dst in [
        ("Dockerfile", "Dockerfile"),
        ("requirements-space.txt", "requirements-space.txt"),
        (".gitignore", ".gitignore"),
        ("LICENSE", "LICENSE"),
    ]:
        p = ROOT / src
        if not p.exists():
            print("skip missing", src)
            continue
        api.upload_file(
            path_or_fileobj=str(p),
            path_in_repo=dst,
            repo_id=REPO_ID,
            repo_type="space",
        )
        print("uploaded", dst)

    for folder in ("alienbody", "web"):
        api.upload_folder(
            folder_path=str(ROOT / folder),
            path_in_repo=folder,
            repo_id=REPO_ID,
            repo_type="space",
            ignore_patterns=["**/__pycache__/**", "**/*.pyc", "**/.git/**"],
        )
        print("uploaded folder", folder)

    print("DONE")
    print(f"URL: https://huggingface.co/spaces/{REPO_ID}")


if __name__ == "__main__":
    main()
