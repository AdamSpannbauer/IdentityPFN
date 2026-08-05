from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def repository_path(*parts: str) -> Path:
    return REPOSITORY_ROOT.joinpath(*parts)
