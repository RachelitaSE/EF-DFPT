from pathlib import Path
import ef_dfpt

def get_repo_root() -> Path:
    return Path(ef_dfpt.__file__).resolve().parents[2]

def get_data_dir() -> Path:
    return get_repo_root() / "data"
