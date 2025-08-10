# paths.py
from pathlib import Path
import os

def _find_root(start: Path | None = None) -> Path:
    """從目前檔案往上找，遇到任一標記檔就當作專案根目錄"""
    p = (start or Path(__file__).resolve()).parent
    markers = {"config.yaml", ".git", "pyproject.toml", "requirements.txt"}
    while p != p.parent:
        if any((p / m).exists() for m in markers):
            return p
        p = p.parent
    return Path(__file__).resolve().parent  # fallback

ROOT = Path(os.getenv("PROJECT_ROOT", _find_root()))
DATA_DIR = ROOT / "data"
CACHE_DIR = ROOT / ".cache"
LOG_DIR = ROOT / "logs"
CONFIG_FILE = Path(os.getenv("APP_CONFIG", ROOT / "config.yaml"))

def rel(*parts: str) -> Path:
    """相對於專案根目錄組路徑"""
    return ROOT.joinpath(*parts)

def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p
