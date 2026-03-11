import os
from dotenv import load_dotenv


def _read_env(name: str, default: str = "") -> str:
    value = os.getenv(name, default)
    if value is None:
        return default
    return str(value).strip()


BASE_DIR = os.path.dirname(__file__)
load_dotenv(os.path.join(BASE_DIR, ".env"))


def _read_keys() -> list[str]:
    raw = _read_env("GEMINI_KEYS", "")
    return [key.strip() for key in raw.split(",") if key.strip()]


# Core Phase-1.5 configuration only.
GEMINI_KEYS = _read_keys()
GEMINI_MODEL = _read_env("GEMINI_MODEL", "gemini-1.5-flash")
DEFAULT_DB_PATH = _read_env("MOSAHAI_DB_PATH", os.path.join(BASE_DIR, "shorts_master.json"))
USAGE_TRACKER_DB_PATH = _read_env("MOSAHAI_USAGE_DB_PATH", os.path.join(BASE_DIR, "usage_tracker.db"))
SCRIPT_VAULT_DB_PATH = _read_env(
    "MOSAHAI_SCRIPT_VAULT_PATH",
    os.path.join(BASE_DIR, "NicheSpecific_ScriptEvolution_History_Bank"),
)
ENABLE_VAULT_STORAGE = _read_env("MOSAHAI_ENABLE_VAULT_STORAGE", "true").lower() in {
    "1",
    "true",
    "yes",
    "on",
}
RSS_BASE_URL_TEMPLATE = "https://news.google.com/rss/search?q={topic}&hl={lang}&gl=IN&ceid=IN:{lang}"

PHASE15_NICHES = {
    "AI": "English",
    "Geopolitics": "English",
    "Hyderabad": "English",
}
