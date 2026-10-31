"""The one location the analysis scripts need: WORKSPACE, the root under which the raw data, the intermediate arrays and
the outputs live, with the same relative layout as on the machines where the analyses ran (README, "Layout").
It is read from config.env at the repository root (copy config.example.env), or from the environment variable of the
same name, which takes precedence; SAMEWELL_CONFIG may point to another file."""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _read(path):
    out = {}
    if os.path.exists(path):
        for line in open(path, encoding="utf-8"):
            line = line.split("#", 1)[0].strip()
            if "=" in line:
                k, v = line.split("=", 1)
                out[k.strip()] = v.strip().strip('"').strip("'")
    return out


def get(key):
    if os.environ.get(key):
        return os.environ[key]
    cfg = _read(os.environ.get("SAMEWELL_CONFIG") or os.path.join(ROOT, "config.env"))
    if key not in cfg:
        raise KeyError("%s is not set: copy config.example.env to config.env and fill it in, or export %s" % (key, key))
    return os.path.expandvars(cfg[key])


try:
    WORKSPACE = get("WORKSPACE").rstrip("/")
except KeyError:
    WORKSPACE = None      # scripts that only draw figures from data/ do not need it


class _Missing(str):
    def __add__(self, other):
        raise KeyError("WORKSPACE is not set: copy config.example.env to config.env and fill it in, or export WORKSPACE")


if WORKSPACE is None:
    WORKSPACE = _Missing("")
