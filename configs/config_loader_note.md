The CLI takes flags, not YAML, so these files are documentation-as-data: every
key maps one-to-one onto `aegis.pipeline.AegisConfig`, and a two-line helper
turns one into a config object without adding a YAML dependency to the core:

```python
from aegis.pipeline import AegisConfig

def load(path):
    """Minimal reader for the flat key: value files in configs/."""
    fields = {f for f in AegisConfig.__dataclass_fields__}
    out = {}
    for line in open(path, encoding="utf-8"):
        line = line.split("#", 1)[0].strip()
        if not line or ":" not in line:
            continue
        key, raw = (part.strip() for part in line.split(":", 1))
        if key not in fields or not raw:
            continue
        if raw in ("true", "false"):
            out[key] = raw == "true"
        else:
            try:
                out[key] = int(raw) if raw.isdigit() else float(raw)
            except ValueError:
                out[key] = raw
    return AegisConfig(**out)
```

We keep it out of the package on purpose: a config format is a dependency and
an API-surface commitment, and the CLI flags already cover every knob.
