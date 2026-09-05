# Configuration files

Each YAML file is a complete, runnable operating point. They are plain
data: nothing here changes what the code *can* do, only what a particular run
declares.

The CLI takes **flags, not YAML** -- a config format is a dependency and an
API-surface commitment, and the flags already cover every knob. So these files
are documentation-as-data: every key maps one-to-one onto
`aegis.pipeline.AegisConfig`, and you either read the values off and pass them,

```bash
aegis run --corpus synpii --alpha 0.10 --gamma 0.10 \
          --engines rule,heuristic-ner,shape --select-by rho_star
```

or use the eight-line reader in
[`config_loader_note.md`](config_loader_note.md) to turn a file into an
`AegisConfig` directly.

| file | what it declares |
|---|---|
| `default.yaml` | the offline default: three pure-Python engines, `alpha = 0.10` |
| `aegis_s.yaml` | the strict point, `alpha = 0.01` -- often infeasible, by design |
| `aegis_b.yaml` | the balanced point, `alpha = 0.10` |
| `aegis_u.yaml` | the **uncertified** F1-maximal point; claims no guarantee |
| `transfer_zero_label.yaml` | calibrate on one corpus, deploy on another |
| `full_engines.yaml` | adds the optional GLiNER and LLM tiers |

## The two numbers to read

`alpha` is a **policy instrument**: it is chosen by whoever owns the risk, and
the escalation budget prices it. On checksum-anchored types a strict `alpha` is
nearly free; on `PERSON`, which has no arithmetic to appeal to, it is bought
with budget.

`rho_star` is **not a setting**. It does not appear in any config file because
it cannot be chosen -- it is a property of the calibration losses and the
selected operating point, computed and reported. Its only operational use is
the comparison `rho_hat <= rho_star`, which gates whether this certificate may
be applied to the documents arriving today.
