# Corpora

Three corpora ship with this release, all synthetic, totalling ~16 MB. Nothing
here contains real personal data, and no network access is needed to run any
experiment.

Run `python scripts/verify_data.py` to check them against
[`MANIFEST.json`](MANIFEST.json), which pins the SHA-256, document count and
label-defect count of every file. A silent corpus change is the one fault that
would invalidate every number in the release without failing a single test, so
the check runs first in `make smoke` and again at Docker build time.

## `synpii/` -- SynPII-F (5 files, ~2 MB)

Template-generated financial documents: KYC onboarding summaries, wire memos
and loan notes, over the nine-type finance taxonomy, with gold spans recorded
**at generation time** rather than annotated afterwards -- which is what makes
exact candidate-miss counts knowable here and nowhere else.

| property | value |
|---|---|
| seeds | 7, 13, 21, 42, 77 (one file each) |
| documents per seed | 341 = 300 unique + 41 injected duplicates |
| duplicate injection | ~12%, half exact and half near (routing footers, whitespace perturbation) |
| gold spans per seed | 2,990-3,002 |
| licence | Apache-2.0, released with this code |

The injected duplicates are the point, not padding: they give the entity graph
recurring entities to link, and `split_calibration_test` deliberately routes
every duplicate into the **test** stream so propagation is exercised exactly
where the certificate is evaluated.

**Provenance caveat.** SynPII-F shares an author with the rule engine, so its
in-domain numbers inherit an advantage the third-party corpora do not grant.
Treat Gretel and Nemotron as the primary evidence; SynPII-F is the controlled
setting where the unobservable quantity is observable.

## `external/gretel_finance.jsonl` (~8.8 MB)

English subset of
[`gretelai/synthetic_pii_finance_multilingual`](https://huggingface.co/datasets/gretelai/synthetic_pii_finance_multilingual),
normalised onto the canonical taxonomy.

| property | value |
|---|---|
| documents | 5,000 |
| gold spans kept | 14,169 |
| known defect | 40 spans point past the end of their document and are **dropped**, not clamped |
| licence | see the upstream dataset card |

One pathologically long document is truncated to 8,000 characters at load
time, uniformly for every system, so no engine gains or loses from it.

## `external/nemotron_finance.jsonl` (~5.6 MB)

The `domain=Finance` rows of
[`nvidia/Nemotron-PII`](https://huggingface.co/datasets/nvidia/Nemotron-PII),
normalised onto the canonical taxonomy.

| property | value |
|---|---|
| documents | 3,990 |
| gold spans kept | 6,654 |
| known defect | 4 labels differ from the document only in case (cosmetic; offsets are correct) |
| licence | see the upstream dataset card |

## Label mapping, and what it costs

Native labels are mapped onto the nine canonical types **by name**. Anything
that does not map -- API keys, passwords, IP addresses -- is **dropped rather
than force-mapped**, because folding an API key into `ACCOUNT_NUMBER` would
make the taxonomy report a number about a population it is not measuring.

`aegis.data.load_corpus` returns the kept/dropped counts alongside the
documents, and every experiment records them, so the size of that decision is
visible in the artefact rather than buried in a preprocessing script.

## Adding your own corpus

One JSONL object per document:

```json
{
  "id": "doc-0001",
  "text": "Borrower Melanie Riley, IBAN DE89370400440532013000.",
  "gold": [
    {"start": 9,  "end": 22, "type": "PERSON", "text": "Melanie Riley"},
    {"start": 29, "end": 51, "type": "IBAN",   "text": "DE89370400440532013000"}
  ]
}
```

`gold` may be omitted or empty -- the shift audit (`aegis audit`) and the
label-free Layer IV diagnostics need no labels at all. Point `AEGIS_DATA_ROOT`
at your directory and extend `aegis.data.loaders` with a loader, or reuse
`load_jsonl` directly.
