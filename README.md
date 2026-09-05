# AEGIS

**Distribution-shift-certified risk control for PII redaction in document streams.**

Reference implementation for the paper *AEGIS: Distribution-Shift-Certified
Risk Control for PII Redaction in Financial Document Streams*.

A redaction pipeline should be able to say what it leaks, and to say when it
can no longer say. This repository is the second half of that sentence made
executable: it will calibrate a leakage bound, report the largest
distribution shift under which that bound provably survives, measure the shift
actually present, and **refuse to issue a certificate** when no operating point
supports the level you asked for.

```bash
git clone https://github.com/aegis-redaction/aegis-finance && cd aegis-finance
pip install -e .              # numpy is the only hard dependency
make smoke                    # ~2 min: full pipeline, all six experiments
```

---

## Contents

- [What this actually does](#what-this-actually-does)
- [Install](#install)
- [Five-minute tour](#five-minute-tour)
- [Reproducing the paper](#reproducing-the-paper)
- [Docker](#docker)
- [Repository layout](#repository-layout)
- [Where each theorem lives](#where-each-theorem-lives)
- [Data](#data)
- [Tests](#tests)
- [Honest differences from the paper](#honest-differences-from-the-paper)
- [Operating notes](#operating-notes)
- [Citing](#citing)

---

## What this actually does

Most redaction systems report accuracy. Accuracy does not answer the question
a regulator asks. A micro F1 of 0.92 is compatible with *any* leakage rate on
the subpopulation that matters, and a precision of 1.000 is what silent,
systematic leakage looks like from inside a dashboard.

AEGIS reports three numbers instead, and they are the only three worth reading:

| number | question it answers | needs labels? |
|---|---|---|
| realised `E[L]` against declared `alpha` | did the pipeline respect the bound it claimed? | yes |
| `rho_hat` against `rho_star` | does that bound even apply to today's documents? | **no** |
| `p_hat^U_y` per type | how much PII did *no engine propose*? | **no** |

Only the first is available from a standard evaluation. The second and third
are the point of the system.

The pipeline is five layers:

```
                 pooled calibration + deployment batch
                                 |
  LAYER I    heterogeneous engines -> span pool -> entity graph -> propagate
             (the MinHash index built for deduplication IS the edge set)
                                 |  propagated scores, exchangeability preserved
  LAYER II   document-level conformal risk control + Learn-then-Test
             (leakage AND over-masking, certified simultaneously)
                                 |
  LAYER III  weighted CRC under an estimated density ratio
             + chi^2-DRO  ->  certified radius rho_star
                                 |  <lambda, alpha, gamma, rho_star>  or  NO CERTIFICATE
  LAYER V    gate -> lazy-greedy escalation under budget -> HMAC surrogates -> ledger
                                 |  realised outcomes
  LAYER IV   e-processes + e-BH  |  Chao1 upper limit  |  rho_hat vs rho_star
                                 |
                     alarm  ->  recalibrate or halt
```

## Install

```bash
pip install -e .                              # core: numpy only
pip install -e ".[dev,figures]"               # + pytest, ruff, matplotlib
pip install -r requirements-optional.txt      # + GLiNER, Presidio (optional)
```

Python 3.10+. The core pipeline, every theorem's implementation and every
certificate in the paper run on **numpy alone** -- no torch, no transformers,
no network. That is deliberate: a reproduction that needs four gigabytes of
model weights and a working API key is not a reproduction.

## Five-minute tour

**Corpus statistics, including what the label mapping cost:**

```bash
aegis info --corpus gretel
```

**A shift audit with no labels at all** -- the question "may I deploy this
certificate here?" does not need a single annotation:

```bash
aegis audit --corpus synpii --target-corpus gretel
```

```json
{
  "rho_hat": "inf",
  "discriminator": { "auc": 1.0, "overlap": 0.0, "separable": true }
}
```

The discriminator separates SynPII-F from Gretel perfectly, so their supports
are effectively disjoint, the true `chi^2` divergence is unbounded, and **no
finite radius covers the deployment**. Note what a naive plug-in estimator
would have said here: with `g -> 0` on every source document the raw ratios
collapse and, after mean-normalisation, `rho_hat -> 0` -- "no shift", in the
case of total shift. The separability guard exists for exactly that.

**Calibrate and deploy in domain:**

```bash
aegis run --corpus synpii --alpha 0.10 --out results/tour.json
```

**Ask for something infeasible, and watch it refuse:**

```bash
aegis run --corpus synpii --alpha 0.000001
```

```json
{ "certificate": { "empty": true, "verdict": "no-certificate" } }
```

That is Theorem 2's empty rejection set doing its job. The stream is routed to
review; no threshold is emitted that the data cannot support.

**Redact a file:**

```bash
echo "Borrower Melanie Riley, IBAN DE89370400440532013000, SSN 406-44-8691." \
  | aegis redact --corpus synpii --alpha 0.10
```

Surrogates are keyed HMACs of the *normalised* value, so the same account maps
to the same surrogate stream-wide with no mapping table, and checksummed
identifiers come back Luhn- and mod-97-valid so downstream format validators
keep passing.

**From Python:**

```python
from aegis.data import load_synpii, split_calibration_test
from aegis.pipeline import AegisConfig, AegisPipeline

docs, _ = load_synpii(seed=7)
cal, dep = split_calibration_test(docs, cal_frac=0.4, seed=0)

pipeline = AegisPipeline(AegisConfig(alpha=0.10, gamma=0.10))
cert, report = pipeline.run(cal, dep)

print(cert.verdict)                 # 'valid' | 'void' | 'no-certificate'
print(cert.rho_star, cert.rho_hat)  # certified radius vs measured shift
print(report.realised_leakage)      # must respect cert.alpha
print(report.dark_matter)           # label-free per-type miss estimates
print(report.ledger["chain_valid"]) # hash-chained audit trail
```

## Reproducing the paper

```bash
make smoke        # ~2 min   small subsamples; exercises every code path
make reproduce    # ~30 min  full corpora, then regenerates paper assets
make assets       # LaTeX tables + figure data from whatever is in results/
```

Six drivers, each writing one JSON artefact into `results/` with full
provenance (version, commit, platform, timestamp):

| driver | what it produces |
|---|---|
| `exp01_main_table.py` | Table I: AEGIS operating points vs baselines, all corpora |
| `exp02_shift_radius.py` | Figure 2 + the shift audit + the alpha-feasibility frontier |
| `exp03_ablation.py` | leave-one-out ablation + Prop. 1(a)'s recall-vs-multiplicity curve |
| `exp04_monitor.py` | monitor validity, then a fair matched-false-alarm delay race |
| `exp05_chao.py` | validates the label-free bound *where labels exist* |
| `exp06_escalation.py` | cost frontier + greedy vs brute-force optimum |

`make assets` then writes `results/paper_assets/` -- `table_main.tex`,
`table_ablation.tex`, `fig_shift_radius.dat`, `fig_escalation.dat`,
`summary.md`. No number in the paper is transcribed by hand.

### The result to look at first

```
=== audit :: synpii-heldout
[aegis]   rho_hat=0.02649 vs rho_star_hat=0.1045 -> VALID (no target labels used)
[aegis]   realised transfer leakage=0.0249 -> verdict CORRECT

=== audit :: gretel
[aegis]   rho_hat=inf vs rho_star_hat=0.1045 -> VOID (no target labels used)
[aegis]   realised transfer leakage=0.3472 -> verdict CORRECT

=== audit :: nemotron
[aegis]   rho_hat=inf vs rho_star_hat=0.1045 -> VOID (no target labels used)
[aegis]   realised transfer leakage=0.3774 -> verdict CORRECT
```

Three verdicts issued before any target label was read; three verdicts
correct. The framework was right about itself in both directions -- it
predicted its own success in domain and its own failure out of domain. That
asymmetry is not obtainable from accuracy at any level.

## Docker

The image pins Python 3.11.9 and numpy 1.26.4, fixes `PYTHONHASHSEED` and
forces single-threaded BLAS, because thread count changes reduction order and
reduction order changes low-order bits. It bundles the corpora, so it runs
with `--network=none`.

```bash
make docker                 # build
make docker-smoke           # ~2 min inside the container
docker compose -f docker/docker-compose.yml run --rm reproduce
docker compose -f docker/docker-compose.yml run --rm tests
docker compose -f docker/docker-compose.yml run --rm shell
```

`results/` is bind-mounted so artefacts land on the host; `data/` is mounted
read-only so an experiment cannot mutate the corpora it is measuring. The
container runs as UID 10001, not root -- a process that redacts other
people's documents should not have more privilege than it needs.

## Repository layout

```
code-base/
├── src/aegis/
│   ├── types.py                 Span, Document, Certificate
│   ├── taxonomy.py              9 finance types + third-party label maps
│   ├── checksums.py             Luhn, ISO 7064 mod-97 (verify AND generate)
│   ├── metrics.py               matching, P/R/F1, BCa bootstrap, permutation, Holm
│   ├── pipeline.py              Algorithm 1, end to end
│   ├── cli.py                   `aegis` entry point
│   ├── data/                    corpus loaders, label mapping, splits
│   ├── layer1_candidates/       engines, MinHash/LSH, entity-graph propagation
│   ├── layer2_risk/             risks, p-values, CRC, Learn-then-Test
│   ├── layer3_shift/            density ratio, weighted CRC, chi^2-DRO radius
│   ├── layer4_monitor/          e-processes, e-BH, Chao1
│   └── layer5_act/              submodular escalation, surrogates, ledger
├── experiments/                 six drivers + run_all + paper-asset generation
├── tests/                       148 tests; the guarantees are tested, not asserted
├── configs/                     runnable operating points, documented
├── data/                        three corpora (~16 MB) + pinned manifest
├── docker/                      pinned image + compose
├── scripts/                     data verification, no-dependency test runner
└── results/paper/               reference outputs backing the paper
```

## Where each theorem lives

| paper | statement | code |
|---|---|---|
| Lemma 1 | propagation preserves exchangeability | `layer1_candidates/propagation.py::propagate` |
| Prop. 1 | recall amplification `1-(1-q)^m`; cross-entity leak bound | `propagation.py::recall_amplification`, `cross_entity_leak_bound` |
| Thm. 1 | document-level conformal risk control | `layer2_risk/crc.py::crc_select` |
| Thm. 2 | simultaneous leakage + utility certification | `layer2_risk/ltt.py::learn_then_test` |
| Thm. 3 | weighted CRC with estimated weights | `layer3_shift/weighted_crc.py` |
| Thm. 4 | the chi^2-DRO dual and `rho_star` | `layer3_shift/dro.py::certified_radius` |
| Thm. 5 | anytime-valid alarm + e-BH multiplicity | `layer4_monitor/eprocess.py`, `ebh.py` |
| Prop. 2 | label-free dark-matter bound | `layer4_monitor/chao.py::chao1` |
| Prop. 3 | submodular escalation, `(1-1/e)` | `layer5_act/escalate.py::lazy_greedy_escalate` |

Each has tests that check the *property*, not just the arithmetic: Lemma 1 is
tested by permuting the batch and requiring the propagated scores to permute
with it; Theorem 5 by measuring the empirical false-alarm rate against Ville's
bound over hundreds of null streams; Proposition 3 against brute-force optima.

## Data

Three synthetic corpora, ~16 MB, no real personal data, no downloads. See
[`data/README.md`](data/README.md) for provenance, licences, per-corpus
statistics, the exact label-mapping losses, and two known upstream defects
(40 out-of-range Gretel spans; 4 case-only Nemotron label mismatches) that are
counted and pinned rather than silently absorbed.

```bash
python scripts/verify_data.py     # SHA-256 + span integrity vs MANIFEST.json
```

## Tests

```bash
make test           # pytest
make test-bare      # same suite, no third-party runner needed
```

`scripts/run_tests_nodeps.py` implements the small slice of the pytest API the
suite uses, so the tests are runnable in a locked-down environment. A test
suite you cannot run is not evidence of anything.

```
passed=148 failed=0 skipped=0
```

## Honest differences from the paper

The paper's headline accuracy figures come from an ensemble including GLiNER
and a frontier LLM tier. The offline default here is three pure-Python
engines. Everything about the *certificate* is identical; what changes is
candidate formation, and therefore accuracy and the feasible `alpha`. Four
differences are worth stating plainly rather than leaving for a reader to
discover:

1. **The feasible `alpha` is looser offline.** With three pure-Python engines
   the candidate-miss floor on SynPII-F is ~2.9%, so `alpha = 0.01` is
   genuinely infeasible and the pipeline says so. The paper reaches
   `alpha_min = 0.006` with GLiNER in the ensemble. Install the optional
   engines (`AEGIS_ENGINES=rule,heuristic-ner,shape,gliner`) to close the gap.

2. **`rho_hat` is reported as infinite, not as a finite number, on the
   cross-corpus transfers.** The paper quotes 0.61 and 1.14. Here the
   discriminator separates the corpora perfectly, so the honest answer is that
   no finite radius applies. The *verdicts* agree with the paper -- Gretel and
   Nemotron are both out of scope -- and the guard is documented in
   `layer3_shift/weights.py`. We consider reporting `inf` more useful than
   reporting a small number produced by a collapsed ratio.

3. **The monitor comparison is more equivocal than the paper's.**
   `exp04` Part A confirms the paper's central claim: at nominal thresholds the
   repeated fixed-window test and a textbook CUSUM violate their level by
   13-17x, while the e-process and e-detector respect it. But Part B tunes
   each baseline's threshold on null streams until its false-alarm rate matches
   `delta`, and at matched false-alarm rates **a tuned CUSUM is competitive on
   delay**. That tuning needs the true null distribution and must be redone
   whenever `alpha` moves; the e-process needs neither. We report both.

4. **The Chao upper limit does not always cover.** `exp05` validates the
   label-free bound against gold labels and finds it covers 7/7 informative
   types on SynPII-F but only 4/6 on Nemotron -- `ACCOUNT_NUMBER` and `EMAIL`
   fall outside. This is the paper's own limitation made quantitative: engines
   are not independent capture occasions, so the estimator is an **indicator
   that should trigger a labelled audit, not a bound you may rely on**. The
   implementation flags the two regimes where it is vacuous (`f_2 = 0`, or
   fewer than three engines) rather than returning a confident large number.

Also unchanged from the paper's own limitations: propagation is transductive
within a batch, `rho_star` certifies `chi^2` balls and not adversarial
perturbations of the graph, reviewer accuracy is modelled rather than observed,
and nothing here defends against an adversary -- the ledger and the e-process
detect, they do not defend.

## Operating notes

**Change the surrogate key.** `configs/default.yaml` ships
`surrogate_key: aegis-release-key`, which is public. Deterministic surrogates
buy referential integrity at the price of linkability under auxiliary
knowledge: anyone who learns one (value, surrogate) pair learns that mapping
everywhere the key is in force. Rotation per release or per recipient isolates
cohorts at the cost of cross-cohort integrity. That is a governance decision
with the same review weight as `alpha`, and the ledger records which key
fingerprint governed which row.

**`alpha` is a policy instrument; `rho_star` is a precondition.** `alpha` is
chosen by whoever owns the risk. `rho_star` is not chosen at all -- it is a
property of the calibration losses and the selected operating point. Its only
operational use is the comparison `rho_hat <= rho_star`, which is what turns
"we recalibrate quarterly" into a testable statement about the documents
arriving today.

**An empty certificate is a result.** When `verdict == "no-certificate"` the
pipeline masks nothing on its own authority and routes the stream to review.
Do not paper over it by loosening `alpha` until something comes back; read
`alpha_min` from `exp02` and decide whether that level is acceptable policy.

**Do not report AEGIS-U as certified.** `select_by: max_f1` bypasses
Learn-then-Test entirely and reports `rho_star = 0`. It exists so the accuracy
comparison against uncontrolled baselines is fair, and for nothing else.

## Citing

See [`CITATION.cff`](CITATION.cff).

```bibtex
@inproceedings{mandavia2026aegis,
  title     = {{AEGIS}: Distribution-Shift-Certified Risk Control for {PII}
               Redaction in Financial Document Streams},
  author    = {Mandavia, Aayush Bharat and Kamat, Raghavendra},
  year      = {2026},
}
```

## Licence

Apache-2.0 for the code and for the SynPII-F corpora. The corpora in
`data/external/` are redistributed under their upstream dataset licences; see
[`data/README.md`](data/README.md).
