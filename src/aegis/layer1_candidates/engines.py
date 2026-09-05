"""Heterogeneous detection engines behind one interface.

Each engine exposes ``name`` and ``detect(text) -> list[Span]``. The point of
the ensemble is *complementarity*, not accuracy of any member: REDACT shows
rule systems and neural extractors fail in different places, so their union is
materially larger than either, their disagreement is what Layer V spends its
escalation budget on, and their independence (such as it is) is what lets
Layer IV treat them as capture occasions.

Three engines are pure Python and always available, so the whole pipeline and
every certificate in the paper can be reproduced offline. They are three
*different architectures*, not three tunings of one -- which matters twice
over: the union is larger than any member (Layer I), and Layer IV's
capture-recapture estimator needs ``M >= 3`` occasions before ``f_1``/``f_2``
carry usable information at all.

    RuleEngine           checksummed surface patterns + context gates
    HeuristicNEREngine   labelled-field structure + name lexicon/morphology
    ShapeClassEngine     pure character-shape classification, context-free

Three optional engines need extras or network access:

    GLiNEREngine         zero-shot span encoder      (pip install '.[gliner]')
    PresidioEngine       Microsoft Presidio baseline (pip install '.[presidio]')
    LLMEngine            frontier LLM tier           (needs ANTHROPIC_API_KEY)
"""
from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from typing import Iterable, Protocol, Sequence, runtime_checkable

from ..checksums import iban_valid, luhn_valid, ssn_plausible
from ..taxonomy import FINANCE_TYPES, PRESIDIO_TO_CANON
from ..types import Span

CONTEXT_CHARS = 48


@runtime_checkable
class Engine(Protocol):
    """Minimal engine contract."""

    name: str

    def detect(self, text: str) -> list[Span]: ...


def _context(text: str, start: int, end: int, width: int = CONTEXT_CHARS) -> str:
    return text[max(0, start - width) : min(len(text), end + width)].lower()


def dedupe_overlaps(spans: Sequence[Span], iou_thr: float = 0.5) -> list[Span]:
    """Keep the highest-scoring span among same-type heavy overlaps."""
    ordered = sorted(spans, key=lambda s: (-s.score, s.start, s.end))
    kept: list[Span] = []
    for span in ordered:
        if not any(k.type == span.type and span.iou(k) > iou_thr for k in kept):
            kept.append(span)
    return sorted(kept, key=lambda s: (s.start, s.end))


# ---------------------------------------------------------------------------
# Engine 1: rules with arithmetic verification
# ---------------------------------------------------------------------------

_EMAIL = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")
_CARD = re.compile(r"\b(?:\d[ \-]?){12,18}\d\b")
_IBAN = re.compile(r"\b[A-Z]{2}\d{2}[ ]?[A-Z0-9][A-Z0-9 ]{8,32}[A-Z0-9]\b")
_SSN_DASHED = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
_SSN_BARE = re.compile(r"\b\d{9}\b")
_PHONE = re.compile(r"(?:\+?1[ .\-]?)?(?:\(\d{3}\)[ .\-]?|\d{3}[ .\-])\d{3}[ .\-]\d{4}\b")
_DATE = re.compile(
    r"\b(?:\d{1,2}[/\-]\d{1,2}[/\-](?:19|20)\d{2}"
    r"|(?:19|20)\d{2}-\d{2}-\d{2}"
    r"|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2},? (?:19|20)\d{2})\b"
)
_ACCOUNT = re.compile(r"\b\d{8,17}\b")
_ADDRESS = re.compile(
    r"\b\d{1,5}\s+[A-Z][A-Za-z]+(?:\s[A-Z][A-Za-z]+)?\s"
    r"(?:St|Street|Ave|Avenue|Rd|Road|Ln|Lane|Blvd|Boulevard|Dr|Drive|Ct|Court|Way|Lock|Place|Pl)\b"
    r"(?:,?\s[A-Z][A-Za-z]+(?:\s[A-Z][A-Za-z]+)?,?\s[A-Z]{2}\s\d{5})?"
)
_NAME_CORE = r"([A-Z][a-z]+(?:\s[A-Z]\.)?(?:\s[A-Z][a-z]+){1,2})"
_HONORIFIC = re.compile(r"\b(?:Mr|Mrs|Ms|Dr|Prof)\.?\s" + _NAME_CORE)
# NOTE: the role keywords are case-insensitive but _NAME_CORE must NOT be --
# a global re.IGNORECASE would make "[A-Z][a-z]+" match "IBAN DE" and create
# large spurious PERSON identity components. Scoped (?i:...) keeps the case
# constraint on the name itself.
_ROLE = re.compile(
    r"(?i:borrower|applicant|client|customer|beneficiary|guarantor|holder|contact|"
    r"account\s+holder|payee|payer|remitter)\s*[,:]?\s+" + _NAME_CORE
)
_CAP_BIGRAM = re.compile(r"\b([A-Z][a-z]{2,})\s([A-Z][a-z]{2,})\b")


class RuleEngine:
    """Checksummed patterns plus context-window evidence.

    Scores are deliberately bimodal: a verified checksum earns ~0.97, an
    unverified match of the same shape earns ~0.35-0.55. That separation is
    what makes a strict alpha almost free on checksum-anchored types and
    expensive on PERSON, which has no arithmetic to appeal to.
    """

    name = "rule"

    def __init__(self, types: Iterable[str] = FINANCE_TYPES) -> None:
        self.types = frozenset(types)

    def _emit(self, out: list[Span], text: str, start: int, end: int, typ: str, score: float) -> None:
        if typ in self.types:
            out.append(Span(start, end, typ, text[start:end], score, self.name))

    def detect(self, text: str) -> list[Span]:
        out: list[Span] = []

        for m in _EMAIL.finditer(text):
            self._emit(out, text, m.start(), m.end(), "EMAIL", 0.97)

        for m in _CARD.finditer(text):
            digits = re.sub(r"\D", "", m.group())
            if 13 <= len(digits) <= 19:
                self._emit(out, text, m.start(), m.end(), "CREDIT_CARD",
                           0.98 if luhn_valid(digits) else 0.35)

        for m in _IBAN.finditer(text):
            self._emit(out, text, m.start(), m.end(), "IBAN",
                       0.99 if iban_valid(m.group()) else 0.40)

        for m in _SSN_DASHED.finditer(text):
            self._emit(out, text, m.start(), m.end(), "SSN",
                       0.96 if ssn_plausible(m.group()) else 0.50)
        for m in _SSN_BARE.finditer(text):
            if re.search(r"ssn|social security", _context(text, m.start(), m.end())):
                self._emit(out, text, m.start(), m.end(), "SSN", 0.85)

        for m in _PHONE.finditer(text):
            ctx = _context(text, m.start(), m.end())
            score = 0.92 if re.search(r"phone|tel|call|fax|mobile|contact|cell", ctx) else 0.55
            self._emit(out, text, m.start(), m.end(), "PHONE", score)

        for m in _DATE.finditer(text):
            ctx = _context(text, m.start(), m.end())
            score = 0.90 if re.search(r"\bdob\b|birth|born", ctx) else 0.20
            self._emit(out, text, m.start(), m.end(), "DOB", score)

        for m in _ACCOUNT.finditer(text):
            ctx = _context(text, m.start(), m.end())
            if re.search(r"account|acct|a/c|routing|wire|iban", ctx):
                self._emit(out, text, m.start(), m.end(), "ACCOUNT_NUMBER", 0.88)

        for m in _ADDRESS.finditer(text):
            score = 0.92 if re.search(r"\d{5}$", m.group()) else 0.80
            self._emit(out, text, m.start(), m.end(), "ADDRESS", score)

        for m in _HONORIFIC.finditer(text):
            self._emit(out, text, m.start(1), m.end(1), "PERSON", 0.85)
        for m in _ROLE.finditer(text):
            self._emit(out, text, m.start(1), m.end(1), "PERSON", 0.78)
        for m in _CAP_BIGRAM.finditer(text):
            self._emit(out, text, m.start(), m.end(), "PERSON", 0.40)

        return dedupe_overlaps(out)


# ---------------------------------------------------------------------------
# Engine 2: lexicon / morphology span extractor (architecturally distinct)
# ---------------------------------------------------------------------------

_GIVEN_NAMES = frozenset(
    """alex jordan riley casey morgan avery quinn rowan sage ellis harper reese
    emerson finley skyler dakota peyton kendall marlow tatum greta ingrid katy
    melanie james john robert michael william david richard joseph thomas charles
    mary patricia jennifer linda elizabeth barbara susan jessica sarah karen
    nancy lisa margaret betty sandra ashley dorothy kimberly emily donna michelle
    carol amanda helen anna maria daniel matthew anthony mark donald steven paul
    andrew joshua kenneth kevin brian george timothy ronald edward jason jeffrey
    ryan jacob gary nicholas eric stephen jonathan larry justin scott brandon
    benjamin samuel gregory frank alexander raymond patrick jack dennis jerry
    aayush raghavendra priya rahul ananya arjun kavya rohan divya""".split()
)
_SURNAME_SUFFIXES = (
    "son", "sen", "berg", "burg", "ford", "field", "wood", "worth", "ton", "ley",
    "man", "mann", "stein", "smith", "wick", "well", "combe", "bury", "more",
    "hart", "row", "dale", "shaw", "brook", "hill", "lane", "moor",
)
_TITLECASE_RUN = re.compile(r"\b(?:[A-Z][a-z]{1,15}|[A-Z]\.)(?:\s+(?:[A-Z][a-z]{1,15}|[A-Z]\.)){0,3}\b")
_LABELLED_FIELD = re.compile(
    r"(?im)^[\s\-*|]*(?:\*\*)?\s*"
    r"(first\s*name|last\s*name|middle\s*name|full\s*name|name|customer|borrower|"
    r"client|holder|email|e-mail|phone(?:\s*number)?|address|street\s*address|"
    r"date\s*of\s*birth|dob|ssn|social\s*security(?:\s*number)?|iban|"
    r"account\s*(?:number|no\.?)|card(?:\s*number)?|credit[/ ]?debit\s*card|"
    r"bank\s*routing\s*number)"
    r"\s*(?:\*\*)?\s*[:=]\s*(?:\*\*)?\s*(.+?)\s*(?:\*\*)?\s*$"
)
_FIELD_TO_TYPE = {
    "first name": "PERSON", "last name": "PERSON", "middle name": "PERSON",
    "full name": "PERSON", "name": "PERSON", "customer": "PERSON",
    "borrower": "PERSON", "client": "PERSON", "holder": "PERSON",
    "email": "EMAIL", "e-mail": "EMAIL",
    "phone": "PHONE", "phone number": "PHONE",
    "address": "ADDRESS", "street address": "ADDRESS",
    "date of birth": "DOB", "dob": "DOB",
    "ssn": "SSN", "social security": "SSN", "social security number": "SSN",
    "iban": "IBAN",
    "account number": "ACCOUNT_NUMBER", "account no": "ACCOUNT_NUMBER",
    "account no.": "ACCOUNT_NUMBER", "bank routing number": "ACCOUNT_NUMBER",
    "card": "CREDIT_CARD", "card number": "CREDIT_CARD",
    "credit/debit card": "CREDIT_CARD", "credit debit card": "CREDIT_CARD",
}
_STOP_TITLECASE = frozenset(
    """the this that these those and or but for with from into under over
    january february march april may june july august september october november
    december monday tuesday wednesday thursday friday saturday sunday
    account number identifier report summary statement notice memo bank branch
    credit debit card loan wire transfer payment customer client borrower
    applicant beneficiary holder compliance risk tier date name address phone
    email total amount balance interest rate term schedule details information
    safety data sheet product manufacturer contact emergency hazard
    classification composition ingredients first aid measures fire fighting
    identification toxicity assessment substance harmful""".split()
)


def _looks_like_person(fragment: str) -> bool:
    tokens = [t for t in fragment.split() if t]
    if not 2 <= len(tokens) <= 4:
        return False
    lowers = [t.strip(".,").lower() for t in tokens]
    if any(t in _STOP_TITLECASE for t in lowers):
        return False
    if any(t in _GIVEN_NAMES for t in lowers):
        return True
    return any(lowers[-1].endswith(sfx) for sfx in _SURNAME_SUFFIXES)


class HeuristicNEREngine:
    """A span extractor that shares no machinery with :class:`RuleEngine`.

    It works from *document structure* (labelled key/value fields, which is how
    the Nemotron and Gretel templates present PII) and from *lexical evidence*
    (given-name lexicon, surname morphology) rather than from checksums and
    regex shapes. That is the point: the two engines miss different spans, so
    the union recovers more and ``f_1``/``f_2`` in the Chao estimator carry
    real information.
    """

    name = "heuristic-ner"

    def __init__(self, types: Iterable[str] = FINANCE_TYPES) -> None:
        self.types = frozenset(types)

    def _emit(self, out: list[Span], text: str, start: int, end: int, typ: str, score: float) -> None:
        if typ in self.types and end > start:
            out.append(Span(start, end, typ, text[start:end], score, self.name))

    def detect(self, text: str) -> list[Span]:
        out: list[Span] = []

        # (a) labelled key/value fields -- the dominant template shape
        for m in _LABELLED_FIELD.finditer(text):
            field = re.sub(r"\s+", " ", m.group(1)).strip().lower()
            typ = _FIELD_TO_TYPE.get(field) or _FIELD_TO_TYPE.get(field.rstrip("."))
            if typ is None:
                continue
            value = m.group(2)
            if not value or value.startswith("[") or value.lower() in ("n/a", "none", "-"):
                continue
            start = m.start(2)
            end = start + len(value)
            trimmed = value.rstrip(" .*|")
            end -= len(value) - len(trimmed)
            self._emit(out, text, start, end, typ, 0.90)

        # (b) lexicon / morphology person mentions anywhere in the text
        for m in _TITLECASE_RUN.finditer(text):
            fragment = m.group()
            if _looks_like_person(fragment):
                score = 0.82 if fragment.split()[0].strip(".,").lower() in _GIVEN_NAMES else 0.62
                self._emit(out, text, m.start(), m.end(), "PERSON", score)

        # (c) emails and IBAN-shaped tokens are cheap to see structurally too,
        #     which is exactly the kind of positive dependence Prop. 2 warns of
        for m in _EMAIL.finditer(text):
            self._emit(out, text, m.start(), m.end(), "EMAIL", 0.93)

        return dedupe_overlaps(out)


# ---------------------------------------------------------------------------
# Engine 3: context-free character-shape classification
# ---------------------------------------------------------------------------

_SHAPE_SPLIT = re.compile(r"[\s,;|]+")
_TRIM = re.compile(r"^[\-*(\[\"\']+|[\-*)\]\"\'.:;]+$")


def shape_signature(token: str) -> str:
    """Collapse a token to a character-class signature.

    ``"406-44-8691" -> "d-d-d"``, ``"Melanie" -> "Ul"``, ``"DE89370400"
    -> "Ud"``. Runs collapse, so the signature is short and the mapping table
    stays readable.
    """
    out: list[str] = []
    for ch in token:
        cls = "d" if ch.isdigit() else ("U" if ch.isupper() else ("l" if ch.islower() else ch))
        if not out or out[-1] != cls:
            out.append(cls)
    return "".join(out)


#: Signature -> (type, score). Deliberately context-free: this engine trades
#: precision for a completely different error surface from the other two.
_SHAPE_TO_TYPE: dict[str, tuple[str, float]] = {
    "d-d-d": ("SSN", 0.72),
    "d/d/d": ("DOB", 0.66),
    "d-d-d-d": ("DOB", 0.52),
    "(d)d-d": ("PHONE", 0.70),
    "d-d": ("PHONE", 0.44),
    "+dd": ("PHONE", 0.58),
    "Ud": ("IBAN", 0.60),
    "d": ("ACCOUNT_NUMBER", 0.34),
    "Ul": ("PERSON", 0.36),
    "Ul.Ul": ("PERSON", 0.40),
    "l@l.l": ("EMAIL", 0.74),
    "l.l@l.l": ("EMAIL", 0.78),
}


class ShapeClassEngine:
    """Context-free shape classification over token groups.

    No regex vocabulary, no field labels, no name lexicon -- just the character
    -class signature of each token group and a small mapping table, with
    length gates to keep the obvious nonsense out. It finds bare identifiers
    the context-gated rule engine suppresses and misses everything that needs
    surrounding words, so its errors are close to orthogonal to the other two
    engines'.
    """

    name = "shape"

    def __init__(self, types: Iterable[str] = FINANCE_TYPES) -> None:
        self.types = frozenset(types)

    def detect(self, text: str) -> list[Span]:
        out: list[Span] = []
        position = 0
        for raw in _SHAPE_SPLIT.split(text):
            if not raw:
                position += 1
                continue
            start = text.find(raw, position)
            if start < 0:
                position += len(raw) + 1
                continue
            position = start + len(raw)
            token = _TRIM.sub("", raw)
            if len(token) < 3:
                continue
            offset = start + raw.find(token)
            hit = _SHAPE_TO_TYPE.get(shape_signature(token))
            if hit is None:
                continue
            typ, score = hit
            if typ not in self.types:
                continue
            digits = sum(c.isdigit() for c in token)
            if typ == "ACCOUNT_NUMBER" and not 8 <= digits <= 17:
                continue
            if typ == "IBAN" and not 12 <= len(token) <= 34:
                continue
            if typ == "PERSON" and (len(token) < 4 or token.lower() in _STOP_TITLECASE):
                continue
            if typ == "SSN" and digits != 9:
                continue
            out.append(Span(offset, offset + len(token), typ, token, score, self.name))

        # a person is two adjacent title-case tokens; look for the pair too
        for m in re.finditer(r"\b[A-Z][a-z]{2,}\s[A-Z][a-z]{2,}\b", text):
            pair = m.group()
            if all(t.lower() not in _STOP_TITLECASE for t in pair.split()):
                out.append(Span(m.start(), m.end(), "PERSON", pair, 0.46, self.name))
        return dedupe_overlaps(out)


# ---------------------------------------------------------------------------
# Optional engine 4: zero-shot span encoder (GLiNER family)
# ---------------------------------------------------------------------------

_GLINER_LABELS = {
    "person name": "PERSON",
    "email address": "EMAIL",
    "phone number": "PHONE",
    "street address": "ADDRESS",
    "date of birth": "DOB",
    "social security number": "SSN",
    "credit card number": "CREDIT_CARD",
    "iban": "IBAN",
    "bank account number": "ACCOUNT_NUMBER",
}


class GLiNEREngine:
    """Zero-shot span encoder. Requires the optional ``gliner`` extra."""

    name = "gliner"

    def __init__(self, model: str = "urchade/gliner_large-v2.1", threshold: float = 0.3,
                 types: Iterable[str] = FINANCE_TYPES) -> None:
        try:
            from gliner import GLiNER  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional path
            raise ImportError(
                "GLiNEREngine needs the 'gliner' extra: pip install '.[gliner]'"
            ) from exc
        self._model = GLiNER.from_pretrained(model)
        self.threshold = threshold
        self.types = frozenset(types)
        self._labels = [k for k, v in _GLINER_LABELS.items() if v in self.types]

    def detect(self, text: str) -> list[Span]:  # pragma: no cover - optional path
        found = self._model.predict_entities(text, self._labels, threshold=self.threshold)
        out: list[Span] = []
        for item in found:
            typ = _GLINER_LABELS.get(item["label"])
            if typ is None:
                continue
            out.append(Span(int(item["start"]), int(item["end"]), typ,
                            item["text"], float(item.get("score", 0.5)), self.name))
        return dedupe_overlaps(out)


# ---------------------------------------------------------------------------
# Optional engine 5: Presidio (used as a baseline, not as an ensemble member)
# ---------------------------------------------------------------------------


class PresidioEngine:
    """Microsoft Presidio under the paper's recall-generous type mappings."""

    def __init__(self, model: str = "en_core_web_lg", types: Iterable[str] = FINANCE_TYPES) -> None:
        try:
            from presidio_analyzer import AnalyzerEngine  # type: ignore
            from presidio_analyzer.nlp_engine import NlpEngineProvider  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional path
            raise ImportError(
                "PresidioEngine needs the 'presidio' extra: pip install '.[presidio]'"
            ) from exc
        provider = NlpEngineProvider(
            nlp_configuration={
                "nlp_engine_name": "spacy",
                "models": [{"lang_code": "en", "model_name": model}],
            }
        )
        self._analyzer = AnalyzerEngine(nlp_engine=provider.create_engine())
        self.name = f"presidio-{model.rsplit('_', 1)[-1]}"
        self.types = frozenset(types)

    def detect(self, text: str) -> list[Span]:  # pragma: no cover - optional path
        out: list[Span] = []
        for r in self._analyzer.analyze(text=text, language="en"):
            typ = PRESIDIO_TO_CANON.get(r.entity_type)
            if typ in self.types:
                out.append(Span(r.start, r.end, typ, text[r.start : r.end],
                                float(r.score), self.name))
        return dedupe_overlaps(out)


# ---------------------------------------------------------------------------
# Optional engine 6: frontier LLM tier (Layer V escalation target)
# ---------------------------------------------------------------------------

LLM_PROMPT = """Extract every personally identifiable information span from the document.
Return ONLY a JSON array, no prose. Each element must be:
{"text": "<verbatim span>", "type": "<one of ACCOUNT_NUMBER|ADDRESS|CREDIT_CARD|DOB|EMAIL|IBAN|PERSON|PHONE|SSN>", "confidence": <0..1>}
Spans must be verbatim substrings of the document.

Document:
"""


class LLMEngine:
    """Frontier LLM span extractor, invoked only on escalated spans.

    Needs ``ANTHROPIC_API_KEY``. ``max_calls`` is a hard cap so a runaway
    experiment cannot spend an unbounded amount; the escalation layer's budget
    is the intended control.
    """

    name = "llm"

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "claude-sonnet-4-6",
        url: str = "https://api.anthropic.com/v1/messages",
        max_calls: int = 10_000,
        timeout: float = 120.0,
        types: Iterable[str] = FINANCE_TYPES,
    ) -> None:
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        if not self.api_key:
            raise RuntimeError(
                "LLMEngine needs an API key (pass api_key= or set ANTHROPIC_API_KEY)"
            )
        self.model = model
        self.url = url
        self.max_calls = max_calls
        self.timeout = timeout
        self.types = frozenset(types)
        self.calls = 0
        self.input_tokens = 0
        self.output_tokens = 0

    def detect(self, text: str) -> list[Span]:  # pragma: no cover - network path
        if self.calls >= self.max_calls:
            return []
        self.calls += 1
        body = json.dumps(
            {
                "model": self.model,
                "max_tokens": 2000,
                "temperature": 0,
                "messages": [{"role": "user", "content": LLM_PROMPT + text}],
            }
        ).encode()
        request = urllib.request.Request(
            self.url,
            data=body,
            headers={
                "content-type": "application/json",
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
            },
        )
        try:
            raw = json.loads(urllib.request.urlopen(request, timeout=self.timeout).read())
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
            return []
        usage = raw.get("usage", {})
        self.input_tokens += int(usage.get("input_tokens", 0))
        self.output_tokens += int(usage.get("output_tokens", 0))
        payload = "".join(b.get("text", "") for b in raw.get("content", []))
        payload = re.sub(r"```(?:json)?|```", "", payload).strip()
        out: list[Span] = []
        try:
            items = json.loads(payload)
        except (json.JSONDecodeError, TypeError):
            return []
        for item in items if isinstance(items, list) else []:
            surface, typ = item.get("text", ""), item.get("type", "")
            if not surface or typ not in self.types:
                continue
            idx = text.find(surface)
            if idx < 0:
                continue
            out.append(Span(idx, idx + len(surface), typ, surface,
                            float(item.get("confidence", 0.5)), self.name))
        return dedupe_overlaps(out)


# ---------------------------------------------------------------------------
# Ensemble
# ---------------------------------------------------------------------------


class Ensemble:
    """Union of engine outputs, with per-engine provenance retained.

    Overlapping same-type proposals from *different* engines are merged into a
    single candidate whose score is the maximum and whose ``engine`` field
    lists every engine that proposed it, separated by ``+``. That list is the
    capture history the Chao estimator of Layer IV consumes.
    """

    def __init__(self, engines: Sequence[Engine], iou_thr: float = 0.5) -> None:
        if not engines:
            raise ValueError("an ensemble needs at least one engine")
        self.engines = list(engines)
        self.iou_thr = iou_thr

    @property
    def names(self) -> list[str]:
        return [e.name for e in self.engines]

    @property
    def n_engines(self) -> int:
        return len(self.engines)

    def detect(self, text: str) -> list[Span]:
        proposals: list[Span] = []
        for engine in self.engines:
            proposals.extend(engine.detect(text))
        return self._merge(proposals)

    def _merge(self, proposals: Sequence[Span]) -> list[Span]:
        merged: list[Span] = []
        for span in sorted(proposals, key=lambda s: (-s.score, s.start, s.end)):
            hit = None
            for kept in merged:
                if kept.type == span.type and kept.iou(span) >= self.iou_thr:
                    hit = kept
                    break
            if hit is None:
                merged.append(
                    Span(span.start, span.end, span.type, span.text, span.score, span.engine)
                )
            else:
                sources = set(hit.engine.split("+")) | {span.engine}
                hit.engine = "+".join(sorted(s for s in sources if s))
                hit.score = max(hit.score, span.score)
        return sorted(merged, key=lambda s: (s.start, s.end))


def build_ensemble(spec: str | Sequence[str], types: Iterable[str] = FINANCE_TYPES,
                   **kwargs) -> Ensemble:
    """Build an ensemble from a comma-separated spec, e.g. ``"rule,heuristic-ner"``.

    Unknown names raise; optional engines raise a clear ImportError/RuntimeError
    telling you which extra to install. The offline default of the released
    configs is ``rule,heuristic-ner``.
    """
    names = [n.strip() for n in (spec.split(",") if isinstance(spec, str) else spec) if n.strip()]
    built: list[Engine] = []
    for name in names:
        key = name.lower()
        if key == "rule":
            built.append(RuleEngine(types=types))
        elif key in ("heuristic-ner", "heuristic_ner", "hner"):
            built.append(HeuristicNEREngine(types=types))
        elif key in ("shape", "shape-class", "shapeclass"):
            built.append(ShapeClassEngine(types=types))
        elif key == "gliner":
            built.append(GLiNEREngine(types=types, **kwargs.get("gliner", {})))
        elif key.startswith("presidio"):
            model = "en_core_web_lg" if key in ("presidio", "presidio-lg") else "en_core_web_sm"
            built.append(PresidioEngine(model=model, types=types))
        elif key == "llm":
            built.append(LLMEngine(types=types, **kwargs.get("llm", {})))
        else:
            raise ValueError(f"unknown engine {name!r}")
    return Ensemble(built)
