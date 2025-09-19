#!/usr/bin/env python3
"""
filepath: main.py

Refined OCRParser focused on robust field-label detection and safer OCR normalization.
Now integrates:
  - AdministrativeRegionsRepository for Province → City/Kab → District → Village normalization (via beam search).
  - KTPReferenceRepository for enumerated KTP fields (gender, religion, etc.) using dual-pass search
    (label-window first, then global fallback).
  - Month-name and numeric date parsing; RT/RW detection in multiple shapes (RT 007 / RW 008 / "007/008").

Why:
  Keep OCR parsing deterministic and maintainable by delegating vocabulary and fuzzy
  resolution to dedicated repositories with consistent normalization. Add small, composable
  heuristics to cover frequent OCR failure modes while keeping behavior explainable.

Public API:
  - OCRParser.interpret(lines: List[str]) -> Dict[str, Any]
"""

from __future__ import annotations

import re
import json
import logging
import unicodedata
from dataclasses import dataclass
from functools import lru_cache
from typing import List, Dict, Any, Optional, Tuple

try:
    import Levenshtein  # only used for label detection distance
except Exception as e:
    raise ImportError("python-Levenshtein is required. Install with: pip install python-Levenshtein") from e

# Repos
from administrative_regions_repository import AdministrativeRegionsRepository, Region
from ktp_reference_repository import KTPReferenceRepository, Ref

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ParsedField:
    """Lightweight holder for extracted values (for potential future provenance needs)."""
    value: Optional[str]
    confidence: float
    source: str


class OCRParser:
    """
    Normalize OCR text into structured KTP fields by:
      - detecting noisy labels (glued or spaced) with look-ahead and small look-back,
      - extracting structured values via regex/heuristics (NIK, dates, RT/RW),
      - delegating fuzzy vocabulary matching to repositories,
      - resolving regions with a small hierarchy-aware beam search.

    Why:
      Keeps parsing logic small and robust; the repos own search/normalization,
      while the parser focuses on segmentation and assembly.

    Notes:
      • Deterministic: no ML models; string ops + SQLite-backed repos.
      • Typesafe: explicit Optional[str] on every output field.
      • Debug: set `debug=True` to attach confidences/evidence (non-breaking).
    """

    # Numeric-dominant token cleanup (letters that often mean digits)
    NUMERIC_OCR_MAP = str.maketrans({
        'O': '0', 'Q': '0',
        'I': '1', 'L': '1', '|': '1',
        'Z': '2', 'S': '5'
    })

    # Alpha-dominant token cleanup (digits that sneak into words)
    REVERSE_OCR_MAP = str.maketrans({
        '0': 'O', '1': 'I', '3': 'E', '4': 'A', '5': 'S', '6': 'G', '7': 'T', '8': 'B'
    })

    # Canonical UI labels (used only as label hints — actual vocab is normalized by repos)
    DEFAULT_LABELS = [
        "NAMA", "NIK", "ALAMAT",
        "TEMPAT LAHIR", "TANGGAL LAHIR", "TGL LAHIR", "TEMPAT TGL LAHIR",
        "PROVINSI", "KABUPATEN", "KOTA", "KECAMATAN", "KELURAHAN", "DESA", "KEL DESA",
        "RT", "RW",
        "AGAMA", "PEKERJAAN",
        "JENIS KELAMIN", "GOLONGAN DARAH",
        "STATUS PERKAWINAN",
        "KEWARGANEGARAAN",
        "BERLAKU HINGGA",
        "KOTA TERBIT", "TANGGAL TERBIT"
    ]

    # Aliases for frequent OCR confusions (labels/keywords only; values handled later)
    LABEL_ALIASES: Dict[str, str] = {
        "TEMPATTGLLAHR": "TEMPAT TGL LAHIR",
        "TEMPATTGL": "TEMPAT TGL LAHIR",
        "JENSKELMN": "JENIS KELAMIN",
        "GOLDRH": "GOLONGAN DARAH",
        "KELDES": "KEL DESA",
        "KEWARGANEGARAN": "KEWARGANEGARAAN",
        "STUSPRKAWNAN": "STATUS PERKAWINAN",
        "ALMAT": "ALAMAT",
        "RTRW": "RT",  # will still parse RW via value
    }

    VALUE_ALIASES: Dict[str, str] = {
        # common OCR in-values fixes for enums
        "PERMPUAN": "PEREMPUAN",
        "KAWlN": "KAWIN",          # l->I confusion
        "BELUMKAWIN": "BELUM KAWIN",
        "WNl": "WNI",
        "BUDHA": "BUDDHA",
        "KRISTEN PROTESTAN": "KRISTEN",
        "KRISTEN KATOLIK": "KATHOLIK",
        "SWASTA": "KARYAWAN SWASTA",  # map to common canonical
        "PEGAWAI SWASTA": "KARYAWAN SWASTA",
    }

    # Month names for Indonesian cards (uppercased after normalization)
    MONTHS_ID: Dict[str, str] = {
        "JAN": "01", "JANUARI": "01",
        "FEB": "02", "FEBRUARI": "02",
        "MAR": "03", "MARET": "03",
        "APR": "04", "APRIL": "04",
        "MEI": "05",
        "JUN": "06", "JUNI": "06",
        "JUL": "07", "JULI": "07",
        "AGU": "08", "AGUSTUS": "08",
        "SEP": "09", "SEPT": "09", "SEPTEMBER": "09",
        "OKT": "10", "OKTOBER": "10",
        "NOV": "11", "NOVEMBER": "11",
        "DES": "12", "DESEMBER": "12",
    }

    # Quick regexes for obvious numbers/dates/RT-RW
    NIK_RE = re.compile(r'\b[0-9OIL\|]{15,18}\b')
    DATE_NUMERIC_RE = re.compile(r'\b[0-3]?\d[-/][0-1]?\d[-/][12]?\d{2,3}\b')
    RT_RW_EXPLICIT_RE = (
        re.compile(r'\bRT\W*0*?(\d{1,3})\b', re.IGNORECASE),
        re.compile(r'\bRW\W*0*?(\d{1,3})\b', re.IGNORECASE)
    )
    RT_RW_SLASH_RE = re.compile(r'(?<!\d)(\d{1,3})\s*/\s*(\d{1,3})(?!\d)')

    def __init__(
        self,
        *,
        regions: Optional[AdministrativeRegionsRepository] = None,
        refs: Optional[KTPReferenceRepository] = None,
        field_labels: Optional[List[str]] = None,
        thresholds: Optional[Dict[str, float]] = None,
        debug: bool = False
    ):
        """
        Prepare the parser with repositories and label/threshold configs.

        Why:
            Keep parsing configurable and auditable. Repos are optional; parser still works
            with pass-through heuristics when repos are unavailable.

        Args:
            regions: Region repository used to normalize Provinsi/Kota/Kecamatan/Kel_Desa.
            refs: Reference repository used to normalize enumerated fields (gender, religion, etc.).
            field_labels: Override/augment candidate label strings to detect in noisy OCR.
            thresholds: Tuning knobs for label matching and cutoffs. Sensible defaults provided.
            debug: When True, attach `_debug` evidence (confidences/candidates) to the output.

        Returns:
            None
        """
        self.regions = regions
        self.refs = refs
        self.debug = debug

        # Label candidates: merge defaults with repo field names if available, plus aliases.
        label_pool = set(lbl.upper() for lbl in (field_labels or self.DEFAULT_LABELS))
        label_pool |= set(self.LABEL_ALIASES.values())
        if self.refs:
            for f in self.refs.get_refs():
                label_pool.add(f.replace('_', ' ').upper())
        self.field_labels = sorted(label_pool)
        self._label_variants = [{'label': lbl, 'label_nospace': lbl.replace(' ', '')} for lbl in self.field_labels]

        defaults = {
            "label_match_threshold": 0.82,
            "label_match_nospace_threshold": 0.75,
            "label_stop_threshold": 0.70,
            "value_lev_threshold": 0.66,
            "max_label_window": 3,
            "max_value_tokens": 8,
            "beam_k": 2,  # per-level branching for region search
        }
        self.thresholds = {**defaults, **(thresholds or {})}

    # -----------------------
    # Normalization & tokenization
    # -----------------------
    def _split_glued_token(self, tok: str) -> List[str]:
        """Keep alnum runs intact (preserve dates/numbers) to avoid over-splitting."""
        parts = re.findall(r'[A-Za-z0-9]+|[^A-Za-z0-9]+', tok)
        return [p for p in parts if re.search(r'[A-Za-z0-9]', p)]

    @staticmethod
    @lru_cache(maxsize=4096)
    def _alias_label_norm(s: str) -> str:
        """Normalize a label-ish string then apply label aliases; why: boost recall with glued/bent labels."""
        if not s:
            return ""
        t = unicodedata.normalize('NFKC', s)
        t = re.sub(r'\s+', ' ', t).strip().upper()
        t_ns = t.replace(' ', '')
        # alias by nospace key if present
        return OCRParser.LABEL_ALIASES.get(t_ns, t)

    def normalize_token(self, tok: str) -> str:
        """Normalize one token to mitigate OCR digit/letter swaps while preserving intent.

        Why:
            OCR often flips O↔0, I/L/|↔1, S↔5. We bias by token makeup so numbers remain numbers.

        Args:
            tok: Raw OCR token (possibly glued).

        Returns:
            Uppercased, minimally cleaned token to feed label/value heuristics.
        """
        if not tok:
            return ""
        t = unicodedata.normalize('NFKC', tok.strip())
        t = re.sub(r'^[^\w]+|[^\w]+$', '', t)
        letters = sum(ch.isalpha() for ch in t)
        digits = sum(ch.isdigit() for ch in t)

        if digits >= max(1, len(t) // 2):
            t = t.translate(self.NUMERIC_OCR_MAP)      # numeric-dominant
        else:
            t = t.translate(self.REVERSE_OCR_MAP)      # alpha-dominant

        t = re.sub(r'\s+', ' ', t)
        return self._alias_label_norm(t)

    def tokenize_lines(self, lines: List[str]) -> List[Dict[str, str]]:
        """Turn raw OCR lines into a token stream that preserves dates/numbers.

        Why:
            Keep line order but analyze on a flat, normalized stream for label/value pairing.

        Args:
            lines: Raw OCR lines.

        Returns:
            [{'raw': <raw>, 'norm': <normalized>}, ...]
        """
        tokens: List[Dict[str, str]] = []
        for line in lines:
            raw_parts = re.split(r'\s+', line.strip())
            for p in raw_parts:
                if not p:
                    continue
                for sp in self._split_glued_token(p):
                    norm = self.normalize_token(sp)
                    if norm:
                        tokens.append({'raw': sp, 'norm': norm})
        return tokens

    # -----------------------
    # Raw surface for regex (keep separators for dates)
    # -----------------------
    def _raw_text_for_regex(self, lines: List[str]) -> str:
        """Produce a raw-ish text line preserving date separators so regex remains effective.

        Why:
            Regex needs `-`/`/` and spacings intact to fish dates/RT/RW reliably.

        Args:
            lines: Raw OCR lines.

        Returns:
            Lightly normalized text (NFKC, common O/I/L/|→digits, compact spaces).
        """
        txt = " ".join(lines)
        txt = unicodedata.normalize('NFKC', txt)
        trans = str.maketrans({'O': '0', 'I': '1', 'l': '1', 'L': '1', '|': '1', 'Q': '0'})
        txt = txt.translate(trans)
        txt = re.sub(r'\s+', ' ', txt)
        return txt

    # -----------------------
    # Date parsing (numeric + month-name)
    # -----------------------
    def _regex_dates_from_raw(self, raw_text: str) -> List[str]:
        """Collect plausible dates in multiple OCR styles to support TTL/issue/expiry.

        Why:
            Real cards often have month names (e.g., '18 MEI 1995'). We normalize to 'DD-MM-YYYY' when possible.

        Args:
            raw_text: Lightly-normalized raw text.

        Returns:
            Unique date-like strings (separators normalized to '-').
        """
        if not raw_text:
            return []
        out: List[str] = []

        # 1) numeric forms
        for m in self.DATE_NUMERIC_RE.findall(raw_text):
            out.append(m.replace(' ', '-').replace('/', '-'))

        # 2) month-name forms (ID)
        #   e.g., 18 MEI 1995, 1 OKTOBER 2012, 05 AGU 12
        pattern = re.compile(
            r'\b([0-3]?\d)\s+([A-ZÄËÖÜA-Z]+)\s+([12]?\d{1,3})\b',
            re.IGNORECASE
        )
        for d, mon, y in pattern.findall(raw_text.upper()):
            mon_num = self.MONTHS_ID.get(mon.strip().upper())
            if not mon_num:
                continue
            yyyy = y
            if len(y) == 2:  # heuristic 2-digit year → 20xx/19xx? keep as-is to avoid wrong century
                yyyy = y
            elif len(y) == 3:  # rare OCR loss, skip
                continue
            out.append(f"{int(d):02d}-{mon_num}-{yyyy}")

        # dedup while keeping order
        seen, dedup = set(), []
        for d in out:
            if d not in seen:
                seen.add(d)
                dedup.append(d)
        return dedup

    # -----------------------
    # Regex deterministic extraction
    # -----------------------
    def regex_extract(self, joined_norm: str, *, raw_text: Optional[str] = None) -> Dict[str, Any]:
        """Grab high-confidence fields with regex so the rest can rely on context.

        Why:
            Structured fields (NIK, dates, RT/RW) are much more reliable via regex than fuzzy search.

        Args:
            joined_norm: Normalized token text (joined with spaces).
            raw_text: Raw-ish text that preserves separators.

        Returns:
            Partial extraction: nik, dates, rt, rw (when detected).
        """
        out: Dict[str, Any] = {}

        # NIK (prefer raw_text to avoid over-normalization)
        nik_src = raw_text or joined_norm
        m = self.NIK_RE.search(nik_src)
        if m:
            r = m.group(0)
            cleaned = r.replace('O', '0').replace('I', '1').replace('l', '1').replace('L', '1').replace('|', '1').replace('Q', '0')
            digits = re.sub(r'\D', '', cleaned)
            if len(digits) >= 16:
                digits = digits[:16]
            if len(digits) == 16:
                out['nik'] = ParsedField(digits, 0.99, 'regex_nik').__dict__

        # dates (numeric + month-name)
        dates = self._regex_dates_from_raw(raw_text or "") if raw_text else []
        if dates:
            out['dates'] = [ParsedField(d, 0.90, 'regex_date').__dict__ for d in dates]

        # RT/RW explicit (RT ddd, RW ddd)
        rt_re, rw_re = self.RT_RW_EXPLICIT_RE
        rt_m = rt_re.search(joined_norm)
        rw_m = rw_re.search(joined_norm)
        if rt_m:
            out['rt'] = ParsedField(rt_m.group(1), 0.95, 'regex_rt').__dict__
        if rw_m:
            out['rw'] = ParsedField(rw_m.group(1), 0.95, 'regex_rw').__dict__

        # combined dd/dd (not only 6 digits)
        if 'rt' not in out or 'rw' not in out:
            mslash = self.RT_RW_SLASH_RE.search(raw_text or joined_norm)
            if mslash:
                rtv, rwv = mslash.group(1), mslash.group(2)
                out.setdefault('rt', ParsedField(rtv, 0.90, 'regex_rt_slash').__dict__)
                out.setdefault('rw', ParsedField(rwv, 0.90, 'regex_rw_slash').__dict__)

        return out

    # -----------------------
    # Label detection
    # -----------------------
    def _lev_ratio(self, a: str, b: str) -> float:
        """Levenshtein ratio for label detection; higher is better."""
        if not a or not b:
            return 0.0
        dist = Levenshtein.distance(a.upper(), b.upper())
        return max(0.0, 1.0 - (dist / max(len(a), len(b))))
    
    def _key(self, s: str) -> str:
        """Return an uppercase, no-space, alnum-only key for fast comparisons.

        Produces a stable key to compare tokens/labels independent of spacing/punctuation.
        OCR noise often inserts/removes separators; a canonical key eliminates those differences.

        Args:
            s: Arbitrary text.

        Returns:
            Uppercase nospace alphanumeric key (e.g., "Jenis Kelamin" → "JENISKELAMIN").
        """
        return re.sub(r'[^A-Z0-9]+', '', unicodedata.normalize('NFKC', s or '').upper())


    def _is_label_at(self, tokens: List[Dict[str, str]], pos: int) -> Optional[Tuple[str, int, float]]:
        """Detect a field label beginning at `pos` and return (canonical_label, window_size, score).

        Slides a 1..N token window and asks `KTPReferenceRepository.search_refs(...)`
        to identify the most likely KTP field name for the window content.
        Lets a single, well-tuned fuzzy search (prefix + trigrams + char distance) handle
        real-world OCR variations (glued, split, misspelled) without hard-coded aliases.

        Args:
            tokens: Token stream as produced by `tokenize_lines` (dicts with 'norm').
            pos:    Index of the token where a label may begin.

        Returns:
            (canonical_label, window_size, score) if a label is recognized, else None.
            canonical_label is uppercase with spaces (e.g., "JENIS KELAMIN").
        """
        n = len(tokens)
        best: Optional[Tuple[str, int, float]] = None
        maxw = int(self.thresholds.get('max_label_window', 3))

        # Score threshold when using refs.search_refs; tuned to catch noisy labels.
        sr_thresh = float(self.thresholds.get('label_searchrefs_threshold', 0.70))

        # Fallback thresholds using built-in label variants if refs unavailable.
        lev_sp = float(self.thresholds.get('label_match_threshold', 0.80))
        lev_ns = float(self.thresholds.get('label_match_nospace_threshold', 0.68))

        for w in range(1, maxw + 1):
            if pos + w > n:
                break
            window_tokens = [tokens[i]['norm'] for i in range(pos, pos + w)]
            joined_space = " ".join(window_tokens)

            # 1) Prefer repository-backed label search when available
            if self.refs:
                hits = self.refs.search_refs(joined_space, k=1)
                if hits:
                    field_name, score = hits[0]
                    if score >= sr_thresh:
                        # Normalize to canonical uppercase with spaces
                        canon = field_name.replace('_', ' ').upper()
                        cand = (canon, w, float(score))
                        if best is None or cand[2] > best[2] or (cand[2] == best[2] and w > best[1]):
                            best = cand
                        # continue scanning larger windows to prefer longer, better matches
                        continue

            # 2) Fallback to internal label pool (no repo case)
            joined_nospace = "".join(window_tokens)
            for var in self._label_variants:
                r_sp = self._lev_ratio(joined_space, var['label'])
                r_ns = self._lev_ratio(joined_nospace, var['label_nospace'])
                r = max(r_sp, r_ns)
                threshold = lev_sp if r_sp >= r_ns else lev_ns
                if r >= threshold:
                    cand = (var['label'], w, r)
                    if best is None or r > best[2] or (r == best[2] and w > best[1]):
                        best = cand

        return best


    def detect_labels_and_values(self, tokens: List[Dict[str, str]], regex_out: Dict[str, Any]) -> Dict[str, Any]:
        """Segment tokens into (label → raw value) pairs using repo-guided label detection.

        Walks the token stream, uses `search_refs` to find labels, then captures a small,
        right-hand value window (stopping early if the next tokens look like a new label).
        Real scans often have missing colons, glued tokens, or broken labels; matching via
        `search_refs` is more reliable than static alias lists and reduces hand tuning.

        Args:
            tokens:    Token stream as produced by `tokenize_lines` (dicts with 'norm').
            regex_out: Early regex hits (NIK/dates/RT-RW) used to avoid mislabeling numeric spans.

        Returns:
            Dict[label_key] -> {'value','confidence','source','evidence'} where label_key
            is the snake_case version of the canonical field (e.g., 'jenis_kelamin').
        """
        out: Dict[str, Any] = {}
        i = 0
        n = len(tokens)

        # Prepare stop keys (canonicalized) to cut value windows early
        stop_names = set(self.field_labels)
        if self.refs:
            # Prefer repo field names (more authoritative)
            stop_names = set(self.refs.get_refs())
        stop_keys = {self._key(name) for name in stop_names}
        # Also treat RT/RW markers as hard stops
        stop_keys.update({self._key(x) for x in ("RT", "RW", "RTRW")})

        def looks_like_label_at(k: int) -> bool:
            hit = self._is_label_at(tokens, k)
            if not hit:
                return False
            # accept lower bar for "stop" than for "start"
            return hit[2] >= float(self.thresholds.get('label_stop_threshold', 0.66))

        def is_hard_stop(tok_norm: str) -> bool:
            return self._key(tok_norm) in stop_keys

        max_val_tokens = int(self.thresholds.get('max_value_tokens', 6))

        while i < n:
            lab = self._is_label_at(tokens, i)
            if lab:
                label_name, span, score = lab  # label_name is uppercase with spaces
                j = i + span
                val_tokens: List[str] = []
                while j < n and len(val_tokens) < max_val_tokens:
                    if looks_like_label_at(j):
                        break
                    if is_hard_stop(tokens[j]['norm']):
                        break
                    val_tokens.append(tokens[j]['norm'])
                    j += 1

                value_text = " ".join(val_tokens).strip()
                if value_text:
                    key = label_name.lower().replace(' ', '_')
                    # Confidence: base on search_refs score if used; cap for sanity
                    conf = min(0.95, 0.60 + 0.25 * float(score))
                    out[key] = {
                        'value': value_text,
                        'confidence': conf,
                        'source': 'label_infer',
                        'evidence': {'label': label_name, 'ratio': float(score), 'span': span}
                    }
                i = j
            else:
                i += 1

        return out
    
    def _scan_enum(self, text: str, *, kind: str) -> Optional[str]:
        """Extract a canonical enum value from a noisy value window (substring-aware).

        Probes small 1–2 token combinations inside the captured value text to find
        the closest canonical enum (gender/blood/religion/marital/job/citizenship).
        Captured windows often contain extra tokens (e.g., "LAK1-LAK1 GolDrh A");
        scanning short subspans fixes many borderline matches without lowering thresholds.

        Args:
            text: Raw captured value text (noisy).
            kind: One of {'gender','blood','religion','marital','job','citizenship'}.

        Returns:
            Canonical enum display value (e.g., "PEREMPUAN", "AB", "ISLAM", "KAWIN", "WIRASWASTA", "WNI"),
            or None if no candidate meets a minimal score.
        """
        if not text or not self.refs:
            return None

        raw = unicodedata.normalize('NFKC', text)
        parts = [p for p in re.split(r'[\s,/|-]+', raw) if p]  # keep short tokens; hyphen splits too

        def best(search_fn, cands: List[str], min_score: float) -> Optional[str]:
            top_val, top_sc = None, 0.0
            for cand in cands:
                hits = search_fn(cand, k=1)
                if hits:
                    ref, sc = hits[0]
                    if sc > top_sc:
                        top_val, top_sc = ref.value, float(sc)
            return top_val if top_sc >= min_score else None

        if kind == "gender":
            cands: List[str] = []
            for i in range(len(parts)):
                cands.append(parts[i])
                if i + 1 < len(parts):
                    cands.append(f"{parts[i]} {parts[i+1]}")
                    cands.append(f"{parts[i]}-{parts[i+1]}")
            return best(self.refs.search_genders, cands, 0.55)

        if kind == "blood":
            # Map common OCR confusions into plausible blood tokens
            norm_parts = []
            for p in parts:
                p2 = p.upper().replace('0', 'O').replace('8', 'B')
                norm_parts.append(p2)
            # try single tokens first, then first token of the window
            cands = norm_parts[:3]  # first few tokens usually carry it
            return best(self.refs.search_blood_types, cands, 0.50)

        if kind == "religion":
            cands = []
            for i in range(len(parts)):
                cands.append(parts[i])
                if i + 1 < len(parts):
                    cands.append(f"{parts[i]} {parts[i+1]}")
            return best(self.refs.search_religions, cands, 0.55)

        if kind == "marital":
            cands = []
            for i in range(len(parts)):
                cands.append(parts[i])
                if i + 1 < len(parts):
                    cands.append(f"{parts[i]} {parts[i+1]}")
            return best(self.refs.search_marital_statuses, cands, 0.55)

        if kind == "job":
            cands = []
            for i in range(len(parts)):
                cands.append(parts[i])
                if i + 1 < len(parts):
                    cands.append(f"{parts[i]} {parts[i+1]}")
            return best(self.refs.search_jobs, cands, 0.55)

        if kind == "citizenship":
            cands = []
            for i in range(len(parts)):
                cands.append(parts[i])
            return best(self.refs.search_citizenships, cands, 0.50)

        return None



    # -----------------------
    # Repositories: helpers
    # -----------------------
    @staticmethod
    def _pick_top(scored: List[Tuple[Any, float]], min_score: float) -> Optional[Any]:
        """Choose top candidate above a threshold so we avoid weak matches."""
        if not scored:
            return None
        item, score = max(scored, key=lambda x: x[1])
        return item if score >= min_score else None

    # -----------------------
    # Region normalization (beam search)
    # -----------------------
    def _normalize_region_hierarchy(
        self,
        raw_regions: Dict[str, str],
        raw_text: Optional[str] = None
    ) -> Dict[str, Optional[str]]:
        """Resolve Province → City/Kab → District → Village using a small beam search.

        Why:
            Local best-at-each-level can pick inconsistent parents. A tiny beam (k≈2)
            with consistency bonus yields robust, explainable paths without complexity.

        Args:
            raw_regions: Raw strings captured near labels (very noisy).
            raw_text: Whole text as a fallback hint.

        Returns:
            {'Provinsi','Kota','Kecamatan','Kel_Desa'} normalized names (or None).
        """
        if not self.regions:
            return {
                "Provinsi": (raw_regions.get('provinsi') or None),
                "Kota": (raw_regions.get('kota') or raw_regions.get('kabupaten') or None),
                "Kecamatan": (raw_regions.get('kecamatan') or None),
                "Kel_Desa": (raw_regions.get('kelurahan') or raw_regions.get('kel_desa') or raw_regions.get('desa') or None),
            }

        dbg_paths: List[Tuple[Tuple[Optional[Region], Optional[Region], Optional[Region], Optional[Region]], float]] = []

        beam_k = int(self.thresholds.get('beam_k', 2))

        prov_q = (raw_regions.get('provinsi') or "").strip() or (raw_text or "")
        prov_hits = self.regions.search_provinces(prov_q, k=max(beam_k, 2)) or []
        prov_hits = prov_hits[:beam_k] if prov_hits else []

        # If nothing from explicit/whole text, leave as None path
        if not prov_hits:
            prov_hits = [(None, 0.0)]  # type: ignore

        kota_q = (raw_regions.get('kota') or raw_regions.get('kabupaten') or "").strip() or (raw_text or "")
        kec_q = (raw_regions.get('kecamatan') or "").strip() or (raw_text or "")
        kel_q = (raw_regions.get('kelurahan') or raw_regions.get('kel_desa') or raw_regions.get('desa') or "").strip() or (raw_text or "")

        for prov, ps in prov_hits:
            prov_id = prov.region_id if isinstance(prov, Region) else None

            city_hits = self.regions.search_cities(kota_q, province_id=prov_id, k=max(beam_k, 2)) if kota_q else []
            if not city_hits:
                # try global city fallback (unscoped)
                city_hits = self.regions.search_cities(kota_q or (raw_text or ""), province_id=None, k=max(beam_k, 2))
            city_hits = city_hits[:beam_k] if city_hits else [(None, 0.0)]  # type: ignore

            for city, cs in city_hits:
                city_id = city.region_id if isinstance(city, Region) else None
                dist_hits = self.regions.search_districts(kec_q, city_id=city_id, k=max(beam_k, 2)) if kec_q else []
                if not dist_hits:
                    dist_hits = self.regions.search_districts(kec_q or (raw_text or ""), city_id=None, k=max(beam_k, 2))
                dist_hits = dist_hits[:beam_k] if dist_hits else [(None, 0.0)]  # type: ignore

                for dist, ds in dist_hits:
                    dist_id = dist.region_id if isinstance(dist, Region) else None
                    vill_hits = self.regions.search_villages(kel_q, district_id=dist_id, k=max(beam_k, 2)) if kel_q else []
                    if not vill_hits:
                        vill_hits = self.regions.search_villages(kel_q or (raw_text or ""), district_id=None, k=max(beam_k, 2))
                    vill_hits = vill_hits[:beam_k] if vill_hits else [(None, 0.0)]  # type: ignore

                    for vill, vs in vill_hits:
                        score = 0.0
                        # base scores
                        score += (ps or 0.0) + (cs or 0.0) + (ds or 0.0) + (vs or 0.0)
                        # consistency bonuses
                        bonus = 0.0
                        if isinstance(prov, Region) and isinstance(city, Region) and city.parent_id == prov.region_id:
                            bonus += 0.05
                        if isinstance(city, Region) and isinstance(dist, Region) and dist.parent_id == city.region_id:
                            bonus += 0.05
                        if isinstance(dist, Region) and isinstance(vill, Region) and vill.parent_id == dist.region_id:
                            bonus += 0.05
                        score += bonus
                        dbg_paths.append(((prov, city, dist, vill), score))

        best = max(dbg_paths, key=lambda x: x[1]) if dbg_paths else (((None, None, None, None)), 0.0)
        (prov_r, city_r, dist_r, vill_r), _ = best

        return {
            "Provinsi": (prov_r.name_off if isinstance(prov_r, Region) else None),
            "Kota": (city_r.name_off if isinstance(city_r, Region) else None),
            "Kecamatan": (dist_r.name_off if isinstance(dist_r, Region) else None),
            "Kel_Desa": (vill_r.name_off if isinstance(vill_r, Region) else None),
        }

    # -----------------------
    # Simple enums via KTPReferenceRepository (dual pass)
    # -----------------------
    def _normalize_value_alias(self, text: str) -> str:
        """Apply curated in-value aliases. Why: common OCR typos map to canonical vocab."""
        t = re.sub(r'\s+', ' ', text).strip().upper()
        return self.VALUE_ALIASES.get(t, t)

    def _standardize_simple_fields(
        self,
        label_out: Dict[str, Any],
        tokens_text: Optional[str] = None
    ) -> Dict[str, Optional[str]]:
        """Normalize enum-like fields to canonical KTP vocabulary.

        Converts noisy captured values for gender, blood type, religion, marital status,
        job, and citizenship into canonical forms using `KTPReferenceRepository`, with
        a robust substring scan fallback.
        Value windows often carry extra tokens or OCR noise. Trying the full window first,
        then probing short subspans, yields high recall without loosening thresholds globally.

        Args:
            label_out: Raw values captured after label detection.
            tokens_text: Optional joined token text (not required here; accepted to match caller).

        Returns:
            Canonical values (or None) for:
            {'Jenis_Kelamin','Golongan_Darah','Agama','Status_Perkawinan','Pekerjaan','Kewarganegaraan'}.
        """
        # Fallback (no refs): minimal uppercase pass-through
        if not self.refs:
            def up(x: str) -> str: return re.sub(r'\s+', ' ', x).strip().upper()
            return {
                "Jenis_Kelamin": up(label_out.get('jenis_kelamin', {}).get('value', '')) or None,
                "Golongan_Darah": up(label_out.get('golongan_darah', {}).get('value', '').split()[-1]) or None,
                "Agama": up(label_out.get('agama', {}).get('value', '')) or None,
                "Status_Perkawinan": up(label_out.get('status_perkawinan', {}).get('value', '')) or None,
                "Pekerjaan": up(label_out.get('pekerjaan', {}).get('value', '')) or None,
                "Kewarganegaraan": up((label_out.get('kewarganegaraan') or label_out.get('kewarganegaran') or {}).get('value', '')) or None,
            }

        def top_val(pairs: List[Tuple[Ref, float]], min_score: float) -> Optional[str]:
            pick = self._pick_top(pairs, min_score)
            return pick.value if isinstance(pick, Ref) else None

        # Jenis Kelamin
        jk_win = (label_out.get('jenis_kelamin', {}) or {}).get('value', '')
        jk = top_val(self.refs.search_genders(jk_win, k=3), 0.60) if jk_win else None
        if not jk and jk_win:
            jk = self._scan_enum(jk_win, kind="gender")

        # Golongan Darah
        gd_win_all = (label_out.get('golongan_darah', {}) or {}).get('value', '')
        gd_token = gd_win_all.split()[-1] if gd_win_all else ''
        gd = top_val(self.refs.search_blood_types(gd_token or gd_win_all, k=3), 0.55) if (gd_token or gd_win_all) else None
        if not gd and gd_win_all:
            gd = self._scan_enum(gd_win_all, kind="blood")

        # Agama
        ag_win = (label_out.get('agama', {}) or {}).get('value', '')
        ag = top_val(self.refs.search_religions(ag_win, k=5), 0.55) if ag_win else None
        if not ag and ag_win:
            ag = self._scan_enum(ag_win, kind="religion")

        # Status Perkawinan
        sp_win = (label_out.get('status_perkawinan', {}) or {}).get('value', '')
        sp = top_val(self.refs.search_marital_statuses(sp_win, k=5), 0.55) if sp_win else None
        if not sp and sp_win:
            sp = self._scan_enum(sp_win, kind="marital")

        # Pekerjaan
        pk_win = (label_out.get('pekerjaan', {}) or {}).get('value', '')
        pk = top_val(self.refs.search_jobs(pk_win, k=5), 0.55) if pk_win else None
        if not pk and pk_win:
            pk = self._scan_enum(pk_win, kind="job")

        # Kewarganegaraan
        kw_win = (label_out.get('kewarganegaraan') or label_out.get('kewarganegaran') or {}).get('value', '')
        kw = top_val(self.refs.search_citizenships(kw_win, k=3), 0.55) if kw_win else None
        if not kw and kw_win:
            kw = self._scan_enum(kw_win, kind="citizenship")

        return {
            "Jenis_Kelamin": jk,
            "Golongan_Darah": gd,
            "Agama": ag,
            "Status_Perkawinan": sp,
            "Pekerjaan": pk,
            "Kewarganegaraan": kw
        }

    # -----------------------
    # Post-processing (cleanup to standard schema)
    # -----------------------
    @staticmethod
    def _clean_person_text(text: str) -> str:
        """Tidy names/places so casing and glued letters look human-readable.

        Why:
            Names are often split into single letters or have OCR digit bleed; this recovers readability.
        """
        def merge_fragments(t: str) -> str:
            parts = t.split()
            buf, cur = [], []
            for p in parts:
                if len(p) == 1:
                    cur.append(p)
                else:
                    if cur:
                        buf.append("".join(cur))
                        cur = []
                    buf.append(p)
            if cur:
                buf.append("".join(cur))
            return " ".join(buf)

        t = text.translate(OCRParser.REVERSE_OCR_MAP)
        t = re.sub(r'\s+', ' ', t).strip()
        t = merge_fragments(t)
        t = re.sub(r'\bPERMPUAN\b', 'PEREMPUAN', t, flags=re.IGNORECASE)
        words = []
        for w in t.split(' '):
            if len(w) <= 3 and w.isupper():
                words.append(w)
            else:
                words.append(w.capitalize())
        return " ".join(words)

    @staticmethod
    def _clean_address_text(text: str) -> str:
        """Normalize address strings while preserving meaningful digits.

        What:
            Tidies spacing, standardizes common abbreviations (e.g., 'JL.'), and
            separates letter/digit runs for readability.

        Why:
            Previous version converted digits → letters (e.g., '14' → 'IA'), which corrupts
            house numbers/RT-RW. Addresses frequently include numbers that must be kept verbatim.

        Args:
            text: Raw address string (noisy OCR).

        Returns:
            Uppercase cleaned address without altering digits.
        """
        if not text:
            return text
        # Keep digits intact; only collapse whitespace and fix common abbreviations.
        t = unicodedata.normalize('NFKC', text)
        t = re.sub(r'\s+', ' ', t).strip()
        t = re.sub(r'\bJL\b\.?', 'JL.', t, flags=re.IGNORECASE)
        # improve readability between letters and digits (e.g., "KAV12" → "KAV 12")
        t = re.sub(r'([A-Za-z])(\d)', r'\1 \2', t)
        t = re.sub(r'(\d)([A-Za-z])', r'\1 \2', t)
        return t.upper()


    @staticmethod
    def _strip_rt_rw_from_alamat(alamat: str) -> Tuple[str, Optional[str], Optional[str]]:
        """Remove explicit RT/RW fragments from address and return (clean, rt, rw).

        Strips **explicitly marked** RT/RW segments only (e.g., 'RT 007', 'RW 008', 'RTRW 007008').
        Does **not** treat generic 'NN/NN' house numbers as RT/RW to avoid false positives.
        Addresses often contain slashes in house numbers (e.g., '45/12'); parsing those as RT/RW
        leads to wrong output. We only honor clear markers.

        Args:
            alamat: Candidate full address (noisy).

        Returns:
            (clean_address, rt, rw) where rt/rw are digit strings (no zero-padding) or None.
        """
        if not alamat:
            return alamat, None, None

        t = alamat

        # Explicit RTRW 6 digits
        m_rtrw = re.search(r'\bRTRW\W*0*(\d{3})0*(\d{3})\b', t, re.IGNORECASE)
        rt, rw = (m_rtrw.group(1), m_rtrw.group(2)) if m_rtrw else (None, None)
        if m_rtrw:
            t = re.sub(r'\bRTRW\W*\d{6}\b', '', t, flags=re.IGNORECASE)

        # Explicit RT / RW markers (prefer explicit over generic)
        m_rt = re.search(r'\bRT\W*0*(\d{1,3})\b', t, re.IGNORECASE)
        m_rw = re.search(r'\bRW\W*0*(\d{1,3})\b', t, re.IGNORECASE)
        if m_rt:
            rt = rt or m_rt.group(1)
            t = re.sub(r'\bRT\W*\d{1,3}\b', '', t, flags=re.IGNORECASE)
        if m_rw:
            rw = rw or m_rw.group(1)
            t = re.sub(r'\bRW\W*\d{1,3}\b', '', t, flags=re.IGNORECASE)

        # Clean leftover spaces
        t = re.sub(r'\s+', ' ', t).strip()
        return t, rt, rw


    @staticmethod
    def _fmt_rt_rw(rt: Optional[str], rw: Optional[str]) -> Optional[str]:
        """Format RT/RW in 'RRR/WWW' or None when missing, preserving numeric intent."""
        if not rt and not rw:
            return None
        def pad(x: Optional[str]) -> str:
            if x and x.isdigit():
                return f"{int(x):03d}"
            return x or ""
        rt3, rw3 = pad(rt), pad(rw)
        if rt3 and rw3:
            return f"{rt3}/{rw3}"
        return rt3 or rw3 or None

    @staticmethod
    def _pick_best_date(dates: List[str]) -> Optional[str]:
        """Prefer 4-digit year forms to reduce ambiguity."""
        if not dates:
            return None
        def score(d: str) -> Tuple[int, int]:
            y = d.split('-')[-1]
            return (1 if len(y) == 4 else 0, 0)
        return sorted(dates, key=score, reverse=True)[0]

    def _extract_ttl(
        self,
        label_out: Dict[str, Any],
        tokens_joined: str,
        all_dates: List[str],
        raw_text: Optional[str]
    ) -> Optional[str]:
        """Assemble 'Tempat, DD-MM-YYYY' from combined/separate cues.

        Why:
            Users expect a compact, human-readable field; TTL is a common composite on KTP.

        Returns:
            'PLACE, DD-MM-YYYY' or None.
        """
        # direct combined label
        comb = label_out.get('tempat_tgl_lahir', {}).get('value')
        if comb:
            m = re.search(r'([0-3]?\d[-/ ][0-1]?\d[-/ ][12]?\d{2,3})', comb)
            if m:
                place = comb[:m.start()].strip()
                date = m.group(1).replace(' ', '-').replace('/', '-')
                place = self._clean_person_text(place).upper()
                return f"{place}, {date}"
            if all_dates:
                place = self._clean_person_text(comb).upper()
                return f"{place}, {self._pick_best_date(all_dates) or all_dates[0]}"

        # separate labels
        place = label_out.get('tempat_lahir', {}).get('value')
        date1 = (label_out.get('tanggal_lahir') or {}).get('value') or (label_out.get('tgl_lahir') or {}).get('value')
        if place and date1:
            place = self._clean_person_text(place).upper()
            return f"{place}, {date1.replace(' ', '-').replace('/', '-')}"

        if place and all_dates:
            place = self._clean_person_text(place).upper()
            return f"{place}, {self._pick_best_date(all_dates) or all_dates[0]}"

        # raw_text fallback
        if raw_text and all_dates:
            m = re.search(r'([0-3]?\d[-/ ][0-1]?\d[-/ ][12]?\d{2,3})', raw_text)
            if m:
                left = raw_text[:m.start()].strip().split()
                cand = " ".join(left[-3:]) if left else ""
                if cand:
                    cand = self._clean_person_text(cand).upper()
                    return f"{cand}, {self._pick_best_date(all_dates) or all_dates[0]}"
        return None

    def _guess_kota_terbit(
        self,
        tokens_text: str,
        prov_region_name: Optional[str],
        label_out: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        """Guess 'Kota_Terbit' using explicit label first, then tail-of-text fallback.

        If the OCR captured an explicit 'Kota Terbit' value, normalize that via city search.
        Otherwise, infer from the trailing tokens of the whole text, constrained by province.
        Issuing city/regency is often printed near the bottom; explicit capture wins,
        but a scoped fallback recovers many cases with missing labels.

        Args:
            tokens_text: Full normalized token text (joined with spaces).
            prov_region_name: Province name to scope city search (improves precision).
            label_out: Optional label capture dict to look for 'kota_terbit'.

        Returns:
            Canonical city/regency name or None.
        """
        if not self.regions:
            return None

        # Province scope → region_id
        province_id: Optional[str] = None
        if prov_region_name:
            ph = self.regions.search_provinces(prov_region_name, k=1)
            top_prov = self._pick_top(ph, 0.58)  # type: ignore[arg-type]
            if isinstance(top_prov, Region):
                province_id = top_prov.region_id

        # 1) Explicit label if present
        city_raw = None
        if label_out:
            city_raw = (label_out.get('kota_terbit') or {}).get('value')

        if city_raw:
            hits = self.regions.search_cities(city_raw, province_id=province_id, k=1)
            best = self._pick_top(hits, 0.58)  # type: ignore[arg-type]
            if isinstance(best, Region):
                return best.name_off

        # 2) Fallback: scan tail tokens
        tail = " ".join(tokens_text.split()[-8:])
        hits = self.regions.search_cities(tail, province_id=province_id, k=1)
        best = self._pick_top(hits, 0.58)  # type: ignore[arg-type]
        return best.name_off if isinstance(best, Region) else None

    # -----------------------
    # Assembly to standard KTP JSON schema
    # -----------------------
    def assemble_ktp_json(
        self,
        tokens: List[Dict[str, str]],
        regex_out: Dict[str, Any],
        label_out: Dict[str, Any],
        raw_text: Optional[str] = None
    ) -> Dict[str, Any]:
        """Assemble the final KTP JSON by merging regex hits, label windows, and repo-normalization.

        What:
            Consolidates deterministic regex fields (NIK/dates/RT-RW), repo-backed enums,
            and hierarchical region resolution into a stable output schema.

        Why:
            Keeps downstream consumers simple—one normalized, minimal payload per record.

        Returns:
            Final KTP payload keyed by your schema (strings or None).
        """
        joined_norm = " ".join(t['norm'] for t in tokens)

        # 1) Regions (raw capture from labels) → normalization
        raw_regions = {
            'provinsi': (label_out.get('provinsi') or {}).get('value'),
            'kota': (label_out.get('kota') or {}).get('value'),
            'kabupaten': (label_out.get('kabupaten') or {}).get('value'),
            'kecamatan': (label_out.get('kecamatan') or {}).get('value'),
            'kelurahan': (label_out.get('kelurahan') or {}).get('value'),
            'kel_desa': (label_out.get('kel_desa') or {}).get('value'),
            'desa': (label_out.get('desa') or {}).get('value'),
        }
        regions_std = self._normalize_region_hierarchy(
            {k: (v or "") for k, v in raw_regions.items()},
            raw_text=raw_text
        )

        # 2) Simple fields via refs (dual pass with substring fallback)
        simple = self._standardize_simple_fields(label_out, joined_norm)

        # 3) NIK
        nik = (regex_out.get('nik') or {}).get('value')

        # 4) Nama
        nama_raw = (label_out.get('nama') or {}).get('value', '')
        nama = self._clean_person_text(nama_raw) if nama_raw else None

        # 5) Alamat (strip explicit RT/RW from address and recycle if missing)
        alamat_raw = (label_out.get('alamat') or {}).get('value', '')
        alamat = self._clean_address_text(alamat_raw) if alamat_raw else None
        rt_from_addr, rw_from_addr = None, None
        if alamat:
            alamat, rt_from_addr, rw_from_addr = self._strip_rt_rw_from_alamat(alamat)

        # 6) RT/RW (regex or from explicit markers in address)
        rt_val = (regex_out.get('rt') or {}).get('value') or rt_from_addr
        rw_val = (regex_out.get('rw') or {}).get('value') or rw_from_addr
        rt_rw = self._fmt_rt_rw(rt_val, rw_val)

        # 7) Tempat, Tgl Lahir
        dates = [d['value'] for d in (regex_out.get('dates') or [])]
        ttl = self._extract_ttl(label_out, joined_norm, dates, raw_text)

        # 8) Berlaku Hingga
        berlaku_raw = (label_out.get('berlaku_hingga') or {}).get('value')
        berlaku = None
        if berlaku_raw:
            in_dates = re.findall(r'[0-3]?\d[-/ ][0-1]?\d[-/ ][12]?\d{2,3}', berlaku_raw)
            berlaku = (in_dates[0] if in_dates else berlaku_raw).replace(' ', '-').replace('/', '-')
        elif dates:
            berlaku = dates[-1]

        # 9) Kota/Tanggal Terbit
        kota_terbit = self._guess_kota_terbit(joined_norm, regions_std.get("Provinsi"), label_out)
        tgl_terbit = (label_out.get('tanggal_terbit') or {}).get('value') or (dates[-1] if dates else None)

        final: Dict[str, Optional[str]] = {
            "Provinsi": regions_std.get("Provinsi"),
            "Kota": regions_std.get("Kota"),
            "NIK": nik,
            "Nama": nama,
            "Tempat_Tgl_Lahir": ttl,
            "Jenis_Kelamin": simple.get("Jenis_Kelamin"),
            "Golongan_Darah": simple.get("Golongan_Darah"),
            "Alamat": alamat,
            "RT_RW": rt_rw,
            "Kel_Desa": regions_std.get("Kel_Desa"),
            "Kecamatan": regions_std.get("Kecamatan"),
            "Agama": simple.get("Agama"),
            "Status_Perkawinan": simple.get("Status_Perkawinan"),
            "Pekerjaan": simple.get("Pekerjaan"),
            "Kewarganegaraan": simple.get("Kewarganegaraan"),
            "Berlaku_Hingga": berlaku,
            "Kota_Terbit": kota_terbit,
            "Tanggal_Terbit": tgl_terbit
        }

        # Final tidy
        def tidy(v: Optional[str]) -> Optional[str]:
            if v is None:
                return None
            x = re.sub(r'\s+', ' ', v).strip()
            return x if x else None

        for k in list(final.keys()):
            final[k] = tidy(final[k])

        return final


    # -----------------------
    # Public API
    # -----------------------
    def interpret(self, lines: List[str]) -> Dict[str, Any]:
        """End-to-end parse: tokenize → regex → label segment → repo-normalize → assembled dict.

        Why:
            Produce a consistent KTP payload from chaotic OCR input with deterministic rules.

        Args:
            lines: Raw OCR lines.

        Returns:
            Standardized KTP payload with best-effort normalized values (see top-level schema).
        """
        tokens = self.tokenize_lines(lines)
        joined_norm = " ".join(tok['norm'] for tok in tokens)
        raw_text = self._raw_text_for_regex(lines)
        regex_out = self.regex_extract(joined_norm, raw_text=raw_text)
        label_out = self.detect_labels_and_values(tokens, regex_out)
        return self.assemble_ktp_json(tokens, regex_out, label_out, raw_text=raw_text)


# -----------------------
# Example run (manual quick test)
# -----------------------
if __name__ == "__main__":
    # Load repos
    regions_repo: Optional[AdministrativeRegionsRepository] = None
    try:
        regions_repo = AdministrativeRegionsRepository(csv_file="kode_wilayah.csv", sqlite_path="wilayah.db")
    except Exception as e:
        logger.warning("Failed loading AdministrativeRegionsRepository: %s", e)

    refs_repo: Optional[KTPReferenceRepository] = None
    try:
        refs_repo = KTPReferenceRepository(sqlite_path="ktp_refs.db")
    except Exception as e:
        logger.warning("Failed loading KTPReferenceRepository: %s", e)

    # Sample OCR rows (kept from your original for sanity checks)
    dataset = [
        ["PROV1NSIDKI", "JAKARTA", "JAKARTABAR4T", "NIK 3171234567890123", "Nama M1RA SETIWAN",
         "TempatTgLLahr JAKRT4 18-02-1986", "JensKelmnn PERMPUAN GolDrh 8", "Almat JL PASTICEPATA7/66",
         "RTRW 007008", "KelDes PEGADUN6AN", "Kecamtan KALIDER3S", "Agm lSLM", "StusPrkawnan KAWlN",
         "Pekrjaan", "PEGAWAlSW4STA", "Kewarganegarn WNl", "BerlakHngga 22-02-2017", "JAKRTABRAT", "02-12-2O12"],
        ["PROV1NSI BANTEN","KOTA TANGERANG","NIK 3605123456789012","Nama R1ZA PUTRA","TempatTgLLahr TANGERANG 05-03-1992",
         "JensKelmn LAK1-LAK1 GolDrh A","Almat JL PANDEGLANG 45/12","RTRW 003005","KelDes PANDEMANG","Kecamtan CILEGON",
         "Agm lSLM","StusPrkawnan KAWlN","Pekrjaan SWASTA","Kewarganegaran", "WNI","BerlakHngga 10-07-2015"],
        ["PROVINSI JAWA BARAT","KABUPATEN BANDUNG","NIK 3276543210987654","Nama D3WI S1RAT","TempatTgLLahr CIHAMPELAS 20-11-1988",
         "JensKelmn PERMPUAN GolDrh B","Almat", "JL C1BIRU 77/8","RTRW", "002004","KelDes CIWIDEY","Kecamtan MARGAASIH",
         "Agm lSLM","StusPrkawnan BELUMKAWIN","Pekrjaan PNS","Kewarganegaran WNI","BerlakHngga 15-02-2012"],
        ["PROVINSI SULAWESI", "SELATAN","KOTA MAKASSAR","NIK 7371234567890123","Nama AH4MAD SYAH","TempatTgLLahr MAKASSAR 12-08-1990",
         "JensKelmn LAK1-LAK1 GolDrh O","Almat JL RANT3MBALU 9/10","RTRW 007009","KelDes TAMALATE","Kecamtan MARISO",
         "Agm lSLM","StusPrkawnan KAWlN","Pekrjaan SWASTA","Kewarganegaran", "WNI","BerlakHngga 01-01-2018"],
        ["PROVINSI", "DKI", "JAKARTA", "KOTA JAKARTA TIMUR", "NIK 3179876543210123", "Nama N1A PUTRI",
         "TempatTgLLahr C1LANGKAP 18-05-1995", "JensKelmn PERMPUAN GolDrh AB", "Almat JL CIPINANG 14/2",
         "RTRW 008009", "KelDes KRAMAT JATI", "Kecamtan JAKARTA TIMUR", "Agm lSLM", "StusPrkawnan KAWlN",
         "Pekrjaan PNS", "Kewarganegaran WNI", "BerlakHngga", "22-02-2020"],
        ["PROVINSI", "JAWA TIMUR","KABUPATEN", "SIDOARJO","NIK 3576543210987654","Nama FAHR1A RAHMA","TempatTgLLahr", "SIDOARJO 03-09-1991",
         "JensKelmn PERMPUAN GolDrh A","Almat JL KEDUNGJATI 5/11","RTRW 001003","KelDes WARU","Kecamtan SIDOARJO",
         "Agm lSLM","StusPrkawnan", "BELUMKAWIN","Pekrjaan SWASTA","Kewarganegaran WNI","BerlakHngga 12-12-2016"],
                 ["PROVINSI", "JAWA TIMUR","KABUPATEN SIDOARJO","NIK 3576543210987654","Nama FAHR1A RAHMA","TempatTgLLahr SIDOARJO 03-09-1991",
         "JensKelmn PERMPUAN GolDrh A","Almat JL KEDUNGJATI 5/11","RTRW 001003","KelDes WARU","Kecamtan SIDOARJO",
         "Agm lSLM","StusPrkawnan", "BELUMKAWIN","SWASTA","Kewarganegaran WNI","BerlakHngga 12-12-2016"]
    ]

    parser = OCRParser(regions=regions_repo, refs=refs_repo, debug=True)
    for rec in dataset:
        parsed = parser.interpret(rec)
        print(json.dumps(parsed, indent=2, ensure_ascii=False))
        print('-' * 80)
