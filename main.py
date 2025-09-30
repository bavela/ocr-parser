#!/usr/bin/env python3
"""
filepath: main.py

Refined OCRParser focused on robust field-label detection and safer OCR normalization.
Now integrates:
  - AdministrativeRegionsRepository for Province -> City/Kab -> District -> Village normalization (beam + mention filter).
  - KTPReferenceRepository for enumerated KTP fields (gender, religion, etc.) using dual-pass search:
      (1) label-window first, (2) global fallback scan over the whole token stream.
  - Month-name and numeric date parsing; RT/RW detection in multiple shapes (RT 007 / RW 008 / "007/008").

Why:
  Keep OCR parsing deterministic and maintainable by delegating vocabulary & fuzzy resolution to repositories,
  while the parser focuses on segmentation, safe normalization, and explainable heuristics.

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
from typing import List, Dict, Any, Optional, Tuple, Set

try:
    import Levenshtein  # only used for label detection distance fallbacks
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
      - resolving regions with a small hierarchy-aware beam search plus “mention filter”.

    Why:
      Keeps parsing logic small and robust; the repos own search/normalization,
      while the parser focuses on segmentation and assembly.

    Notes:
      • Deterministic: no ML models; string ops + SQLite-backed repos.
      • Typesafe: explicit Optional[str] on every output field.
    """

    # Numeric-dominant token cleanup (letters that often mean digits)
    NUMERIC_OCR_MAP = str.maketrans({
        'O': '0', 'Q': '0',
        'I': '1', 'L': '1', '|': '1',
        'Z': '2', 'S': '5'
    })

    # Alpha-dominant token cleanup (digits that sneak into words)
    REVERSE_OCR_MAP = str.maketrans({
        '0': 'O', '1': 'I', '3': 'E', '4': 'A', '5': 'S', '6': 'G', '7': 'T', '8': 'B', '9': 'G'
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
        greedy_pick: bool = True,
    ):
        """
        Prepare the parser with repositories and label/threshold configs.

        What:
            `greedy_pick=True` means we always accept the highest-scoring repo candidate
            for enums & regions even if the score is below the usual threshold.

        Why:
            Drops fewer fields to null on noisy OCR; we still keep label detection
            thresholds to avoid mis-segmenting values under wrong labels.
        """
        self.regions = regions
        self.refs = refs
        self.greedy_pick = greedy_pick

        label_pool = set(lbl.upper() for lbl in (field_labels or self.DEFAULT_LABELS))
        if self.refs:
            for f in self.refs.get_refs():
                label_pool.add(f.replace('_', ' ').upper())
        self.field_labels = sorted(label_pool)
        self._label_variants = [{'label': lbl, 'label_nospace': lbl.replace(' ', '')} for lbl in self.field_labels]

        # LOWER, more permissive defaults
        defaults = {
            "label_match_threshold": 0.68,
            "label_match_nospace_threshold": 0.60,
            "label_stop_threshold": 0.55,
            "label_searchrefs_threshold": 0.55,
            "value_lev_threshold": 0.50,
            "max_label_window": 3,
            "max_value_tokens": 12,
            "beam_k": 3,
        }
        self.thresholds = {**defaults, **(thresholds or {})}

    # -----------------------
    # Normalization & tokenization
    # -----------------------
    def _split_glued_token(self, tok: str) -> List[str]:
        """Keep alnum runs intact (preserve dates/numbers) to avoid over-splitting."""
        parts = re.findall(r'[A-Za-z0-9]+|[^A-Za-z0-9]+', tok)
        return [p for p in parts if re.search(r'[A-Za-z0-9]', p)]

    def _normalize_basic(self, tok: str) -> str:
        """Normalize for generic comparisons; does not apply label aliases.

        What:
            Uppercases, strips outer punctuation, collapses spaces, and applies digit/letter
            swaps depending on token makeup.

        Why:
            We want clean tokens for label/value heuristics, but must avoid changing the
            raw digits we’ll need when reconstructing values (we will use raw tokens for values).
        """
        if not tok:
            return ""
        t = unicodedata.normalize('NFKC', tok.strip())
        t = re.sub(r'^[^\w]+|[^\w]+$', '', t)
        letters = sum(ch.isalpha() for ch in t)
        digits = sum(ch.isdigit() for ch in t)

        # numeric-dominant -> fix O/I/| to digits; alpha-dominant -> fix digits to letters
        if digits >= max(1, len(t) // 2):
            t = t.translate(self.NUMERIC_OCR_MAP)
        else:
            t = t.translate(self.REVERSE_OCR_MAP)

        t = re.sub(r'\s+', ' ', t)
        return t.upper()

    def tokenize_lines(self, lines: List[str]) -> List[Dict[str, str]]:
        """Turn raw OCR lines into a token stream that preserves dates/numbers.

        What:
            Produces [{'raw': <raw>, 'norm': <normalized>}, ...] so we can use 'raw'
            for value reconstruction (digits intact) while relying on 'norm' for detection.

        Why:
            Using normalized tokens for label detection but raw tokens for values prevents
            digit->letter artifacts that break addresses and enums.
        """
        tokens: List[Dict[str, str]] = []
        for line in lines:
            raw_parts = re.split(r'\s+', line.strip())
            for p in raw_parts:
                if not p:
                    continue
                for sp in self._split_glued_token(p):
                    norm = self._normalize_basic(sp)
                    if norm:
                        tokens.append({'raw': sp, 'norm': norm})
        return tokens

    # -----------------------
    # Raw surface for regex (keep separators for dates)
    # -----------------------
    def _raw_text_for_regex(self, lines: List[str]) -> str:
        """Produce a raw-ish text line preserving separators so regex remains effective.

        Why:
            Regex needs `-`/`/` and spacings intact to fish dates/RT/RW reliably.
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
        """
        if not raw_text:
            return []
        out: List[str] = []

        # 1) numeric forms
        for m in self.DATE_NUMERIC_RE.findall(raw_text):
            out.append(m.replace(' ', '-').replace('/', '-'))

        # 2) month-name forms (ID)
        pattern = re.compile(r'\b([0-3]?\d)\s+([A-ZÄËÖÜA-Z]+)\s+([12]?\d{1,3})\b', re.IGNORECASE)
        for d, mon, y in pattern.findall(raw_text.upper()):
            mon_num = self.MONTHS_ID.get(mon.strip().upper())
            if not mon_num:
                continue
            yyyy = y
            if len(y) == 3:  # ambiguous
                continue
            out.append(f"{int(d):02d}-{mon_num}-{yyyy}")

        # dedup keep order
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
        """
        out: Dict[str, Any] = {}

        # NIK (prefer raw_text)
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

        # dates
        dates = self._regex_dates_from_raw(raw_text or "") if raw_text else []
        if dates:
            out['dates'] = [ParsedField(d, 0.90, 'regex_date').__dict__ for d in dates]

        # RT/RW explicit
        rt_re, rw_re = self.RT_RW_EXPLICIT_RE
        rt_m = rt_re.search(joined_norm)
        rw_m = rw_re.search(joined_norm)
        if rt_m:
            out['rt'] = ParsedField(rt_m.group(1), 0.95, 'regex_rt').__dict__
        if rw_m:
            out['rw'] = ParsedField(rm := rw_m.group(1), 0.95, 'regex_rw').__dict__

        # combined dd/dd
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
        """Uppercase nospace alnum key for quick comparisons."""
        return re.sub(r'[^A-Z0-9]+', '', unicodedata.normalize('NFKC', s or '').upper())

    @lru_cache(maxsize=2048)
    def _field_keys(self) -> Set[str]:
        names = set(self.field_labels)
        if self.refs:
            names = set(self.refs.get_refs())
        return {self._key(n) for n in names}

    def _is_label_at(self, tokens: List[Dict[str, str]], pos: int) -> Optional[Tuple[str, int, float]]:
        """Detect a field label beginning at `pos` and return (canonical_label, window_size, score).

        What:
            Slides 1..N token window and uses `refs.search_refs` to identify the label.
            Falls back to a small internal list if refs are not available.

        Why:
            Real-world OCR labels get glued/split/misspelled; repository-backed search
            is more stable than maintaining alias lists.
        """
        n = len(tokens)
        best: Optional[Tuple[str, int, float]] = None
        maxw = int(self.thresholds.get('max_label_window', 3))

        sr_thresh = float(self.thresholds.get('label_searchrefs_threshold', 0.70))
        lev_sp = float(self.thresholds.get('label_match_threshold', 0.82))
        lev_ns = float(self.thresholds.get('label_match_nospace_threshold', 0.72))

        for w in range(1, maxw + 1):
            if pos + w > n:
                break
            # build label-window text from normalized tokens (no alias expansion)
            window_tokens = [tokens[i]['norm'] for i in range(pos, pos + w)]
            joined_space = " ".join(window_tokens)
            joined_nospace = "".join(window_tokens)

            # 1) Prefer repository-backed label search when available
            if self.refs:
                hits = self.refs.search_refs(joined_space, k=1)
                if hits:
                    field_name, score = hits[0]
                    if score >= sr_thresh:
                        canon = field_name.replace('_', ' ').upper()
                        cand = (canon, w, float(score))
                        if best is None or cand[2] > best[2] or (cand[2] == best[2] and w > best[1]):
                            best = cand
                        continue

            # 2) Fallback to internal label pool (no repo case)
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
        """Segment tokens into (label -> raw value) pairs using repo-guided label detection,
        with stricter 'next-label' checks and small fallbacks to avoid empty value windows."""
        out: Dict[str, Any] = {}
        i = 0
        n = len(tokens)

        stop_keys = self._field_keys()
        stop_keys.update({self._key(x) for x in ("RT", "RW", "RTRW")})

        def is_hard_stop(tok_norm: str) -> bool:
            return self._key(tok_norm) in stop_keys

        base_stop = float(self.thresholds.get('label_stop_threshold', 0.66))
        max_val_tokens = int(self.thresholds.get('max_value_tokens', 10))
        maxw = int(self.thresholds.get('max_label_window', 3))

        def looks_like_label_at(k: int, *, current_label: Optional[str] = None) -> bool:
            """Stricter early-stop: require stronger confidence for 1-token hits and ignore ultra-short tokens."""
            hit = self._is_label_at(tokens, k)
            if not hit:
                return False
            canon, span, score = hit
            if current_label and canon == current_label:
                return False
            # tougher guard for single-token 'stops'
            if span == 1:
                # tiny tokens or low-confidence shouldn't stop a value window
                if len(tokens[k]['norm']) <= 3:
                    return False
                if score < (base_stop + 0.12):
                    return False
            return score >= base_stop

        while i < n:
            lab = self._is_label_at(tokens, i)
            if not lab:
                i += 1
                continue

            label_name, span, score = lab  # e.g., "JENIS KELAMIN"
            j = i + span
            val_raw_tokens: List[str] = []

            # capture window
            while j < n and len(val_raw_tokens) < max_val_tokens:
                if looks_like_label_at(j, current_label=label_name):
                    break
                if is_hard_stop(tokens[j]['norm']):
                    break
                val_raw_tokens.append(tokens[j]['raw'])  # preserve digits
                j += 1

            # if empty (common with low stop thresholds), apply tiny, safe fallbacks
            if not val_raw_tokens and j < n:
                want = 4 if label_name == "NAMA" else 1
                taken = 0
                j2 = j
                while j2 < n and taken < want:
                    if looks_like_label_at(j2, current_label=label_name):
                        break
                    if is_hard_stop(tokens[j2]['norm']):
                        break
                    # For NAMA, ensure the token looks name-like (has letters)
                    if label_name == "NAMA" and not any(ch.isalpha() for ch in tokens[j2]['raw']):
                        break
                    val_raw_tokens.append(tokens[j2]['raw'])
                    taken += 1
                    j2 += 1
                j = j2

            value_text = " ".join(val_raw_tokens).strip()
            if value_text:
                key = label_name.lower().replace(' ', '_')
                conf = min(0.95, 0.60 + 0.25 * float(score))
                out[key] = {
                    'value': value_text,
                    'confidence': conf,
                    'source': 'label_infer',
                    'evidence': {'label': label_name, 'ratio': float(score), 'span': span}
                }

            i = j

        return out

    # -----------------------
    # Enum extraction helpers
    # -----------------------
    def _scan_enum(self, text: str, *, kind: str) -> Optional[str]:
        """Extract a canonical enum value from a noisy value window (substring-aware)."""
        if not text or not self.refs:
            return None

        import unicodedata, re
        raw = unicodedata.normalize('NFKC', text)
        parts = [p for p in re.split(r'[\s,/|-]+', raw) if p]

        def best(search_fn, cands: List[str], min_score: float) -> Optional[str]:
            top_val, top_sc = None, 0.0
            for cand in cands:
                hits = search_fn(cand, k=1)
                if hits:
                    ref, sc = hits[0]
                    if sc > top_sc:
                        top_val, top_sc = ref.value, float(sc)
            # key change: allow greedy pick if enabled
            return top_val if (self.greedy_pick or top_sc >= min_score) else None

        if kind == "gender":
            cands: List[str] = []
            for i in range(len(parts)):
                cands.append(parts[i])
                if i + 1 < len(parts):
                    cands.append(f"{parts[i]} {parts[i+1]}")
                    cands.append(f"{parts[i]}-{parts[i+1]}")
            return best(self.refs.search_genders, cands, 0.55)

        if kind == "blood":
            norm_parts = [p.upper().replace('0', 'O').replace('8', 'B') for p in parts]
            cands = norm_parts[:4]
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
            cands = parts[:5]
            return best(self.refs.search_citizenships, cands, 0.50)

        return None

    def _global_enum_fallback(self, tokens_text: str) -> Dict[str, Optional[str]]:
        """Search the whole token stream for enum values when label windows fail."""
        out: Dict[str, Optional[str]] = {
            "Jenis_Kelamin": None, "Golongan_Darah": None, "Agama": None,
            "Status_Perkawinan": None, "Pekerjaan": None, "Kewarganegaraan": None
        }
        if not self.refs or not tokens_text:
            return out

        import re
        toks = [t for t in re.split(r'\s+', tokens_text) if t]

        def scan(min_score: float, search_fn) -> Optional[str]:
            best_val, best_sc = None, 0.0
            # size-1 and size-2 windows
            for i in range(len(toks)):
                cand = toks[i]
                hits = search_fn(cand, k=1)
                if hits:
                    ref, sc = hits[0]
                    if sc > best_sc:
                        best_val, best_sc = ref.value, float(sc)
                if i + 1 < len(toks):
                    cand2 = f"{toks[i]} {toks[i+1]}"
                    hits2 = search_fn(cand2, k=1)
                    if hits2:
                        ref2, sc2 = hits2[0]
                        if sc2 > best_sc:
                            best_val, best_sc = ref2.value, float(sc2)
            # key change: allow greedy pick if enabled
            return best_val if (self.greedy_pick or best_sc >= min_score) else None

        out["Jenis_Kelamin"]     = scan(0.60, self.refs.search_genders)
        out["Golongan_Darah"]    = scan(0.55, self.refs.search_blood_types)
        out["Agama"]             = scan(0.55, self.refs.search_religions)
        out["Status_Perkawinan"] = scan(0.55, self.refs.search_marital_statuses)
        out["Pekerjaan"]         = scan(0.58, self.refs.search_jobs)
        out["Kewarganegaraan"]   = scan(0.55, self.refs.search_citizenships)
        return out


    # -----------------------
    # Repo helpers
    # -----------------------
    def _pick_top(self, scored: List[Tuple[Any, float]], min_score: float) -> Optional[Any]:
        """Return best candidate, optionally ignoring `min_score` when `greedy_pick=True`.

        What:
            Chooses the highest-scoring item. If `greedy_pick` is enabled, we skip the
            threshold and return the top non-None item.

        Why:
            Aggressive recall for enums/regions when OCR is very noisy.

        Returns:
            The top item or None when list is empty / only None items exist.
        """
        if not scored:
            return None
        # prefer a real item over (None, 0.0)
        item, score = max(scored, key=lambda x: (x[0] is not None, x[1]))
        if self.greedy_pick:
            return item if item is not None else None
        return item if (item is not None and score >= min_score) else None


    @staticmethod
    def _token_set(text: str) -> Set[str]:
        """Token set (UPPER) for mention checks; drop very short bits."""
        toks = [t for t in re.split(r'[^A-Z0-9]+', unicodedata.normalize('NFKC', text.upper())) if t]
        return {t for t in toks if len(t) >= 3}

    def _mention_ok(self, name_off: str, text_tokens: Set[str]) -> bool:
        """Require at least one meaningful token of the candidate to appear in OCR text."""
        bad = {"PROVINSI", "KOTA", "KAB", "KABUPATEN", "ADM", "TIMUR", "BARAT", "UTARA", "SELATAN"}
        nts = {t for t in name_off.upper().split() if len(t) >= 3 and t not in bad}
        return any(t in text_tokens for t in nts)

    # -----------------------
    # Region normalization (beam + mention filter)
    # -----------------------
    def _normalize_region_hierarchy(
        self,
        raw_regions: Dict[str, str],
        raw_text: Optional[str] = None
    ) -> Dict[str, Optional[str]]:
        """Resolve Province -> City/Kab -> District -> Village using a small beam search + mention filter.

        Why:
            Local best-at-each-level can pick inconsistent parents; a tiny beam with a simple
            mention check (candidate tokens must occur in OCR text) yields robust, explainable paths.
        """
        if not self.regions:
            return {
                "Provinsi": (raw_regions.get('provinsi') or None),
                "Kota": (raw_regions.get('kota') or raw_regions.get('kabupaten') or None),
                "Kecamatan": (raw_regions.get('kecamatan') or None),
                "Kel_Desa": (raw_regions.get('kelurahan') or raw_regions.get('kel_desa') or raw_regions.get('desa') or None),
            }

        text_tokens = self._token_set(raw_text or "")
        beam_k = int(self.thresholds.get('beam_k', 2))

        # Queries prefer explicit captured strings; no explicit -> very cautious global use
        prov_q = (raw_regions.get('provinsi') or "").strip()
        kota_q = (raw_regions.get('kota') or raw_regions.get('kabupaten') or "").strip()
        kec_q = (raw_regions.get('kecamatan') or "").strip()
        kel_q = (raw_regions.get('kelurahan') or raw_regions.get('kel_desa') or raw_regions.get('desa') or "").strip()

        prov_hits = self.regions.search_provinces(prov_q or (raw_text or ""), k=max(beam_k * 3, 4))
        # Mention filter for province
        prov_hits = [(r, s) for (r, s) in prov_hits if isinstance(r, Region) and self._mention_ok(r.name_off, text_tokens)]
        if not prov_hits and prov_q:
            # if explicit prov_q existed but no mention-passing hits, try again without mention filter
            prov_hits = self.regions.search_provinces(prov_q, k=max(beam_k, 2))
        prov_hits = prov_hits[:max(1, beam_k)] or [(None, 0.0)]  # type: ignore

        dbg_paths: List[Tuple[Tuple[Optional[Region], Optional[Region], Optional[Region], Optional[Region]], float]] = []

        for prov, ps in prov_hits:
            prov_id = prov.region_id if isinstance(prov, Region) else None

            city_hits = self.regions.search_cities(kota_q or (raw_text or ""), province_id=prov_id, k=max(beam_k * 3, 4))
            city_hits = [(r, s) for (r, s) in city_hits if isinstance(r, Region) and self._mention_ok(r.name_off, text_tokens)]
            if not city_hits and kota_q:
                city_hits = self.regions.search_cities(kota_q, province_id=prov_id, k=max(beam_k, 2))
            city_hits = city_hits[:max(1, beam_k)] or [(None, 0.0)]  # type: ignore

            for city, cs in city_hits:
                city_id = city.region_id if isinstance(city, Region) else None

                dist_hits = self.regions.search_districts(kec_q or (raw_text or ""), city_id=city_id, k=max(beam_k * 3, 4))
                dist_hits = [(r, s) for (r, s) in dist_hits if isinstance(r, Region) and self._mention_ok(r.name_off, text_tokens)]
                if not dist_hits and kec_q:
                    dist_hits = self.regions.search_districts(kec_q, city_id=city_id, k=max(beam_k, 2))
                dist_hits = dist_hits[:max(1, beam_k)] or [(None, 0.0)]  # type: ignore

                for dist, ds in dist_hits:
                    dist_id = dist.region_id if isinstance(dist, Region) else None

                    vill_hits = self.regions.search_villages(kel_q or (raw_text or ""), district_id=dist_id, k=max(beam_k * 3, 4))
                    vill_hits = [(r, s) for (r, s) in vill_hits if isinstance(r, Region) and self._mention_ok(r.name_off, text_tokens)]
                    if not vill_hits and kel_q:
                        vill_hits = self.regions.search_villages(kel_q, district_id=dist_id, k=max(beam_k, 2))
                    vill_hits = vill_hits[:max(1, beam_k)] or [(None, 0.0)]  # type: ignore

                    for vill, vs in vill_hits:
                        score = 0.0
                        score += (ps or 0.0) + (cs or 0.0) + (ds or 0.0) + (vs or 0.0)
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
    # Simple enums via KTPReferenceRepository (dual pass + global fallback)
    # -----------------------
    def _standardize_simple_fields(
        self,
        label_out: Dict[str, Any],
        tokens_text: Optional[str] = None
    ) -> Dict[str, Optional[str]]:
        """Normalize enum-like fields to canonical KTP vocabulary.

        What:
            Converts noisy captured values for gender, blood type, religion, marital status,
            job, and citizenship into canonical forms using `KTPReferenceRepository`, with
            robust substring scan and a global full-stream fallback.

        Why:
            Value windows often carry extra tokens or OCR noise. Trying the full window first,
            then probing short subspans, and finally a global scan yields high recall without
            loosening thresholds globally.
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
            """Pick best enum candidate; honors greedy_pick to ignore `min_score`."""
            pick = self._pick_top(pairs, min_score)
            return pick.value if isinstance(pick, Ref) else None

        # Label-window pass + substring scan (unchanged)
        def norm_field(win: str, kind: str, search_fn, min_score: float) -> Optional[str]:
            if not win:
                return None
            # try whole window
            v = top_val(search_fn(win, k=3), min_score)
            if v:
                return v
            # subspans
            return self._scan_enum(win, kind=kind)


        jk = norm_field((label_out.get('jenis_kelamin', {}) or {}).get('value', ''), "gender", self.refs.search_genders, 0.60)
        gd_win = (label_out.get('golongan_darah', {}) or {}).get('value', '')
        gd = norm_field(gd_win, "blood", self.refs.search_blood_types, 0.55) if gd_win else None
        ag = norm_field((label_out.get('agama', {}) or {}).get('value', ''), "religion", self.refs.search_religions, 0.55)
        sp = norm_field((label_out.get('status_perkawinan', {}) or {}).get('value', ''), "marital", self.refs.search_marital_statuses, 0.55)
        pk = norm_field((label_out.get('pekerjaan', {}) or {}).get('value', ''), "job", self.refs.search_jobs, 0.58)
        kw = norm_field((label_out.get('kewarganegaraan') or label_out.get('kewarganegaran') or {}).get('value', ''), "citizenship", self.refs.search_citizenships, 0.55)

        # Global fallback if still missing
        if tokens_text:
            global_hits = self._global_enum_fallback(tokens_text)
            jk = jk or global_hits["Jenis_Kelamin"]
            gd = gd or global_hits["Golongan_Darah"]
            ag = ag or global_hits["Agama"]
            sp = sp or global_hits["Status_Perkawinan"]
            pk = pk or global_hits["Pekerjaan"]
            kw = kw or global_hits["Kewarganegaraan"]

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
            We reconstruct from RAW tokens so digits are intact; this only polishes spacing.
        """
        if not text:
            return text
        t = unicodedata.normalize('NFKC', text)
        t = re.sub(r'\s+', ' ', t).strip()
        t = re.sub(r'\bJL\b\.?', 'JL.', t, flags=re.IGNORECASE)
        t = re.sub(r'([A-Za-z])(\d)', r'\1 \2', t)
        t = re.sub(r'(\d)([A-Za-z])', r'\1 \2', t)
        return t.upper()

    @staticmethod
    def _strip_rt_rw_from_alamat(alamat: str) -> Tuple[str, Optional[str], Optional[str]]:
        """Remove explicit RT/RW fragments from address and return (clean, rt, rw).

        What:
            Strips **explicitly marked** RT/RW segments only (e.g., 'RT 007', 'RW 008', 'RTRW 007008').
            Does **not** treat generic 'NN/NN' house numbers as RT/RW to avoid false positives.
        """
        if not alamat:
            return alamat, None, None

        t = alamat

        m_rtrw = re.search(r'\bRTRW\W*0*(\d{3})0*(\d{3})\b', t, re.IGNORECASE)
        rt, rw = (m_rtrw.group(1), m_rtrw.group(2)) if m_rtrw else (None, None)
        if m_rtrw:
            t = re.sub(r'\bRTRW\W*\d{6}\b', '', t, flags=re.IGNORECASE)

        m_rt = re.search(r'\bRT\W*0*(\d{1,3})\b', t, re.IGNORECASE)
        m_rw = re.search(r'\bRW\W*0*(\d{1,3})\b', t, re.IGNORECASE)
        if m_rt:
            rt = rt or m_rt.group(1)
            t = re.sub(r'\bRT\W*\d{1,3}\b', '', t, flags=re.IGNORECASE)
        if m_rw:
            rw = rw or m_rw.group(1)
            t = re.sub(r'\bRW\W*\d{1,3}\b', '', t, flags=re.IGNORECASE)

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
        """
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
                return f"{place}, {all_dates[0]}"

        place = label_out.get('tempat_lahir', {}).get('value')
        date1 = (label_out.get('tanggal_lahir') or {}).get('value') or (label_out.get('tgl_lahir') or {}).get('value')
        if place and date1:
            place = self._clean_person_text(place).upper()
            return f"{place}, {date1.replace(' ', '-').replace('/', '-')}"

        if place and all_dates:
            place = self._clean_person_text(place).upper()
            return f"{place}, {all_dates[0]}"

        if raw_text and all_dates:
            m = re.search(r'([0-3]?\d[-/ ][0-1]?\d[-/ ][12]?\d{2,3})', raw_text)
            if m:
                left = raw_text[:m.start()].strip().split()
                cand = " ".join(left[-3:]) if left else ""
                if cand:
                    cand = self._clean_person_text(cand).upper()
                    return f"{cand}, {all_dates[0]}"
        return None
    
    def _fallback_name_from_tokens(self, tokens: List[Dict[str, str]]) -> Optional[str]:
        """
        Heuristic rescue for 'Nama' when the label window wasn't captured.
        Finds the token 'NAMA' and grabs up to 4 following name-like tokens,
        stopping at the next label/hard-stop.
        """
        if not tokens:
            return None

        n = len(tokens)
        stop_keys = self._field_keys()
        stop_keys.update({self._key(x) for x in ("RT", "RW", "RTRW")})
        base_stop = float(self.thresholds.get('label_stop_threshold', 0.66))

        def is_hard_stop(tok_norm: str) -> bool:
            return self._key(tok_norm) in stop_keys

        def looks_like_label_at(k: int) -> bool:
            hit = self._is_label_at(tokens, k)
            if not hit:
                return False
            _, span, score = hit
            # demand a bit more confidence for single-token "stops"
            if span == 1 and score < (base_stop + 0.12):
                return False
            return score >= base_stop

        for i in range(n):
            if tokens[i]['norm'] != 'NAMA':
                continue
            j = i + 1
            picked: List[str] = []
            while j < n and len(picked) < 4:
                if looks_like_label_at(j) or is_hard_stop(tokens[j]['norm']):
                    break
                raw = tokens[j]['raw']
                # must be name-ish: contain at least one letter; avoid pure numbers
                if any(ch.isalpha() for ch in raw):
                    picked.append(raw)
                else:
                    # stop if the very next token is non-namey (prevents scooping house numbers)
                    if not picked:
                        j += 1
                        continue
                    break
                j += 1
            if picked:
                return " ".join(picked)

        return None


    def _guess_kota_terbit(
        self,
        tokens_text: str,
        prov_region_name: Optional[str],
        label_out: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        """Guess 'Kota_Terbit' using explicit label first, then tail-of-text fallback."""
        if not self.regions:
            return None

        # Province scope -> region_id
        province_id: Optional[str] = None
        if prov_region_name:
            ph = self.regions.search_provinces(prov_region_name, k=1)
            top_prov = self._pick_top(ph, 0.58)  # type: ignore[arg-type]
            if isinstance(top_prov, Region):
                province_id = top_prov.region_id

        city_raw = (label_out or {}).get('kota_terbit', {}).get('value') if label_out else None
        if city_raw:
            hits = self.regions.search_cities(city_raw, province_id=province_id, k=1)
            best = self._pick_top(hits, 0.58)  # type: ignore[arg-type]
            if isinstance(best, Region):
                return best.name_off

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
        """Assemble the final KTP JSON by merging regex hits, label windows, and repo-normalization."""
        joined_norm = " ".join(t['norm'] for t in tokens)
        joined_raw = " ".join(t['raw'] for t in tokens)

        # 1) Regions (raw capture from labels) → normalization with mention filter
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
            raw_text=joined_norm
        )

        # 2) Simple fields via refs (dual pass + global fallback)
        simple = self._standardize_simple_fields(label_out, joined_norm)

        # 3) NIK
        nik = (regex_out.get('nik') or {}).get('value')

        # 4) Nama (prefer label capture; if missing/empty, use token-based fallback)
        nama_raw = (label_out.get('nama') or {}).get('value', '')
        if not nama_raw:
            guess = self._fallback_name_from_tokens(tokens)
            if guess:
                nama_raw = guess
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
        """End-to-end parse: tokenize -> regex -> label segment -> repo-normalize -> assembled dict.

        Why:
            Produce a consistent KTP payload from chaotic OCR input with deterministic rules.
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

    # Sample OCR rows (same as your dataset)
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

    parser = OCRParser(
        regions=regions_repo,
        refs=refs_repo,
        greedy_pick=True,  # always take the best enum/region candidate
        thresholds={
            # you can tweak further if needed
            "label_searchrefs_threshold": 0.52,
            "label_stop_threshold": 0.52,
        }
    )
    for rec in dataset:
        parsed = parser.interpret(rec)
        print(json.dumps(parsed, indent=2, ensure_ascii=False))
        print('-' * 80)
