#!/usr/bin/env python3
"""
filepath: main.py

Refined OCRParser focused on robust field-label detection and safer OCR normalization.
Now integrates:
  - AdministrativeRegionsRepository for Province → City/Kab → District → Village normalization (via search_*).
  - KTPReferenceRepository for enumerated KTP fields (gender, religion, etc.) using trigram search.

Why:
  Keep OCR parsing deterministic and maintainable by delegating vocabulary and fuzzy
  resolution to dedicated repositories with consistent normalization.

Author: Assistant (refined for user's OCR KTP pipeline)
"""
from __future__ import annotations

import re
import json
import logging
import unicodedata
from dataclasses import dataclass
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
      - detecting noisy labels (glued or spaced),
      - extracting obvious regex-able values (NIK, dates, RT/RW),
      - delegating fuzzy vocabulary matching to repositories.

    Why:
      Keeps parsing logic small and robust; the repos own search/normalization,
      while the parser focuses on segmentation and assembly.
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
        "KEWARGANEGARAAN", "KEWARGANEGARAN",
        "BERLAKU HINGGA",
        "KOTA TERBIT", "TANGGAL TERBIT"
    ]

    # Quick regexes for obvious numbers/dates/RT-RW
    NIK_RE = re.compile(r'\b[0-9OIl\|]{15,18}\b')
    DATE_RE = re.compile(r'\b[0-3]?\d[-/][0-1]?\d[-/][0-9OIl]{2,4}\b')
    RT_RW_EXPLICIT_RE = (
        re.compile(r'\bRT\W*0*?(\d{1,3})\b', re.IGNORECASE),
        re.compile(r'\bRW\W*0*?(\d{1,3})\b', re.IGNORECASE)
    )

    def __init__(
        self,
        *,
        regions: Optional[AdministrativeRegionsRepository] = None,
        refs: Optional[KTPReferenceRepository] = None,
        field_labels: Optional[List[str]] = None,
        thresholds: Optional[Dict[str, float]] = None
    ):
        """
        Prepare the parser with repositories and label/threshold configs.

        Args:
            regions (Optional[AdministrativeRegionsRepository]):
                Region repository used to normalize Provinsi/Kota/Kecamatan/Kel_Desa.
            refs (Optional[KTPReferenceRepository]):
                Reference repository used to normalize enumerated fields (gender, religion, etc.).
            field_labels (Optional[List[str]]):
                Override candidate label strings to detect in noisy OCR.
            thresholds (Optional[Dict[str, float]]):
                Tuning knobs for label matching and cutoffs. Sensible defaults provided.

        Returns:
            None
        """
        self.regions = regions
        self.refs = refs

        # Label candidates: merge defaults with repo field names if available
        label_pool = set(lbl.upper() for lbl in (field_labels or self.DEFAULT_LABELS))
        if self.refs:
            # Add canonical KTP field names as hints
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
            "max_value_tokens": 8
        }
        self.thresholds = {**defaults, **(thresholds or {})}

    # -----------------------
    # Normalization & tokenization
    # -----------------------
    def _split_glued_token(self, tok: str) -> List[str]:
        """Keep alnum runs intact (preserve dates/numbers) to avoid over-splitting."""
        parts = re.findall(r'[A-Za-z0-9]+|[^A-Za-z0-9]+', tok)
        return [p for p in parts if re.search(r'[A-Za-z0-9]', p)]

    def normalize_token(self, tok: str) -> str:
        """Normalize one token to mitigate OCR digit/letter swaps while preserving intent.

        Args:
            tok (str): Raw OCR token (possibly glued).

        Returns:
            str: Uppercased, minimally cleaned token to feed label/value heuristics.
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
        return t.upper()

    def tokenize_lines(self, lines: List[str]) -> List[Dict[str, str]]:
        """Turn raw OCR lines into a token stream that preserves dates/numbers.

        Args:
            lines (List[str]): Raw OCR lines.

        Returns:
            List[Dict[str, str]]: [{'raw': <raw>, 'norm': <normalized>}, ...]
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

        Args:
            lines (List[str]): Raw OCR lines.

        Returns:
            str: Lightly normalized text (NFKC, common O/I→digits, compact spaces).
        """
        txt = " ".join(lines)
        txt = unicodedata.normalize('NFKC', txt)
        trans = str.maketrans({'O': '0', 'I': '1', 'l': '1', '|': '1'})
        txt = txt.translate(trans)
        txt = re.sub(r'\s+', ' ', txt)
        return txt

    def _regex_dates_from_raw(self, raw_text: str) -> List[str]:
        """Collect plausible dates in multiple OCR styles to support TTL and issue dates.

        Args:
            raw_text (str): Lightly-normalized raw text.

        Returns:
            List[str]: Unique date-like strings (separators normalized to '-').
        """
        strict = re.findall(r'\b[0-3]?\d[-/][0-1]?\d[-/][12]\d{3}\b', raw_text)
        strict2 = re.findall(r'\b[0-3]?\d[-/][0-1]?\d[-/]\d{2}\b', raw_text)
        spacey = re.findall(r'\b[0-3]?\d\s[0-1]?\d\s[12]?\d{2,3}\b', raw_text)
        out = strict + strict2 + [re.sub(r'\s', '-', s) for s in spacey]
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

        Args:
            joined_norm (str): Normalized token text (joined with spaces).
            raw_text (Optional[str]): Raw-ish text that preserves separators.

        Returns:
            Dict[str, Any]: Partial extraction: nik, dates, rt, rw (when detected).
        """
        out: Dict[str, Any] = {}

        # NIK
        m = self.NIK_RE.search(joined_norm) or (re.search(r'\b[0-9OIl\|]{15,18}\b', raw_text or "") if raw_text else None)
        if m:
            r = m.group(0)
            cleaned = r.replace('O', '0').replace('I', '1').replace('l', '1').replace('|', '1')
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
            out['rw'] = ParsedField(rw_m.group(1), 0.95, 'regex_rw').__dict__

        # combined 6-digit RT/RW
        m6 = re.search(r'\b(\d{6})\b', joined_norm)
        if m6 and 'rt' not in out and 'rw' not in out:
            s = m6.group(1)
            out['rt'] = ParsedField(s[:3], 0.90, 'regex_rt_combined').__dict__
            out['rw'] = ParsedField(s[3:], 0.90, 'regex_rw_combined').__dict__

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

    def _is_label_at(self, tokens: List[Dict[str, str]], pos: int) -> Optional[Tuple[str, int, float]]:
        """Return (label, window_size, score) if a label is detected starting at pos."""
        n = len(tokens)
        best: Optional[Tuple[str, int, float]] = None
        maxw = self.thresholds['max_label_window']
        for w in range(1, maxw + 1):
            if pos + w > n:
                break
            window_tokens = [tokens[i]['norm'] for i in range(pos, pos + w)]
            joined_space = " ".join(window_tokens)
            joined_nospace = "".join(window_tokens)
            for var in self._label_variants:
                r_sp = self._lev_ratio(joined_space, var['label'])
                r_ns = self._lev_ratio(joined_nospace, var['label_nospace'])
                r = max(r_sp, r_ns)
                threshold = self.thresholds['label_match_threshold'] if r_sp >= r_ns else self.thresholds['label_match_nospace_threshold']
                if r >= threshold:
                    if best is None or r > best[2] or (r == best[2] and w > best[1]):
                        best = (var['label'], w, r)
        return best

    def detect_labels_and_values(self, tokens: List[Dict[str, str]], regex_out: Dict[str, Any]) -> Dict[str, Any]:
        """Segment the token stream into (label → raw value) pairs for later normalization.

        Args:
            tokens (List[Dict[str, str]]): Token stream.
            regex_out (Dict[str, Any]): Early regex hits to avoid mislabeling numeric spans.

        Returns:
            Dict[str, Any]: label_key → {value, confidence, source, evidence}
        """
        out: Dict[str, Any] = {}
        i = 0
        n = len(tokens)

        def looks_like_label_at(k: int) -> bool:
            lab = self._is_label_at(tokens, k)
            return (lab is not None and lab[2] >= self.thresholds['label_stop_threshold'])

        while i < n:
            lab = self._is_label_at(tokens, i)
            if lab:
                label_name, span, ratio = lab
                val_tokens = []
                j = i + span
                max_val_tokens = self.thresholds['max_value_tokens']
                while j < n and len(val_tokens) < max_val_tokens:
                    if looks_like_label_at(j):
                        break
                    val_tokens.append(tokens[j]['norm'])
                    j += 1
                value_text = " ".join(val_tokens).strip()
                key = label_name.lower().replace(' ', '_')
                if value_text:
                    out[key] = {
                        'value': value_text,
                        'confidence': min(0.95, 0.6 + 0.25 * ratio),
                        'source': 'label_infer',
                        'evidence': {'label': label_name, 'ratio': ratio, 'span': span}
                    }
                i = j
            else:
                i += 1
        return out

    # -----------------------
    # Repositories: helpers
    # -----------------------
    @staticmethod
    def _pick_top(scored: List[Tuple[Any, float]], min_score: float) -> Optional[Any]:
        """Choose top candidate above a threshold so we avoid weak matches.

        Args:
            scored (List[Tuple[Any, float]]): (item, score) sorted/unsorted.
            min_score (float): Minimum acceptable score.

        Returns:
            Optional[Any]: The item if its score meets the bar; otherwise None.
        """
        if not scored:
            return None
        item, score = max(scored, key=lambda x: x[1])
        return item if score >= min_score else None

    # -----------------------
    # Region normalization (via AdministrativeRegionsRepository.search_*)
    # -----------------------
    def _normalize_region_hierarchy(
        self,
        raw_regions: Dict[str, str],
        raw_text: Optional[str] = None
    ) -> Dict[str, Optional[str]]:
        """Resolve Province → City/Kab → District → Village via repo search (context-aware).

        Args:
            raw_regions (Dict[str, str]): Raw strings captured near labels (very noisy).
            raw_text (Optional[str]): Whole text as a fallback hint.

        Returns:
            Dict[str, Optional[str]]: {'Provinsi', 'Kota', 'Kecamatan', 'Kel_Desa'} normalized names.
        """
        if not self.regions:
            # Pass-through if repo isn't available
            return {
                "Provinsi": (raw_regions.get('provinsi') or None),
                "Kota": (raw_regions.get('kota') or raw_regions.get('kabupaten') or None),
                "Kecamatan": (raw_regions.get('kecamatan') or None),
                "Kel_Desa": (raw_regions.get('kelurahan') or raw_regions.get('kel_desa') or raw_regions.get('desa') or None),
            }

        # Province
        prov_raw = (raw_regions.get('provinsi') or '').strip()
        prov_region = None
        if prov_raw:
            prov_hit = self.regions.search_provinces(prov_raw, k=1)
            prov_region = self._pick_top(prov_hit, 0.58)  # type: ignore[arg-type]

        # City/Kab (prefer within province)
        kota_raw = (raw_regions.get('kota') or raw_regions.get('kabupaten') or '').strip()
        city_region = None
        if kota_raw:
            if isinstance(prov_region, Region):
                city_hit = self.regions.search_cities(kota_raw, province_id=prov_region.region_id, k=1)
                city_region = self._pick_top(city_hit, 0.58)  # type: ignore[arg-type]
            if not city_region:
                city_hit = self.regions.search_cities(kota_raw, province_id=None, k=1)
                city_region = self._pick_top(city_hit, 0.60)  # type: ignore[arg-type]

        # District (prefer within city)
        kec_raw = (raw_regions.get('kecamatan') or '').strip()
        dist_region = None
        if kec_raw:
            if isinstance(city_region, Region):
                dist_hit = self.regions.search_districts(kec_raw, city_id=city_region.region_id, k=1)
                dist_region = self._pick_top(dist_hit, 0.58)  # type: ignore[arg-type]
            if not dist_region:
                dist_hit = self.regions.search_districts(kec_raw, city_id=None, k=1)
                dist_region = self._pick_top(dist_hit, 0.60)  # type: ignore[arg-type]

        # Village (prefer within district)
        kel_raw = (raw_regions.get('kelurahan') or raw_regions.get('kel_desa') or raw_regions.get('desa') or '').strip()
        vill_region = None
        if kel_raw:
            if isinstance(dist_region, Region):
                vill_hit = self.regions.search_villages(kel_raw, district_id=dist_region.region_id, k=1)
                vill_region = self._pick_top(vill_hit, 0.60)  # type: ignore[arg-type]
            if not vill_region:
                vill_hit = self.regions.search_villages(kel_raw, district_id=None, k=1)
                vill_region = self._pick_top(vill_hit, 0.62)  # type: ignore[arg-type]

        # Whole-text fallback (broad guesses: province/city/district/village by best scores)
        if raw_text and (prov_region is None or city_region is None or dist_region is None or vill_region is None):
            # Province fallback
            if prov_region is None:
                ph = self.regions.search_provinces(raw_text, k=1)
                prov_region = self._pick_top(ph, 0.58)  # type: ignore[arg-type]
            # City fallback
            if city_region is None:
                ch = self.regions.search_cities(raw_text, province_id=(prov_region.region_id if isinstance(prov_region, Region) else None), k=1)
                city_region = self._pick_top(ch, 0.58)  # type: ignore[arg-type]
            # District fallback
            if dist_region is None:
                dh = self.regions.search_districts(raw_text, city_id=(city_region.region_id if isinstance(city_region, Region) else None), k=1)
                dist_region = self._pick_top(dh, 0.58)  # type: ignore[arg-type]
            # Village fallback
            if vill_region is None:
                vh = self.regions.search_villages(raw_text, district_id=(dist_region.region_id if isinstance(dist_region, Region) else None), k=1)
                vill_region = self._pick_top(vh, 0.60)  # type: ignore[arg-type]

        return {
            "Provinsi": (prov_region.name_off if isinstance(prov_region, Region) else None),
            "Kota": (city_region.name_off if isinstance(city_region, Region) else None),
            "Kecamatan": (dist_region.name_off if isinstance(dist_region, Region) else None),
            "Kel_Desa": (vill_region.name_off if isinstance(vill_region, Region) else None),
        }

    # -----------------------
    # Simple enums via KTPReferenceRepository
    # -----------------------
    def _standardize_simple_fields(self, label_out: Dict[str, Any]) -> Dict[str, Optional[str]]:
        """Normalize enumerated fields using KTPReferenceRepository so outputs match canonical vocab.

        Args:
            label_out (Dict[str, Any]): Raw values captured after label detection.

        Returns:
            Dict[str, Optional[str]]: Canonical values for enum-like fields (or None when absent).
        """
        if not self.refs:
            # Fallback: uppercase pass-through for minimal disruption
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
        jk_raw = (label_out.get('jenis_kelamin', {}) or {}).get('value', '')
        jk = top_val(self.refs.search_genders(jk_raw, k=3), 0.66) if jk_raw else None

        # Golongan Darah (often contains extra words; keep last token heuristic)
        gd_raw_all = (label_out.get('golongan_darah', {}) or {}).get('value', '')
        gd_token = gd_raw_all.split()[-1] if gd_raw_all else ''
        gd = top_val(self.refs.search_blood_types(gd_token or gd_raw_all, k=3), 0.60) if (gd_token or gd_raw_all) else None

        # Agama
        ag_raw = (label_out.get('agama', {}) or {}).get('value', '')
        ag = top_val(self.refs.search_religions(ag_raw, k=5), 0.60) if ag_raw else None

        # Status Perkawinan
        sp_raw = (label_out.get('status_perkawinan', {}) or {}).get('value', '')
        sp = top_val(self.refs.search_marital_statuses(sp_raw, k=5), 0.60) if sp_raw else None

        # Pekerjaan
        pk_raw = (label_out.get('pekerjaan', {}) or {}).get('value', '')
        pk = top_val(self.refs.search_jobs(pk_raw, k=5), 0.60) if pk_raw else None

        # Kewarganegaraan
        kw_raw = (label_out.get('kewarganegaraan') or label_out.get('kewarganegaran') or {}).get('value', '')
        kw = top_val(self.refs.search_citizenships(kw_raw, k=3), 0.60) if kw_raw else None

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

        Args:
            text (str): Raw noisy string.

        Returns:
            str: Cleaned display string with sensible capitalization.
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
        """Normalize address shapes; keep abbreviations and split letters/digits.

        Args:
            text (str): Raw address string.

        Returns:
            str: Uppercase cleaned address without RT/RW tails.
        """
        t = text.translate(OCRParser.REVERSE_OCR_MAP)
        t = re.sub(r'\s+', ' ', t).strip()
        t = re.sub(r'\bJL\b\.?', 'JL.', t, flags=re.IGNORECASE)
        t = re.sub(r'([A-Z])(\d)', r'\1 \2', t)
        return t.upper()

    @staticmethod
    def _strip_rt_rw_from_alamat(alamat: str) -> str:
        """Remove trailing RT/RW fragments that belong to a dedicated field.

        Args:
            alamat (str): Candidate full address.

        Returns:
            str: Address without RT/RW suffixes.
        """
        if not alamat:
            return alamat
        t = alamat
        t = re.sub(r'(?:\s|\b)(\d{3})[\/ ](\d{3})\s*$', '', t)
        t = re.sub(r'(?:\s|\b)RT\W*\d{1,3}\s*$', '', t, flags=re.IGNORECASE)
        t = re.sub(r'(?:\s|\b)RW\W*\d{1,3}\s*$', '', t, flags=re.IGNORECASE)
        t = re.sub(r'(?:\s|\b)RTRW\W*\d{6}\s*$', '', t, flags=re.IGNORECASE)
        return t.strip()

    @staticmethod
    def _fmt_rt_rw(rt: Optional[str], rw: Optional[str]) -> Optional[str]:
        """Format RT/RW in 'RRR/WWW' or None when missing, preserving numeric intent."""
        if not rt and not rw:
            return None
        rt3 = f"{int(rt):03d}" if rt and rt.isdigit() else (rt or "")
        rw3 = f"{int(rw):03d}" if rw and rw.isdigit() else (rw or "")
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

        Args:
            label_out (Dict[str, Any]): Detected label values.
            tokens_joined (str): Normalized joined tokens.
            all_dates (List[str]): Candidate dates found by regex.
            raw_text (Optional[str]): Raw-ish text as a final fallback.

        Returns:
            Optional[str]: 'PLACE, DD-MM-YYYY' or None.
        """
        # 1) direct combined label
        comb = label_out.get('tempat_tgl_lahir', {}).get('value')
        if comb:
            m = re.search(r'([0-3]?\d[-/ ][0-1]?\d[-/ ][12]?\d{2,3})', comb)
            if m:
                place = comb[:m.start()].strip()
                date = m.group(1).replace(' ', '-')
                place = self._clean_person_text(place).upper()
                return f"{place}, {date}"
            if all_dates:
                place = self._clean_person_text(comb).upper()
                return f"{place}, {all_dates[0]}"

        # 2) separate labels
        place = label_out.get('tempat_lahir', {}).get('value')
        date1 = (label_out.get('tanggal_lahir') or {}).get('value') or (label_out.get('tgl_lahir') or {}).get('value')
        if place and date1:
            place = self._clean_person_text(place).upper()
            return f"{place}, {date1.replace(' ', '-')}"

        if place and all_dates:
            place = self._clean_person_text(place).upper()
            return f"{place}, {all_dates[0]}"

        # 3) raw_text heuristic
        if raw_text and all_dates:
            m = re.search(r'([0-3]?\d[-/ ][0-1]?\d[-/ ][12]?\d{2,3})', raw_text)
            if m:
                left = raw_text[:m.start()].strip().split()
                cand = " ".join(left[-3:]) if left else ""
                if cand:
                    cand = self._clean_person_text(cand).upper()
                    return f"{cand}, {all_dates[0]}"
        return None

    def _guess_kota_terbit(self, tokens_text: str, prov_region_name: Optional[str]) -> Optional[str]:
        """Guess 'Kota_Terbit' from trailing tokens using region search for context.

        Args:
            tokens_text (str): Normalized joined tokens (tail examined).
            prov_region_name (Optional[str]): Province name to scope city search.

        Returns:
            Optional[str]: Best-guess issuing city/regency name.
        """
        if not self.regions:
            return None
        tail = " ".join(tokens_text.split()[-6:])
        province_id = None
        if prov_region_name:
            # map province name → search top1 → get region_id context
            ph = self.regions.search_provinces(prov_region_name, k=1)
            top_prov = self._pick_top(ph, 0.58)  # type: ignore[arg-type]
            if isinstance(top_prov, Region):
                province_id = top_prov.region_id

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
        """Build the final KTP dict by merging regex, label segmentation, and repo normalizations.

        Args:
            tokens (List[Dict[str, str]]): Normalized token stream.
            regex_out (Dict[str, Any]): Regex extraction results (NIK/dates/RT-RW).
            label_out (Dict[str, Any]): Raw label->value segmentation results.
            raw_text (Optional[str]): Raw-ish text as a fallback hint.

        Returns:
            Dict[str, Any]: Final standardized KTP payload keyed by your schema.
        """
        joined_norm = " ".join(t['norm'] for t in tokens)

        # 1) Regions (raw capture from labels)
        raw_regions = {
            'provinsi': (label_out.get('provinsi') or {}).get('value'),
            'kota': (label_out.get('kota') or {}).get('value'),
            'kabupaten': (label_out.get('kabupaten') or {}).get('value'),
            'kecamatan': (label_out.get('kecamatan') or {}).get('value'),
            'kelurahan': (label_out.get('kelurahan') or {}).get('value'),
            'kel_desa': (label_out.get('kel_desa') or {}).get('value'),
            'desa': (label_out.get('desa') or {}).get('value'),
        }
        regions_std = self._normalize_region_hierarchy({k: (v or "") for k, v in raw_regions.items()}, raw_text=raw_text)

        # 2) Simple fields via refs
        simple = self._standardize_simple_fields(label_out)

        # 3) NIK
        nik = (regex_out.get('nik') or {}).get('value')

        # 4) Nama
        nama_raw = (label_out.get('nama') or {}).get('value', '')
        nama = self._clean_person_text(nama_raw) if nama_raw else None

        # 5) Alamat (strip RT/RW tail)
        alamat_raw = (label_out.get('alamat') or {}).get('value', '')
        alamat = self._clean_address_text(alamat_raw) if alamat_raw else None
        if alamat:
            alamat = self._strip_rt_rw_from_alamat(alamat)

        # 6) RT/RW
        rt_val = (regex_out.get('rt') or {}).get('value')
        rw_val = (regex_out.get('rw') or {}).get('value')
        rt_rw = self._fmt_rt_rw(rt_val, rw_val)

        # 7) Tempat, Tgl Lahir
        dates = [d['value'] for d in (regex_out.get('dates') or [])]
        ttl = self._extract_ttl(label_out, joined_norm, dates, raw_text)

        # 8) Berlaku Hingga
        berlaku_raw = (label_out.get('berlaku_hingga') or {}).get('value')
        berlaku = None
        if berlaku_raw:
            in_dates = re.findall(r'[0-3]?\d[-/ ][0-1]?\d[-/ ][12]?\d{2,3}', berlaku_raw)
            berlaku = (in_dates[0] if in_dates else berlaku_raw).replace(' ', '-')
        elif dates:
            berlaku = dates[-1]

        # 9) Kota/Tanggal Terbit
        kota_terbit = self._guess_kota_terbit(joined_norm, regions_std.get("Provinsi"))
        tgl_terbit = dates[-1] if dates else None

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

        # Final tidy (collapse spaces; empty to None)
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

        Args:
            lines (List[str]): Raw OCR lines.

        Returns:
            Dict[str, Any]: Standardized KTP payload with best-effort normalized values.
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

    # Sample OCR rows
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

    parser = OCRParser(regions=regions_repo, refs=refs_repo)
    for rec in dataset:
        parsed = parser.interpret(rec)
        print(json.dumps(parsed, indent=2, ensure_ascii=False))
        print('-' * 80)



# Design notes — OCRParser hardening (no AI/ML, purely deterministic)

# Goal:
#     Extract KTP fields reliably from chaotic OCR text: missing labels, split/merged tokens,
#     and heavy OCR noise. This parser stays rule-based (string ops + SQLite-backed repos).

# 1) Preprocessing & tokenization
#     • Dual views:
#         - Keep original line groups for weak positional cues.
#         - Build a flattened token stream (one token per array item) and a character stream.
#       Why: handles ["PROVINSI","DKI","JAKARTA"] and "PROV1NSIDKI" equally well.
#     • Super-tokens:
#         - Synthesize bigrams/trigrams of adjacent tokens (e.g., "PROVINSI DKI").
#       Why: covers split labels/values without guessing boundaries.
#     • Token repair variants:
#         - For each token, precompute variants: nospace, dehyphenated, digits→letters and letters→digits,
#           plus short-edit neighbors (distance ≤ 1–2).
#       Why: catches "PROV1NSIDKI" ↔ "PROVINSI DKI" and "LAK1-LAK1" ↔ "LAKI-LAKI".

# 2) Label detection (robust to missing/split/merged)
#     • Trie + sliding window over flattened tokens AND super-tokens.
#       Use nospace and spaced forms for all label variants (defaults + fields from refs repo).
#     • Consider ":", ".", "—", and newlines as weak separators but not required.
#       Why: many scans omit colons; we still want a value window to the right/next.

# 3) Value capture when labels are broken or missing
#     • Proximity pairing:
#         - Given a detected label, capture the closest rightward value tokens (bounded window).
#         - Allow look-ahead across line breaks for split labels/values.
#     • Label-free guessing for enums:
#         - If a label is missing, scan globally through KTPReferenceRepository and take strong hits
#           (score ≥ τ) unless already consumed by another field.

# 4) Regions — let hierarchy do the work
#     • Top-down beam search:
#         1) Take top-2/3 provinces from search.
#         2) For each province, search cities scoped under it.
#         3) For each city, search districts; then villages.
#         4) Final path score = (local score per level) × (hierarchy consistency bonus).
#       Why: if one level is noisy, parent-child consistency rescues alignment.
#     • Whole-text backstop:
#         - If any level missing, also search with the entire flattened text; blend global and local scores.
#     • Conflict repair:
#         - If city ∉ province:
#             (a) switch to the city's true province if city score ≫ province score; OR
#             (b) keep province and re-search city constrained to it.

# 5) Enumerations via refs (when labels/values are split/absent)
#     • Dual pass:
#         - Label-guided pass: restrict to the next N tokens.
#         - Global pass: if nothing strong found, search the full stream via refs repo,
#           pick top ≥ τ (and not already used by another field).

# 6) Deterministic extracts with REGEX (preferred for non-repo fields)
#     • Use regular expressions for strongly structured fields not powered by repos:
#         - NIK: 16 digits, accept O/I/| as 0/1; normalize and clamp to first 16.
#         - Date: DD[-/ ]MM[-/ ]YYYY or DD[-/ ]MM[-/ ]YY; allow O/I→0/1; normalize spaces→'-'.
#         - RT/RW: explicit "RT ddd" / "RW ddd", or combined 6 digits → "ddd/ddd".
#       Rationale: regex is concise, fast, and accurate for these constrained formats.
#       (Optional fallback: tiny finite-state scanners if you ever want regex-free builds.)

# 7) Ambiguity & confidence
#     • Confidence by source:
#         - regex extract > label-guided repo match > global repo match.
#     • Cross-field consistency boosts:
#         - +bonus if city∈province, district∈city, village∈district; -penalty on mismatches.
#     • Keep alternates internally; emit top pick with confidence (optionally expose suggestions in UI).

# 8) Performance
#     • Aggressive caching:
#         - LRU for normalized tokens, nospace forms, trigrams, and repeated repo queries.
#         - Deduplicate repeated tokens/phrases before searching repos.
#     • Smaller candidate sets:
#         - Use SQLite prefix + trigram LIKE prefilters; target ≤ 200 candidates/query; ≤ 3 queries/level.
#         - For enums, even tighter: ≤ 50 candidates before scoring.
#     • Fast distances:
#         - Use python-Levenshtein; precompute nospace(query) once for batch scoring.
#     • Early exits:
#         - If a field is high-confidence, skip global searches.
#     • Parallelize independent searches:
#         - Provinces/enums can run in a thread pool; WAL mode handles concurrent reads.

# 9) Real-scan QoL
#     • If OCR provides bounding boxes: prefer same-line-right of label, then next line (big accuracy win).
#     • Alias dictionary for frequent confusions:
#         - e.g., "PERMPUAN"→"PEREMPUAN", "KAWlN"→"KAWIN", "WNl"→"WNI", "AGM"→"AGAMA".
#     • Domain heuristics:
#         - NIK prefix hints province/city; bias the beam.
#         - Strip RT/RW embedded in Alamat and reuse if RT/RW fields are missing.

# 10) Ingestion shape
#     • Yes: split into atomic strings per item AND also keep a re-joined variant.
#       - Atomic items help super-token recombination.
#       - Original line grouping keeps weak positional cues.

# 11) Explicit edge cases
#     • Repeated labels / out of order → keep the best-confidence (or latest with higher confidence).
#     • Value-before-label → allow a short look-back window.
#     • Two values glued → accept '/', '-', and nospace merges (e.g., LAK1-LAK1, RTRW007008).
#     • District present but city/province missing → resolve district globally, infer its parents.

# 12) Practical thresholds (starting points)
#     • label_match_threshold ≈ 0.80; label_nospace_threshold ≈ 0.74
#     • enums: accept ≥ 0.60 (global), ≥ 0.66 (label-window)
#     • regions: province ≥ 0.58; city ≥ 0.60; district ≥ 0.60; village ≥ 0.62
#     • hierarchy bonus +0.05 per valid parent link; mismatch penalty −0.08

