#!/usr/bin/env python3
"""
filepath: main.py

Refined OCRParser focused on robust field-label detection and safer OCR normalization.
Now integrates AdministrativeRegions to normalize province/city/district/village.

Requirements:
    pip install python-Levenshtein
(py-bk-tree optional; code falls back to brute-force)

Author: Assistant (refined for user's OCR KTP pipeline)
"""
from __future__ import annotations

import re
import json
import logging
import unicodedata

from typing import List, Dict, Any, Optional, Tuple

try:
    import Levenshtein
except Exception as e:
    raise ImportError("python-Levenshtein is required. Install with: pip install python-Levenshtein") from e

# Optional BK-tree for scale (not required)
try:
    import pybktree
    _HAS_PYBK = True
except Exception:
    _HAS_PYBK = False

# Administrative regions (your class)
try:
    from administrative_regions import AdministrativeRegions
except Exception:
    AdministrativeRegions = None  # type: ignore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OCRParser:
    """
    OCRParser: normalized pipeline for OCR token -> structured field extraction.
    Focus:
        - accurate field-label detection (single or multi-token, glued or separated)
        - region normalization via AdministrativeRegions (Province -> City/Kab -> District -> Village)
    """

    # Conservative OCR mappings for numeric-like tokens
    NUMERIC_OCR_MAP = str.maketrans({
        'O': '0', 'Q': '0',
        'I': '1', 'L': '1', '|': '1',
        'Z': '2', 'S': '5'
    })

    # For "texty" tokens we sometimes need the reverse (turn digits into letters) – carefully.
    # Applied only to names/streets/places (NOT to NIK/dates).
    REVERSE_OCR_MAP = str.maketrans({
        '0': 'O', '1': 'I', '3': 'E', '4': 'A', '5': 'S', '6': 'G', '7': 'T', '8': 'B'
    })

    # Label defaults (uppercase)
    DEFAULT_LABELS = [
        "NAMA", "NIK", "ALAMAT",
        "TEMPAT LAHIR", "TANGGAL LAHIR", "TGL LAHIR", "TEMPAT TGL LAHIR",
        "PROVINSI", "KABUPATEN", "KOTA", "KECAMATAN", "KELURAHAN", "DESA", "KEL DESA",
        "RT", "RW",
        "AGAMA", "PEKERJAAN",
        "JENIS KELAMIN", "GOLONGAN DARAH",
        "STATUS PERKAWINAN",
        "KEWARGANEGARAAN", "KEWARGANEGARAN",
        "BERLAKU HINGGA"
    ]

    # Regex
    NIK_RE = re.compile(r'\b[0-9OIl\|]{15,18}\b')
    DATE_RE = re.compile(r'\b[0-3]?\d[-/][0-1]?\d[-/][0-9OIl]{2,4}\b')
    RT_RW_EXPLICIT_RE = (
        re.compile(r'\bRT\W*0*?(\d{1,3})\b', re.IGNORECASE),
        re.compile(r'\bRW\W*0*?(\d{1,3})\b', re.IGNORECASE)
    )

    def __init__(self,
                 field_labels: Optional[List[str]] = None,
                 value_dicts: Optional[Dict[str, List[str]]] = None,
                 thresholds: Optional[Dict[str, float]] = None,
                 regions: Optional[AdministrativeRegions] = None):
        """
        Initialize parser with optional custom labels, dictionaries, thresholds, and AdministrativeRegions.

        Args:
            regions: instance of AdministrativeRegions; used to normalize Provinsi/Kota/Kecamatan/Kel/Des.
        """
        self.field_labels = [lbl.upper() for lbl in (field_labels or self.DEFAULT_LABELS)]
        self._label_variants = [{'label': lbl, 'label_nospace': lbl.replace(' ', '')} for lbl in self.field_labels]

        # Minimal sample value dicts (uppercased)
        sample_value_dicts = {
            "province": ["DKI JAKARTA", "JAWA BARAT", "JAWA TIMUR", "BANTEN", "SULAWESI SELATAN"],
            "religion": ["ISLAM", "KRISTEN", "KATHOLIK", "HINDU", "BUDDHA", "KONGHUCU"],
            "nationality": ["WNI", "WNA"],
            "gender": ["LAKI-LAKI", "PEREMPUAN"],
            "blood": ["A", "B", "AB", "O"]
        }
        self.value_dicts = {
            k: [v.upper() for v in (value_dicts or {}).get(k, sample_value_dicts[k])]
            for k in sample_value_dicts
        }

        defaults = {
            "label_match_threshold": 0.82,
            "label_match_nospace_threshold": 0.75,
            "label_stop_threshold": 0.70,
            "value_lev_threshold": 0.66,
            "max_label_window": 3,
            "max_value_tokens": 8  # allow a bit more for addresses
        }
        self.thresholds = {**defaults, **(thresholds or {})}

        self._bk_trees = {}
        if _HAS_PYBK:
            for name, vals in self.value_dicts.items():
                if vals:
                    try:
                        from pybktree import BKTree  # type: ignore
                        self._bk_trees[name] = BKTree(Levenshtein.distance, vals)
                    except Exception:
                        pass

        # Regions
        self.regions = regions

    # -----------------------
    # Normalization & Tokenization
    # -----------------------
    def _is_numeric_like(self, s: str) -> bool:
        digits = sum(ch.isdigit() for ch in s)
        return digits >= max(1, len(s) // 2)

    def normalize_token(self, tok: str) -> str:
        if not tok:
            return ""
        t = unicodedata.normalize('NFKC', tok.strip())
        t = re.sub(r'^[^\w]+|[^\w]+$', '', t)
        if self._is_numeric_like(t):
            t = t.translate(self.NUMERIC_OCR_MAP)
        t = re.sub(r'\s+', ' ', t)
        return t.upper()

    def _split_glued_token(self, tok: str) -> List[str]:
        parts = re.findall(r'[A-Za-z]+|\d+|[^A-Za-z0-9]+', tok)
        parts = [p for p in parts if p and not re.fullmatch(r'[^A-Za-z0-9]+', p)]
        return parts if parts else [tok]

    def tokenize_lines(self, lines: List[str]) -> List[Dict[str, str]]:
        tokens: List[Dict[str, str]] = []
        for line in lines:
            raw_parts = re.split(r'[\t\n\r\f\v]+|\s+|[\/,;:]', line.strip())
            for p in raw_parts:
                if not p:
                    continue
                for sp in self._split_glued_token(p):
                    norm = self.normalize_token(sp)
                    if norm:
                        tokens.append({'raw': sp, 'norm': norm})
        return tokens

    # -----------------------
    # Regex deterministic extraction
    # -----------------------
    def regex_extract(self, joined_norm: str) -> Dict[str, Any]:
        out: Dict[str, Any] = {}

        # NIK
        m = self.NIK_RE.search(joined_norm)
        if m:
            r = m.group(0)
            cleaned = r.replace('O', '0').replace('I', '1').replace('l', '1').replace('|', '1')
            digits = re.sub(r'\D', '', cleaned)
            if len(digits) >= 16:
                digits = digits[:16]
            if len(digits) == 16:
                out['nik'] = {'value': digits, 'confidence': 0.99, 'source': 'regex_nik'}

        # dates
        dates = [d.replace('O', '0').replace('I', '1').replace('l', '1').replace('|', '1') for d in self.DATE_RE.findall(joined_norm)]
        if dates:
            out['dates'] = [{'value': d, 'confidence': 0.90, 'source': 'regex_date'} for d in dates]

        # RT/RW explicit
        rt_re, rw_re = self.RT_RW_EXPLICIT_RE
        rt_m = rt_re.search(joined_norm)
        rw_m = rw_re.search(joined_norm)
        if rt_m:
            out['rt'] = {'value': rt_m.group(1), 'confidence': 0.95, 'source': 'regex_rt'}
        if rw_m:
            out['rw'] = {'value': rw_m.group(1), 'confidence': 0.95, 'source': 'regex_rw'}

        # combined 6-digit RT/RW
        m6 = re.search(r'\b(\d{6})\b', joined_norm)
        if m6 and 'rt' not in out and 'rw' not in out:
            s = m6.group(1)
            out['rt'] = {'value': s[:3], 'confidence': 0.9, 'source': 'regex_rt_combined'}
            out['rw'] = {'value': s[3:], 'confidence': 0.9, 'source': 'regex_rw_combined'}

        return out

    # -----------------------
    # Label detection
    # -----------------------
    def _lev_ratio(self, a: str, b: str) -> float:
        if not a or not b:
            return 0.0
        dist = Levenshtein.distance(a.upper(), b.upper())
        return max(0.0, 1.0 - (dist / max(len(a), len(b))))

    def _is_label_at(self, tokens: List[Dict[str, str]], pos: int) -> Optional[Tuple[str, int, float]]:
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
    # Value dict matching utils
    # -----------------------
    def _find_closest_in_list(self, token: str, options: List[str], threshold: float) -> Optional[str]:
        token_up = token.upper()
        if not options:
            return None
        best, best_ratio = None, -1.0
        for opt in options:
            r = self._lev_ratio(token_up, opt.upper())
            if r > best_ratio:
                best, best_ratio = opt, r
        return best if best and best_ratio >= threshold else None

    def _find_closest_in_dict(self, token: str, dict_name: str) -> Optional[Dict[str, Any]]:
        token_up = token.upper()
        vals = self.value_dicts.get(dict_name, [])
        if not vals:
            return None

        # BK-tree if available
        if dict_name in self._bk_trees:
            tree = self._bk_trees[dict_name]
            max_dist = max(1, int(0.25 * max(1, len(token_up))))
            try:
                cand = tree.find(token_up, max_dist)
                if not cand:
                    return None
                dist, val = min(cand, key=lambda x: x[0])
                ratio = 1 - dist / max(len(token_up), len(val))
                if ratio >= self.thresholds['value_lev_threshold']:
                    return {'value': val, 'ratio': ratio, 'distance': dist}
            except Exception:
                pass

        # brute force
        best_val, best_dist = None, 10**9
        for v in vals:
            d = Levenshtein.distance(token_up, v.upper())
            if d < best_dist:
                best_dist = d
                best_val = v
        if best_val is None:
            return None
        ratio = 1 - (best_dist / max(len(token_up), len(best_val)))
        if ratio >= self.thresholds['value_lev_threshold']:
            return {'value': best_val, 'ratio': ratio, 'distance': best_dist}
        return None

    # -----------------------
    # Region normalization via AdministrativeRegions
    # -----------------------
    def _normalize_region_hierarchy(self, raw_regions: Dict[str, str]) -> Dict[str, Optional[str]]:
        """
        Normalize: Provinsi -> Kota/Kab -> Kecamatan -> Kel/Desa using strong fuzzy matchers.
        Accepts very noisy/glued OCR tokens (e.g., JAKARTABAR4T, Kecamtan).
        """
        if not self.regions:
            return {
                "Provinsi": raw_regions.get('provinsi') or None,
                "Kota": (raw_regions.get('kota') or raw_regions.get('kabupaten')) or None,
                "Kecamatan": raw_regions.get('kecamatan') or None,
                "Kel_Desa": raw_regions.get('kelurahan') or raw_regions.get('kel_desa') or raw_regions.get('desa') or None,
            }

        # Raw upper strings
        prov_raw = (raw_regions.get('provinsi') or '').upper()
        kota_raw = ((raw_regions.get('kota') or '') or (raw_regions.get('kabupaten') or '')).upper()
        kec_raw  = (raw_regions.get('kecamatan') or '').upper()
        kel_raw  = ((raw_regions.get('kelurahan') or '') or (raw_regions.get('kel_desa') or '') or (raw_regions.get('desa') or '')).upper()

        # 1) Province
        prov = self.regions.find_best_province(prov_raw) if prov_raw else None

        # 2) City/Kabupaten (try within province first, then global)
        city = None
        if kota_raw:
            city = self.regions.find_best_city(kota_raw, province_name=prov) or self.regions.find_best_city(kota_raw, province_name=None)

        # 3) Kecamatan (try within found city first, then global)
        kec = None
        if kec_raw:
            kec = self.regions.find_best_district(kec_raw, city_name=city) or self.regions.find_best_district(kec_raw, city_name=None)

        # 4) Kelurahan/Desa (try within found kecamatan first, then global)
        kel = None
        if kel_raw:
            kel = self.regions.find_best_village(kel_raw, district_name=kec) or self.regions.find_best_village(kel_raw, district_name=None)

        return {
            "Provinsi": prov,
            "Kota": city,
            "Kecamatan": kec,
            "Kel_Desa": kel
        }


    # -----------------------
    # Post-processing (cleanup to standard schema)
    # -----------------------
    @staticmethod
    def _clean_person_text(text: str) -> str:
        # reverse OCR digit→letter cautiously, collapse spaces, title-case names
        t = text.translate(OCRParser.REVERSE_OCR_MAP)
        t = re.sub(r'\s+', ' ', t).strip()
        # expand common consonant-loss like "PERMPUAN" -> "PEREMPUAN"
        t = re.sub(r'\bPERMPUAN\b', 'PEREMPUAN', t)
        # title-case but keep all-caps acronyms like WNI
        words = []
        for w in t.split(' '):
            if len(w) <= 3 and w.isupper():
                words.append(w)
            else:
                words.append(w.capitalize())
        return " ".join(words)

    @staticmethod
    def _clean_address_text(text: str) -> str:
        t = text.translate(OCRParser.REVERSE_OCR_MAP)
        t = re.sub(r'\s+', ' ', t).strip()
        # Normalize "JL" to "JL."
        t = re.sub(r'\bJL\b\.?', 'JL.', t, flags=re.IGNORECASE)
        # Fix common glued tokens (e.g., CEPATA7 -> CEPAT A7)
        t = re.sub(r'([A-Z])(\d)', r'\1 \2', t)
        return t.upper()

    @staticmethod
    def _fmt_rt_rw(rt: Optional[str], rw: Optional[str]) -> Optional[str]:
        if not rt and not rw:
            return None
        rt3 = f"{int(rt):03d}" if rt and rt.isdigit() else (rt or "")
        rw3 = f"{int(rw):03d}" if rw and rw.isdigit() else (rw or "")
        if rt3 and rw3:
            return f"{rt3}/{rw3}"
        return rt3 or rw3 or None

    @staticmethod
    def _pick_best_date(dates: List[str]) -> Optional[str]:
        if not dates:
            return None
        # prefer YYYY with 4 digits; keep original order otherwise
        def score(d: str) -> Tuple[int, int]:
            y = d.split('-')[-1]
            return (1 if len(y) == 4 else 0, 0)
        return sorted(dates, key=score, reverse=True)[0]

    def _extract_ttl(self, label_out: Dict[str, Any], tokens_joined: str, all_dates: List[str]) -> Optional[str]:
        """
        Try to form "Tempat, DD-MM-YYYY"
        Sources:
            - "TEMPAT TGL LAHIR" label value (often "CITY DD-MM-YYYY")
            - "TEMPAT LAHIR" + a date nearby
            - fallback: best city-like token + best date
        """
        # 1) direct combined label
        comb = label_out.get('tempat_tgl_lahir', {}).get('value')
        if comb:
            # split last date if present
            m = self.DATE_RE.search(comb)
            if m:
                place = comb[:m.start()].strip()
                date = m.group(0)
                place = self._clean_person_text(place).upper()
                return f"{place}, {date}"
            # otherwise, try append best detected date
            best_date = self._pick_best_date(all_dates)
            if best_date:
                place = self._clean_person_text(comb).upper()
                return f"{place}, {best_date}"

        # 2) separate labels
        place = label_out.get('tempat_lahir', {}).get('value')
        date1 = label_out.get('tanggal_lahir', {}).get('value') or label_out.get('tgl_lahir', {}).get('value')
        if place and date1:
            place = self._clean_person_text(place).upper()
            return f"{place}, {date1}"

        # 3) regex date anywhere + guess place from preceding token
        best_date = self._pick_best_date(all_dates)
        if best_date:
            # naive: take up to 3 tokens before first date match as place
            m = self.DATE_RE.search(tokens_joined)
            if m:
                left = tokens_joined[:m.start()].strip().split()
                cand = " ".join(left[-3:]) if left else ""
                if cand:
                    cand = self._clean_person_text(cand).upper()
                    return f"{cand}, {best_date}"
        return None

    def _standardize_simple_fields(self, label_out: Dict[str, Any]) -> Dict[str, Optional[str]]:
        # Gender
        gender_raw = (label_out.get('jenis_kelamin', {}) or {}).get('value', '')
        gender = None
        if gender_raw:
            hit = self._find_closest_in_dict(gender_raw, 'gender')
            gender = (hit or {}).get('value')

        # Blood type
        blood_raw = (label_out.get('golongan_darah', {}) or {}).get('value', '')
        blood = None
        if blood_raw:
            # choose the last token (A/B/AB/O) if a sentence slipped in
            tok = blood_raw.split()[-1]
            hit = self._find_closest_in_dict(tok, 'blood')
            blood = (hit or {}).get('value')

        # Agama
        agm_raw = (label_out.get('agama', {}) or {}).get('value', '')
        agama = None
        if agm_raw:
            hit = self._find_closest_in_dict(agm_raw, 'religion')
            agama = (hit or {}).get('value') or agm_raw.upper()

        # Kewarganegaraan
        kew_raw = (label_out.get('kewarganegaraan', {}) or label_out.get('kewarganegaran', {}) or {}).get('value', '')
        kew = None
        if kew_raw:
            hit = self._find_closest_in_dict(kew_raw, 'nationality')
            kew = (hit or {}).get('value') or kew_raw.upper()

        # Status Perkawinan (just uppercase clean)
        status_raw = (label_out.get('status_perkawinan', {}) or {}).get('value', '')
        status = status_raw.upper() if status_raw else None

        # Pekerjaan (uppercase)
        job_raw = (label_out.get('pekerjaan', {}) or {}).get('value', '')
        job = job_raw.upper() if job_raw else None

        return {
            "Jenis_Kelamin": gender,
            "Golongan_Darah": blood,
            "Agama": agama,
            "Status_Perkawinan": status,
            "Pekerjaan": job,
            "Kewarganegaraan": kew
        }

    def _guess_kota_terbit(self, tokens_text: str, prov_guess: Optional[str]) -> Optional[str]:
        """
        Use AdministrativeRegions to guess 'Kota_Terbit' from trailing tokens
        (often appears near the end, e.g., 'JAKRTABRAT' -> 'JAKARTA BARAT').
        """
        if not self.regions:
            return None

        # Collect candidate text from the last ~6 tokens
        tail = tokens_text.split()[-6:]
        tail_text = " ".join(tail)
        # Try to match against cities of the province first
        if prov_guess:
            cities = self.regions.get_cities(prov_guess)
            hit = self._find_closest_in_list(tail_text, cities, 0.6)
            if hit:
                return hit
        # Fallback across all cities
        all_cities: List[str] = []
        for _, d in self.regions.cities.items():
            all_cities.extend(list(d.values()))
        return self._find_closest_in_list(tail_text, all_cities, 0.7)

    # -----------------------
    # Assembly to your standard KTP JSON schema
    # -----------------------
    def assemble_ktp_json(self, tokens: List[Dict[str, str]], regex_out: Dict[str, Any], label_out: Dict[str, Any]) -> Dict[str, Any]:
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
        regions_std = self._normalize_region_hierarchy({k: (v or "") for k, v in raw_regions.items()})

        # 2) Simple fields
        simple = self._standardize_simple_fields(label_out)

        # 3) NIK
        nik = (regex_out.get('nik') or {}).get('value')

        # 4) Nama
        nama_raw = (label_out.get('nama') or {}).get('value', '')
        nama = self._clean_person_text(nama_raw) if nama_raw else None

        # 5) Alamat
        alamat_raw = (label_out.get('alamat') or {}).get('value', '')
        alamat = self._clean_address_text(alamat_raw) if alamat_raw else None

        # 6) RT/RW
        rt_val = (regex_out.get('rt') or {}).get('value')
        rw_val = (regex_out.get('rw') or {}).get('value')
        rt_rw = self._fmt_rt_rw(rt_val, rw_val)

        # 7) Tempat, Tgl Lahir
        dates = [d['value'] for d in (regex_out.get('dates') or [])]
        ttl = self._extract_ttl(label_out, joined_norm, dates)

        # 8) Berlaku Hingga
        berlaku_raw = (label_out.get('berlaku_hingga') or {}).get('value')
        berlaku = None
        if berlaku_raw:
            # pick the best date looking inside it
            in_dates = self.DATE_RE.findall(berlaku_raw)
            berlaku = in_dates[0].replace('O', '0').replace('I', '1') if in_dates else berlaku_raw

        # 9) Kota/Tanggal Terbit (heuristic)
        kota_terbit = self._guess_kota_terbit(joined_norm, regions_std.get("Provinsi"))
        tgl_terbit = None
        if dates:
            # Often the last date in lines is issuance date; choose the last found date.
            tgl_terbit = dates[-1]

        # Build final JSON using your exact keys
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

        # Final tidy: normalize empty strings -> None, trim, and uppercase fields that should be uppercase
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
        """
        Full pipeline: tokenize -> regex -> label/value detect -> assemble KTP JSON.
        Returns the final standardized dict WITHOUT debug internals.
        """
        tokens = self.tokenize_lines(lines)
        joined_norm = " ".join(tok['norm'] for tok in tokens)
        regex_out = self.regex_extract(joined_norm)
        label_out = self.detect_labels_and_values(tokens, regex_out)
        return self.assemble_ktp_json(tokens, regex_out, label_out)


# -----------------------
# Example run (manual quick test)
# -----------------------
if __name__ == "__main__":
    # NOTE: Provide your regions CSV path here for best results.
    regions = None
    try:
        regions = AdministrativeRegions("kode_wilayah.csv")
        pass
    except Exception as e:
        logger.warning("Failed loading AdministrativeRegions: %s", e)

    # Sample of OCR result
    dataset = [
        [ "PROV1NSIDKI JAKARTA", "JAKARTABAR4T", "NIK 3171234567890123", "Nama M1RA SETIWAN",
          "TempatTgLLahr JAKRT4 18-02-1986", "JensKelmnn PERMPUAN GolDrh 8", "Almat JL PASTICEPATA7/66",
          "RTRW 007008", "KelDes PEGADUN6AN", "Kecamtan KALIDER3S", "Agm lSLM", "StusPrkawnan KAWlN",
          "Pekrjaan PEGAWAlSW4STA", "Kewarganegarn WNl", "BerlakHngga 22-02-2017", "JAKRTABRAT", "02-12-2O12" ],
        ["PROV1NSI BANTEN","KOTA TANGERANG","NIK 3605123456789012","Nama R1ZA PUTRA","TempatTgLLahr TANGERANG 05-03-1992",
         "JensKelmn LAK1-LAK1 GolDrh A","Almat JL PANDEGLANG 45/12","RTRW 003005","KelDes PANDEMANG","Kecamtan CILEGON",
         "Agm lSLM","StusPrkawnan KAWlN","Pekrjaan SWASTA","Kewarganegaran WNI","BerlakHngga 10-07-2015"],
        ["PROVINSI JAWA BARAT","KABUPATEN BANDUNG","NIK 3276543210987654","Nama D3WI S1RAT","TempatTgLLahr CIHAMPELAS 20-11-1988",
         "JensKelmn PERMPUAN GolDrh B","Almat JL C1BIRU 77/8","RTRW 002004","KelDes CIWIDEY","Kecamtan MARGAASIH",
         "Agm lSLM","StusPrkawnan BELUMKAWIN","Pekrjaan PNS","Kewarganegaran WNI","BerlakHngga 15-02-2012"],
        ["PROVINSI SULAWESI SELATAN","KOTA MAKASSAR","NIK 7371234567890123","Nama AH4MAD SYAH","TempatTgLLahr MAKASSAR 12-08-1990",
         "JensKelmn LAK1-LAK1 GolDrh O","Almat JL RANT3MBALU 9/10","RTRW 007009","KelDes TAMALATE","Kecamtan MARISO",
         "Agm lSLM","StusPrkawnan KAWlN","Pekrjaan SWASTA","Kewarganegaran WNI","BerlakHngga 01-01-2018"],
        ["PROVINSI DKI JAKARTA", "KOTA JAKARTA TIMUR", "NIK 3179876543210123", "Nama N1A PUTRI",
         "TempatTgLLahr C1LANGKAP 18-05-1995", "JensKelmn PERMPUAN GolDrh AB", "Almat JL CIPINANG 14/2",
         "RTRW 008009", "KelDes KRAMAT JATI", "Kecamtan JAKARTA TIMUR", "Agm lSLM", "StusPrkawnan KAWlN",
         "Pekrjaan PNS", "Kewarganegaran WNI", "BerlakHngga 22-02-2020"],
        ["PROVINSI JAWA TIMUR","KABUPATEN SIDOARJO","NIK 3576543210987654","Nama FAHR1A RAHMA","TempatTgLLahr SIDOARJO 03-09-1991",
         "JensKelmn PERMPUAN GolDrh A","Almat JL KEDUNGJATI 5/11","RTRW 001003","KelDes WARU","Kecamtan SIDOARJO",
         "Agm lSLM","StusPrkawnan BELUMKAWIN","Pekrjaan SWASTA","Kewarganegaran WNI","BerlakHngga 12-12-2016"]
    ]

    parser = OCRParser(regions=regions)
    for rec in dataset:
        parsed = parser.interpret(rec)
        print(json.dumps(parsed, indent=2, ensure_ascii=False))
        print('-' * 80)
