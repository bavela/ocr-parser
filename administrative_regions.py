"""
filepath: administrative_regions.py

Robust AdministrativeRegions:
- Loads a 2-column CSV (id,name) with hierarchical IDs: 11 / 11.01 / 11.01.01 / 11.01.01.2001
- Tolerant to headers or BOM
- Fuzzy, OCR-aware matching for Province -> City/Kab -> District -> Village
- Public API kept compatible:
    get_provinces() -> List[str]
    get_cities(province_name: str) -> List[str]
    get_districts(city_name: str) -> List[str]
    get_villages(district_name: str) -> List[str]
- Extra resolvers for single-best match:
    find_best_province(name) -> Optional[str]
    find_best_city(name, province_name=None) -> Optional[str]
    find_best_district(name, city_name=None) -> Optional[str]
    find_best_village(name, district_name=None) -> Optional[str]
    find_anywhere(text) -> Dict[str, Optional[str]]   # whole-text fallback resolver
"""

from __future__ import annotations

import csv
import re
import unicodedata
import difflib
from collections import defaultdict
from typing import List, Optional, Dict, Tuple


class AdministrativeRegions:
    """
    Manage hierarchical administrative regions (Province -> City/Kab -> District -> Village)
    with OCR-noise-resilient fuzzy matching.
    """

    # --- Prefix patterns we want to ignore when normalizing noisy inputs ---
    _PREFIX_PATTERNS = {
        "province": re.compile(r'^(PROV(?:INSI)?\.?\s+)', re.I),
        "city":     re.compile(r'^(KAB(?:UPATEN)?\.?\s+|KOTA(?:\s+ADM\.?)?\s+|KOTAMADYA\s+)', re.I),
        "district": re.compile(r'^(KEC(?:AMATAN)?\.?\s+)', re.I),
        "village":  re.compile(r'^(KEL(?:URAHAN)?\.?\s+|DESA\s+)', re.I),
        "generic":  re.compile(r'^(KAB(?:UPATEN)?\.?\s+|KOTA(?:\s+ADM\.?)?\s+|KOTAMADYA\s+|KEC(?:AMATAN)?\.?\s+|KEL(?:URAHAN)?\.?\s+|DESA\s+|PROV(?:INSI)?\.?\s+)', re.I),
    }

    # OCR digit->letter fix for typical confusions (helps "JAKARTABAR4T" -> "JAKARTABARAT")
    _OCR_DIGIT_TO_LETTER = str.maketrans({
        '0': 'O', '1': 'I', '2': 'Z', '3': 'E', '4': 'A', '5': 'S', '6': 'G', '7': 'T', '8': 'B', '9': 'G'
    })

    def __init__(self, csv_file: str):
        # Hierarchies
        self.provinces: Dict[str, str] = {}                 # {province_id: province_name}
        self.cities: Dict[str, Dict[str, str]] = defaultdict(dict)       # {province_id: {city_id: city_name}}
        self.districts: Dict[str, Dict[str, str]] = defaultdict(dict)    # {city_id: {district_id: district_name}}
        self.villages: Dict[str, Dict[str, str]] = defaultdict(dict)     # {district_id: {village_id: village_name}}

        # Reverse indexes (filled after load)
        self._city_name_to_ids: Dict[str, Tuple[str, str]] = {}
        self._district_name_to_ids: Dict[str, Tuple[str, str]] = {}
        self._village_name_to_ids: Dict[str, Tuple[str, str]] = {}

        self._load_csv(csv_file)

    # ----------------------------
    # CSV loading
    # ----------------------------
    def _load_csv(self, csv_file: str) -> None:
        """
        Load a 2-column CSV: id,name (header optional). Builds all hierarchy maps.
        """
        with open(csv_file, newline='', encoding='utf-8-sig') as f:
            reader = csv.reader(f)
            for row in reader:
                if not row or len(row) < 2:
                    continue
                _id, name = row[0].strip(), row[1].strip()
                if not _id or not name:
                    continue

                # Skip header if present
                if _id.lower() in ("id", "kode", "kode_wilayah") and name.lower() in ("name", "nama", "wilayah"):
                    continue

                parts = _id.split('.')
                # Normalize stored official names to UPPER (as in many gov datasets)
                off_name = self._std_official_name(name)

                if len(parts) == 1:  # Province
                    self.provinces[_id] = off_name
                elif len(parts) == 2:  # City/Kab
                    province_id = parts[0]
                    self.cities[province_id][_id] = off_name
                elif len(parts) == 3:  # District
                    city_id = f"{parts[0]}.{parts[1]}"
                    self.districts[city_id][_id] = off_name
                elif len(parts) == 4:  # Village
                    district_id = f"{parts[0]}.{parts[1]}.{parts[2]}"
                    self.villages[district_id][_id] = off_name

        # Build reverse indexes for fast parent-scoped lookups
        self._build_reverse_indexes()

    def _build_reverse_indexes(self) -> None:
        self._city_name_to_ids.clear()
        self._district_name_to_ids.clear()
        self._village_name_to_ids.clear()

        for prov_id, prov_name in self.provinces.items():
            for city_id, city_name in self.cities.get(prov_id, {}).items():
                self._city_name_to_ids[self._norm(city_name, "city")] = (city_id, prov_id)
                for dist_id, dist_name in self.districts.get(city_id, {}).items():
                    self._district_name_to_ids[self._norm(dist_name, "district")] = (dist_id, city_id)
                    for vill_id, vill_name in self.villages.get(dist_id, {}).items():
                        self._village_name_to_ids[self._norm(vill_name, "village")] = (vill_id, dist_id)

    # ----------------------------
    # Normalization & scoring
    # ----------------------------
    @staticmethod
    def _std_official_name(s: str) -> str:
        # Standardize names as they appear in datasets (usually uppercase; keep dots/slashes as-is)
        t = unicodedata.normalize("NFKC", s).strip()
        t = re.sub(r'\s+', ' ', t)
        return t.upper()

    def _norm(self, s: str, level: str = "generic") -> str:
        """
        Heavy-duty text normalizer for OCR'd Indonesian admin names:
          - Unicode NFKC
          - OCR digit->letter fix (4->A, 1->I, etc.)
          - strip common punctuation to create soft breaks
          - remove known prefixes (KAB/KOTA/KEC/KEL/DESA/PROV)
          - uppercase & collapse spaces
        """
        if not s:
            return ""
        t = unicodedata.normalize("NFKC", s)
        t = t.translate(self._OCR_DIGIT_TO_LETTER)
        t = re.sub(r'[.,;:/\-]+', ' ', t)
        t = re.sub(r'\s+', ' ', t).strip()
        pat = self._PREFIX_PATTERNS.get(level, self._PREFIX_PATTERNS["generic"])
        t = pat.sub('', t)
        t = t.upper()
        t = re.sub(r'\s+', ' ', t).strip()
        return t

    def _norm_nospace(self, s: str, level: str = "generic") -> str:
        return re.sub(r'\s+', '', self._norm(s, level))

    @staticmethod
    def _lev_ratio(a: str, b: str) -> float:
        if not a or not b:
            return 0.0
        return difflib.SequenceMatcher(None, a.upper(), b.upper()).ratio()

    def _token_overlap(self, a: str, b: str) -> float:
        A = set(self._norm(a).split())
        B = set(self._norm(b).split())
        if not A or not B:
            return 0.0
        return len(A & B) / len(A | B)

    def _combo_score(self, a: str, b: str, level: str) -> float:
        """
        Combined score to be stable under glued tokens & OCR:
          - spaced Levenshtein
          - no-space Levenshtein
          - token overlap
        Weights tuned for noisy KTP OCR.
        """
        a1, b1 = self._norm(a, level), self._norm(b, level)
        a2, b2 = self._norm_nospace(a, level), self._norm_nospace(b, level)
        lev1 = self._lev_ratio(a1, b1)
        lev2 = self._lev_ratio(a2, b2)
        tok = self._token_overlap(a, b)
        return 0.35 * lev1 + 0.45 * lev2 + 0.20 * tok

    def _best_key_by_value(self, mapping: Dict[str, str], name: str, level: str, cutoff: float) -> Optional[str]:
        if not name or not mapping:
            return None
        best_id, best_score = None, 0.0
        for _id, official in mapping.items():
            sc = self._combo_score(name, official, level)
            if sc > best_score:
                best_id, best_score = _id, sc
        return best_id if best_score >= cutoff else None

    # ----------------------------
    # Public list APIs (compatible with your current usage)
    # ----------------------------
    def get_provinces(self) -> List[str]:
        return list(self.provinces.values())

    def get_cities(self, province_name: str) -> List[str]:
        """
        Return cities/kab for a (possibly noisy) province name.
        """
        prov_id = self._best_key_by_value(self.provinces, province_name, level="province", cutoff=0.60)
        if prov_id and prov_id in self.cities:
            return list(self.cities[prov_id].values())
        return []

    def get_districts(self, city_name: str) -> List[str]:
        """
        Return districts for a (possibly noisy) city/kab name (searching across all provinces).
        """
        best_city_id, best_score = None, 0.0
        for prov_id, city_map in self.cities.items():
            cid = self._best_key_by_value(city_map, city_name, level="city", cutoff=0.58)
            if cid:
                sc = self._combo_score(city_name, city_map[cid], "city")
                if sc > best_score:
                    best_city_id, best_score = cid, sc
        if best_city_id and best_city_id in self.districts:
            return list(self.districts[best_city_id].values())
        return []

    def get_villages(self, district_name: str) -> List[str]:
        """
        Return villages for a (possibly noisy) district name (searching across all cities).
        """
        best_dist_id, best_score = None, 0.0
        for city_id, dist_map in self.districts.items():
            did = self._best_key_by_value(dist_map, district_name, level="district", cutoff=0.58)
            if did:
                sc = self._combo_score(district_name, dist_map[did], "district")
                if sc > best_score:
                    best_dist_id, best_score = did, sc
        if best_dist_id and best_dist_id in self.villages:
            return list(self.villages[best_dist_id].values())
        return []

    # ----------------------------
    # Single-best resolvers
    # ----------------------------
    def find_best_province(self, name: str) -> Optional[str]:
        pid = self._best_key_by_value(self.provinces, name, level="province", cutoff=0.60)
        return self.provinces.get(pid) if pid else None

    def find_best_city(self, name: str, province_name: Optional[str] = None) -> Optional[str]:
        # If province is given, restrict to that province’s cities first
        if province_name:
            prov_id = self._best_key_by_value(self.provinces, province_name, level="province", cutoff=0.58)
            if prov_id:
                cid = self._best_key_by_value(self.cities.get(prov_id, {}), name, level="city", cutoff=0.58)
                if cid:
                    return self.cities[prov_id][cid]
        # fallback: search across all provinces
        best_name, best_score = None, 0.0
        for prov_id, city_map in self.cities.items():
            cid = self._best_key_by_value(city_map, name, level="city", cutoff=0.58)
            if cid:
                sc = self._combo_score(name, city_map[cid], "city")
                if sc > best_score:
                    best_name, best_score = city_map[cid], sc
        return best_name

    def find_best_district(self, name: str, city_name: Optional[str] = None) -> Optional[str]:
        # If city is given, narrow to that city’s districts
        if city_name:
            best_city_id, best = None, 0.0
            for prov_id, city_map in self.cities.items():
                cid = self._best_key_by_value(city_map, city_name, level="city", cutoff=0.58)
                if cid:
                    sc = self._combo_score(city_name, city_map[cid], "city")
                    if sc > best:
                        best_city_id, best = cid, sc
            if best_city_id:
                did = self._best_key_by_value(self.districts.get(best_city_id, {}), name, level="district", cutoff=0.58)
                return self.districts[best_city_id].get(did) if did else None

        # fallback: search across all districts
        best_name, best_score = None, 0.0
        for city_id, dist_map in self.districts.items():
            did = self._best_key_by_value(dist_map, name, level="district", cutoff=0.58)
            if did:
                sc = self._combo_score(name, dist_map[did], "district")
                if sc > best_score:
                    best_name, best_score = dist_map[did], sc
        return best_name

    def find_best_village(self, name: str, district_name: Optional[str] = None) -> Optional[str]:
        # If district is given, narrow to that district’s villages
        if district_name:
            best_dist_id, best = None, 0.0
            for city_id, dist_map in self.districts.items():
                did = self._best_key_by_value(dist_map, district_name, level="district", cutoff=0.58)
                if did:
                    sc = self._combo_score(district_name, dist_map[did], "district")
                    if sc > best:
                        best_dist_id, best = did, sc
            if best_dist_id:
                vid = self._best_key_by_value(self.villages.get(best_dist_id, {}), name, level="village", cutoff=0.58)
                return self.villages[best_dist_id].get(vid) if vid else None

        # fallback: search across all villages
        best_name, best_score = None, 0.0
        for dist_id, vill_map in self.villages.items():
            vid = self._best_key_by_value(vill_map, name, level="village", cutoff=0.60)
            if vid:
                sc = self._combo_score(name, vill_map[vid], "village")
                if sc > best_score:
                    best_name, best_score = vill_map[vid], sc
        return best_name

    # ----------------------------
    # Whole-text fallback resolver
    # ----------------------------
    def find_anywhere(self, text: str) -> Dict[str, Optional[str]]:
        """
        Aggressively guess Province -> City -> District -> Village from a free-form page text.
        Uses the same OCR-aware comparators but searches across all levels.
        """
        result: Dict = {"Provinsi": None, "Kota": None, "Kecamatan": None, "Kel_Desa": None}
        if not text:
            return result

        # Province
        best_prov, best_p = None, 0.0
        for pid, pname in self.provinces.items():
            sc = self._combo_score(text, pname, "province")
            if sc > best_p:
                best_prov, best_p = pname, sc
        result["Provinsi"] = best_prov if best_p >= 0.58 else None

        # Resolve province_id from chosen name
        prov_id = None
        for pid, pname in self.provinces.items():
            if pname == result["Provinsi"]:
                prov_id = pid
                break

        # City (prefer within province)
        def best_city_in(_prov_id: Optional[str]) -> Optional[str]:
            if not _prov_id:
                return None
            best_name, best_s = None, 0.0
            for cid, cname in self.cities.get(_prov_id, {}).items():
                sc = self._combo_score(text, cname, "city")
                if sc > best_s:
                    best_name, best_s = cname, sc
            return best_name if best_s >= 0.58 else None

        city = best_city_in(prov_id) if prov_id else None
        if not city:
            best_name, best_s = None, 0.0
            for _pid, cmap in self.cities.items():
                for _cid, cname in cmap.items():
                    sc = self._combo_score(text, cname, "city")
                    if sc > best_s:
                        best_name, best_s = cname, sc
            city = best_name if best_s >= 0.60 else None
        result["Kota"] = city

        # District (prefer within resolved city)
        def best_dist_in(city_name: Optional[str]) -> Optional[str]:
            if not city_name:
                return None
            best_city_id, best_s = None, 0.0
            for _pid, cmap in self.cities.items():
                for _cid, cname in cmap.items():
                    sc = self._combo_score(city_name, cname, "city")
                    if sc > best_s:
                        best_city_id, best_s = _cid, sc
            if not best_city_id:
                return None
            best_name, best_sc = None, 0.0
            for did, dname in self.districts.get(best_city_id, {}).items():
                sc = self._combo_score(text, dname, "district")
                if sc > best_sc:
                    best_name, best_sc = dname, sc
            return best_name if best_sc >= 0.58 else None

        kec = best_dist_in(result["Kota"])
        if not kec:
            best_name, best_s = None, 0.0
            for _cid, dmap in self.districts.items():
                for did, dname in dmap.items():
                    sc = self._combo_score(text, dname, "district")
                    if sc > best_s:
                        best_name, best_s = dname, sc
            kec = best_name if best_s >= 0.60 else None
        result["Kecamatan"] = kec

        # Village (prefer within resolved district)
        def best_vill_in(dist_name: Optional[str]) -> Optional[str]:
            if not dist_name:
                return None
            best_dist_id, best_ds = None, 0.0
            for _cid, dmap in self.districts.items():
                for did, dname in dmap.items():
                    sc = self._combo_score(dist_name, dname, "district")
                    if sc > best_ds:
                        best_dist_id, best_ds = did, sc
            if not best_dist_id:
                return None
            best_vname, best_vs = None, 0.0
            for vid, vname in self.villages.get(best_dist_id, {}).items():
                sc = self._combo_score(text, vname, "village")
                if sc > best_vs:
                    best_vname, best_vs = vname, sc
            return best_vname if best_vs >= 0.60 else None

        kel = best_vill_in(result["Kecamatan"])
        if not kel:
            best_vname, best_vs = None, 0.0
            for did, vmap in self.villages.items():
                for vid, vname in vmap.items():
                    sc = self._combo_score(text, vname, "village")
                    if sc > best_vs:
                        best_vname, best_vs = vname, sc
            kel = best_vname if best_vs >= 0.62 else None
        result["Kel_Desa"] = kel

        return result
