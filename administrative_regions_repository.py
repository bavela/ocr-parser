# administrative_regions_repository.py

from __future__ import annotations

import csv
import sqlite3
import unicodedata
import difflib
import string
from dataclasses import dataclass, asdict
from functools import lru_cache
from typing import List, Optional, Dict, Tuple, Any


# ----------------------------
# Model
# ----------------------------
@dataclass(frozen=True)
class Region:
    """Typed node in the Indonesian administrative hierarchy.

    What:
        A value object representing a single region (province/city/district/village).

    Why:
        Using a small typed model avoids leaking sqlite rows and keeps the rest of the
        code explicit, testable, and serialization-friendly.
    """
    region_id: str
    level: str        # 'province' | 'city' | 'district' | 'village'
    parent_id: Optional[str]
    name_off: str     # standardized official name (UPPER)

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serializable mapping of this Region for logs, APIs, or tests.

        Returns:
            Dict[str, Any]: {'region_id','level','parent_id','name_off'}
        """
        return asdict(self)


# ----------------------------
# Repository
# ----------------------------
class AdministrativeRegionsRepository:
    """SQLite-backed repository for hierarchy-respecting fuzzy search.

    What:
        Loads a single CSV (id,name) with hierarchical identifiers (e.g., '32.73.01')
        into a small sqlite database and exposes both exact-GET and fuzzy-search APIs.

    Why:
        Region names on KTP scans are noisy and often partially present; this repository
        centralizes normalization and fast, deterministic search so the parser remains small.

    Storage model (single table):
        regions(
            region_id    TEXT PRIMARY KEY,     -- "11", "11.01", "11.01.01", "11.01.01.2001"
            level        TEXT NOT NULL,        -- 'province'|'city'|'district'|'village'
            parent_id    TEXT,                 -- NULL for provinces
            name_raw     TEXT NOT NULL,        -- CSV name as-is
            name_off     TEXT NOT NULL,        -- standardized official name (UPPER)
            norm_name    TEXT NOT NULL,        -- normalized generic (OCR-fixed, punctuation-folded, UPPER)
            norm_nospace TEXT NOT NULL,        -- norm_name without spaces
            tokens       TEXT NOT NULL,        -- space-separated tokens of norm_name
            trigrams     TEXT NOT NULL         -- pg_trgm-style trigrams of norm_nospace, space-padded
        )

    Indices:
        - (level, norm_name)
        - (level, norm_nospace)
        - (parent_id, level)
        - (level, trigrams)
    """

    # OCR digit->letter fix for typical confusions (helps "JAKARTABAR4T" -> "JAKARTABARAT")
    # NOTE: deliberately **not** mapping '9' to 'G' to avoid over-correction.
    _OCR_DIGIT_TO_LETTER = str.maketrans({
        '0': 'O', '1': 'I', '2': 'Z', '3': 'E', '4': 'A', '5': 'S', '6': 'G', '7': 'T', '8': 'B'
    })

    # punctuation map: translate these to space (no regex)
    _PUNCTS = (
        string.punctuation
        + "·•—–‐-‒–—―…“”‘’´`´¨^~¨¸«»‹›•··"
    )
    _PUNCT_TO_SPACE = str.maketrans({ch: " " for ch in _PUNCTS})

    # Token-level aliases to harmonize common prefixes/abbreviations
    _TOKEN_ALIASES: Dict[str, str] = {
        "KAB.": "KABUPATEN",
        "KAB": "KABUPATEN",
        "KOTA ADM": "KOTA",
        "ADM.": "",  # drop administrative suffixes that vary by source
        "ADM": "",
    }

    def __init__(self, csv_file: str, sqlite_path: str | None = None):
        """Initialize and ingest the CSV so searches work immediately.

        Why:
            Keeping an embedded sqlite store avoids repeated CSV scans and enables indexed queries.

        Args:
            csv_file: Path to a 2-column CSV (id,name); header row is optional.
            sqlite_path: SQLite file path (persistent) or None (in-memory).
        """
        self.csv_file = csv_file
        self.sqlite_path = sqlite_path
        self.conn = sqlite3.connect(sqlite_path or ":memory:")
        self.conn.execute("PRAGMA journal_mode=WAL;")
        self.conn.execute("PRAGMA synchronous=NORMAL;")
        self._ensure_schema()
        self._load_csv_into_db(csv_file)

    # ----------------------------
    # Schema & load
    # ----------------------------
    def _ensure_schema(self) -> None:
        """Create tables and indices if missing.

        Why:
            One-time setup so downstream code can rely on fast indexed lookups.
        """
        cur = self.conn.cursor()
        cur.execute("""
        CREATE TABLE IF NOT EXISTS regions (
            region_id    TEXT PRIMARY KEY,
            level        TEXT NOT NULL CHECK(level IN ('province','city','district','village')),
            parent_id    TEXT,
            name_raw     TEXT NOT NULL,
            name_off     TEXT NOT NULL,
            norm_name    TEXT NOT NULL,
            norm_nospace TEXT NOT NULL,
            tokens       TEXT NOT NULL,
            trigrams     TEXT NOT NULL
        )
        """)
        cur.execute("CREATE INDEX IF NOT EXISTS idx_regions_level_norm ON regions(level, norm_name)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_regions_level_nns  ON regions(level, norm_nospace)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_regions_parent_level ON regions(parent_id, level)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_regions_level_trigrams ON regions(level, trigrams)")
        self.conn.commit()

    def _load_csv_into_db(self, csv_file: str) -> None:
        """Load CSV rows into sqlite, computing normalized features once.

        Why:
            Precomputing features keeps queries cheap and deterministic at runtime.

        Args:
            csv_file: CSV path with columns [id,name].
        """
        cur = self.conn.cursor()
        cur.execute("SELECT COUNT(1) FROM regions")
        count, = cur.fetchone()
        if count:
            return

        with open(csv_file, newline='', encoding='utf-8-sig') as f:
            reader = csv.reader(f)
            to_insert: List[Tuple[str, str, Optional[str], str, str, str, str, str, str]] = []
            for row in reader:
                if not row or len(row) < 2:
                    continue
                _id, name = (row[0] or "").strip(), (row[1] or "").strip()
                if not _id or not name:
                    continue

                # Skip header if present
                low_id, low_name = _id.lower(), name.lower()
                if low_id in ("id", "kode", "kode_wilayah") and low_name in ("name", "nama", "wilayah"):
                    continue

                parts = _id.split('.')
                if len(parts) == 1:
                    level, parent = 'province', None
                elif len(parts) == 2:
                    level, parent = 'city', parts[0]
                elif len(parts) == 3:
                    level, parent = 'district', f"{parts[0]}.{parts[1]}"
                elif len(parts) == 4:
                    level, parent = 'village', f"{parts[0]}.{parts[1]}.{parts[2]}"
                else:
                    continue  # unknown id depth

                name_off = self._std_official_name(name)
                norm_name = self._norm_generic(name_off)  # normalize the official form
                norm_nns = self._nospace(norm_name)
                tokens = self._tokens(norm_name)
                trigrams = self._trigrams(norm_nns)
                to_insert.append((_id, level, parent, name, name_off, norm_name, norm_nns, tokens, trigrams))

            cur.executemany("""
                INSERT OR IGNORE INTO regions(region_id, level, parent_id, name_raw, name_off,
                                              norm_name, norm_nospace, tokens, trigrams)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, to_insert)
            self.conn.commit()

    # ----------------------------
    # Normalization & features (regex-free)
    # ----------------------------
    @staticmethod
    @lru_cache(maxsize=8192)
    def _std_official_name(s: str) -> str:
        """Normalize to an uppercase, stable display form.

        Why:
            Ensures consistent rendering regardless of CSV capitalization/spacing quirks.

        Args:
            s: Original label as found in the CSV.

        Returns:
            Uppercased, whitespace-collapsed canonical name.
        """
        t = unicodedata.normalize("NFKC", s).strip()
        t = " ".join(t.split())
        return t.upper()

    @classmethod
    @lru_cache(maxsize=65536)
    def _norm_generic(cls, s: str) -> str:
        """Produce a robust, regex-free normalized string ready for fuzzy search.

        Why:
            Dampens OCR/punctuation noise and harmonizes common admin prefixes
            so searches are tolerant but deterministic.

        Args:
            s: Input text.

        Returns:
            Uppercased, punctuation-folded, whitespace-collapsed string with OCR digit fixes and token aliases applied.
        """
        if not s:
            return ""
        t = unicodedata.normalize("NFKC", s)
        t = t.translate(cls._OCR_DIGIT_TO_LETTER)
        t = t.translate(cls._PUNCT_TO_SPACE)
        t = " ".join(t.split()).upper()
        # apply token-level aliases
        tokens = t.split()
        mapped: List[str] = []
        i = 0
        while i < len(tokens):
            # join two-token alias like "KOTA ADM"
            if i + 1 < len(tokens) and (tokens[i] + " " + tokens[i+1]) in cls._TOKEN_ALIASES:
                alias = cls._TOKEN_ALIASES[tokens[i] + " " + tokens[i+1]]
                if alias:
                    mapped.append(alias)
                i += 2
                continue
            # single-token alias
            alias = cls._TOKEN_ALIASES.get(tokens[i])
            mapped.append(alias if alias is not None and alias != "" else tokens[i] if alias is None else "")
            i += 1
        t2 = " ".join(tok for tok in mapped if tok)
        return t2 or t  # fallback to original normalized if aliasing produced empty

    @staticmethod
    @lru_cache(maxsize=65536)
    def _nospace(s: str) -> str:
        """Space-free variant for glued-token comparisons.

        Args:
            s: Normalized string.

        Returns:
            Input with all whitespace removed.
        """
        return "".join(ch for ch in s if not ch.isspace())

    @staticmethod
    @lru_cache(maxsize=65536)
    def _tokens(norm_name: str) -> str:
        """Expose a tokenized view for simple token comparisons/filters.

        Args:
            norm_name: Normalized name.

        Returns:
            Space-separated tokens (single spaces).
        """
        return " ".join(norm_name.split())

    @staticmethod
    @lru_cache(maxsize=65536)
    def _trigrams(norm_nospace: str) -> str:
        """Generate pg_trgm-style character trigrams to cheaply narrow candidates.

        Args:
            norm_nospace: Normalized name without spaces.

        Returns:
            Space-padded trigram list, e.g. " abc bcd cde ".
        """
        s = f"  {norm_nospace}  "
        tris = []
        for i in range(len(s) - 2):
            tris.append(s[i:i+3])
        return " " + " ".join(tris) + " "

    @staticmethod
    @lru_cache(maxsize=65536)
    def _lev_ratio(a: str, b: str) -> float:
        """Character-level similarity in [0, 1].

        Why:
            Measures small edits from OCR or typos without heavy dependencies.
        """
        if not a or not b:
            return 0.0
        return difflib.SequenceMatcher(None, a.upper(), b.upper()).ratio()

    def _token_overlap(self, a: str, b: str) -> float:
        """Estimate shared vocabulary so semantically aligned candidates rank higher.

        Args:
            a: First string.
            b: Second string.

        Returns:
            Jaccard-like token overlap in [0, 1].
        """
        A = set(self._norm_generic(a).split())
        B = set(self._norm_generic(b).split())
        if not A or not B:
            return 0.0
        return len(A & B) / max(1, len(A | B))

    def _combo_score(self, a: str, b: str) -> float:
        """Composite spaced/no-space similarity + token overlap.

        Why:
            Blending views makes matching resilient to both glued tokens and word-level noise.

        Args:
            a: Query text.
            b: Candidate text.

        Returns:
            Composite similarity score in [0, 1].
        """
        a1, b1 = self._norm_generic(a), self._norm_generic(b)
        a2, b2 = self._nospace(a1), self._nospace(b1)
        lev1 = self._lev_ratio(a1, b1)
        lev2 = self._lev_ratio(a2, b2)
        tok = self._token_overlap(a, b)
        return 0.35 * lev1 + 0.45 * lev2 + 0.20 * tok

    # ----------------------------
    # Candidate retrieval (SQLite-assisted)
    # ----------------------------
    def _candidate_query(
        self,
        level: str,
        query: str,
        parent_id: Optional[str] = None,
        limit: int = 200
    ) -> List[sqlite3.Row]:
        """Fetch a compact candidate set before Python-side scoring.

        Why:
            Keeps the fuzzy stage fast while staying robust for very short/noisy queries.

        Args:
            level: Target granularity ('province'|'city'|'district'|'village').
            query: Free-form search text.
            parent_id: Optional parent scope to keep results contextual.
            limit: Upper bound on rows fetched for scoring.

        Returns:
            List of sqlite rows (id, level, parent, name_off, normalized fields).
        """
        self.conn.row_factory = sqlite3.Row
        cur = self.conn.cursor()

        norm = self._norm_generic(query)
        nns = self._nospace(norm)
        qlen = len(nns)

        p3 = norm[:3]
        q3 = nns[:3]

        tri_text = self._trigrams(nns)
        tri_list = [t for t in tri_text.strip().split()][:4]  # up to 4 trigrams

        where = ["level = ?"]
        params: List[Any] = [level]

        if parent_id:
            where.append("parent_id = ?")
            params.append(parent_id)

        # Loosen filters for very short queries (≤ 3 chars): skip trigrams, rely on prefixes/parent.
        if p3:
            where.append("norm_name LIKE ?")
            params.append(f"{p3}%")
        if q3 and qlen > 3:
            where.append("norm_nospace LIKE ?")
            params.append(f"{q3}%")

        if tri_list and qlen > 3:
            like_parts = []
            for tri in tri_list:
                like_parts.append("trigrams LIKE ?")
                params.append(f"% {tri} %")
            where.append("(" + " OR ".join(like_parts) + ")")

        sql = f"""
            SELECT region_id, level, parent_id, name_off, norm_name, norm_nospace
            FROM regions
            WHERE {' AND '.join(where)}
            LIMIT {int(limit)}
        """
        rows = cur.execute(sql, tuple(params)).fetchall()
        if rows:
            return rows

        # fallback — by level (and parent) with LIMIT, when filters were too strict
        base_where = ["level = ?"]
        base_params: List[Any] = [level]
        if parent_id:
            base_where.append("parent_id = ?")
            base_params.append(parent_id)
        sql2 = f"""
            SELECT region_id, level, parent_id, name_off, norm_name, norm_nospace
            FROM regions
            WHERE {' AND '.join(base_where)}
            LIMIT {int(limit)}
        """
        return cur.execute(sql2, tuple(base_params)).fetchall()

    @staticmethod
    def _row_to_region(r: sqlite3.Row | Tuple[str, str, Optional[str], str]) -> Region:
        """Convert a sqlite row/tuple into a Region entity.

        Args:
            r: sqlite row or (region_id, level, parent_id, name_off) tuple.

        Returns:
            Region model.
        """
        if isinstance(r, sqlite3.Row):
            return Region(r["region_id"], r["level"], r["parent_id"], r["name_off"])
        return Region(r[0], r[1], r[2], r[3])

    # ----------------------------
    # Pure GET APIs (by parent ID) — return Region objects
    # ----------------------------
    def get_region(self, region_id: str) -> Optional[Region]:
        """Fetch a single Region by its exact region_id.

        Why:
            Anchors lookups to a known node when a higher layer has already resolved.

        Args:
            region_id: Exact hierarchical ID (e.g., '11.01.01').

        Returns:
            The Region if found; otherwise None.
        """
        cur = self.conn.cursor()
        row = cur.execute("""
            SELECT region_id, level, parent_id, name_off
            FROM regions WHERE region_id=?
        """, (region_id,)).fetchone()
        return self._row_to_region(row) if row else None

    def get_provinces(self) -> List[Region]:
        """List all provinces.

        Why:
            Useful for populating dropdowns or a top-level traversal.

        Returns:
            Provinces ordered by official name.
        """
        cur = self.conn.cursor()
        rows = cur.execute("""
            SELECT region_id, level, parent_id, name_off
            FROM regions WHERE level='province' ORDER BY name_off
        """).fetchall()
        return [self._row_to_region(r) for r in rows]

    def get_cities(self, province_id: str) -> List[Region]:
        """List cities/kab under a province.

        Args:
            province_id: Parent province region_id.

        Returns:
            Cities/kab ordered by official name.
        """
        cur = self.conn.cursor()
        rows = cur.execute("""
            SELECT region_id, level, parent_id, name_off
            FROM regions
            WHERE level='city' AND parent_id=?
            ORDER BY name_off
        """, (province_id,)).fetchall()
        return [self._row_to_region(r) for r in rows]

    def get_districts(self, city_id: str) -> List[Region]:
        """List districts under a city.

        Args:
            city_id: Parent city/kab region_id.

        Returns:
            Districts ordered by official name.
        """
        cur = self.conn.cursor()
        rows = cur.execute("""
            SELECT region_id, level, parent_id, name_off
            FROM regions
            WHERE level='district' AND parent_id=?
            ORDER BY name_off
        """, (city_id,)).fetchall()
        return [self._row_to_region(r) for r in rows]

    def get_villages(self, district_id: str) -> List[Region]:
        """List villages under a district.

        Args:
            district_id: Parent district region_id.

        Returns:
            Villages ordered by official name.
        """
        cur = self.conn.cursor()
        rows = cur.execute("""
            SELECT region_id, level, parent_id, name_off
            FROM regions
            WHERE level='village' AND parent_id=?
            ORDER BY name_off
        """, (district_id,)).fetchall()
        return [self._row_to_region(r) for r in rows]

    # ----------------------------
    # SEARCH APIs (pg_trgm-ish; return [(Region, score)])
    # ----------------------------
    def search_provinces(self, q: str, k: int = 10) -> List[Tuple[Region, float]]:
        """Search for provinces most likely matching a query.

        Args:
            q: Free-form query text.
            k: Number of top results to return.

        Returns:
            Ranked (Region, score) pairs.
        """
        return self._search_level('province', q, k)

    def search_cities(self, q: str, province_id: Optional[str] = None, k: int = 10) -> List[Tuple[Region, float]]:
        """Search for cities/kab, optionally constrained to a province.

        Why:
            Scoping by parent improves precision on common names.

        Args:
            q: Query text.
            province_id: Optional province region_id.
            k: Number of top results.

        Returns:
            Ranked (Region, score) pairs.
        """
        return self._search_level('city', q, k, parent_id=province_id)

    def search_districts(self, q: str, city_id: Optional[str] = None, k: int = 10) -> List[Tuple[Region, float]]:
        """Search for districts, optionally constrained to a city/kab.

        Args:
            q: Query text.
            city_id: Optional city/kab region_id.
            k: Number of top results.

        Returns:
            Ranked (Region, score) pairs.
        """
        return self._search_level('district', q, k, parent_id=city_id)

    def search_villages(self, q: str, district_id: Optional[str] = None, k: int = 10) -> List[Tuple[Region, float]]:
        """Search for villages, optionally constrained to a district.

        Args:
            q: Query text.
            district_id: Optional district region_id.
            k: Number of top results.

        Returns:
            Ranked (Region, score) pairs.
        """
        return self._search_level('village', q, k, parent_id=district_id)

    def _search_level(self, level: str, q: str, k: int, parent_id: Optional[str] = None) -> List[Tuple[Region, float]]:
        """Rank and return top-k candidates at a given level.

        Why:
            A small, deterministic scoring keeps behavior predictable and explainable.

        Args:
            level: 'province'|'city'|'district'|'village'.
            q: Search text.
            k: Number of results to return.
            parent_id: Optional parent scope.

        Returns:
            Top-k (Region, score) pairs sorted by score desc.
        """
        cands = self._candidate_query(level, q, parent_id, limit=max(200, k * 12))
        scored: List[Tuple[Region, float]] = []
        for row in cands:
            sc = self._combo_score(q, row["name_off"])
            scored.append((self._row_to_region(row), sc))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:k]


if __name__ == "__main__":
    # Quick standalone sanity check for the regions repository.
    # Run this file directly to verify CSV ingestion and fuzzy search.
    repo = AdministrativeRegionsRepository(csv_file="kode_wilayah.csv", sqlite_path="wilayah.db")

    # Pure GET by parent id (smoke test)
    provinces: List[Region] = repo.get_provinces()
    cities: List[Region] = repo.get_cities(province_id="11")
    districts: List[Region] = repo.get_districts(city_id="11.01")
    villages: List[Region] = repo.get_villages(district_id="11.01.01")

    # Search (returns (Region, score))
    prov_hits = repo.search_provinces("dki jkart4", k=5)
    city_hits = repo.search_cities("bandar lampoeng", province_id=None, k=10)
    dist_hits = repo.search_districts("coblong", city_id="32.73")
    vill_hits = repo.search_villages("kr4ng anyar", district_id=None)

    # Quick prints
    print("Provinces:", len(provinces))
    print("Cities 11:", len(cities))
    print("Districts 11.01:", len(districts))
    print("Villages 11.01.01:", len(villages))
    print("Search provinces:", [(r.name_off, f"{s:.3f}") for r, s in prov_hits])
    print("Search cities:", [(r.name_off, f"{s:.3f}") for r, s in city_hits][:5])
    print("Search districts:", [(r.name_off, f"{s:.3f}") for r, s in dist_hits][:5])
    print("Search villages:", [(r.name_off, f"{s:.3f}") for r, s in vill_hits][:5])
