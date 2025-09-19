# administrative_regions_repository.py

from __future__ import annotations

import csv
import sqlite3
import unicodedata
import difflib
import string
from dataclasses import dataclass, asdict
from typing import List, Optional, Dict, Tuple, Any


# ----------------------------
# Model
# ----------------------------
@dataclass(frozen=True)
class Region:
    region_id: str
    level: str        # 'province' | 'city' | 'district' | 'village'
    parent_id: Optional[str]
    name_off: str     # standardized official name (UPPER)

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serializable mapping of this Region for logging, APIs, or comparisons.

        Returns:
            Dict[str, Any]: A plain dictionary with region_id, level, parent_id, and name_off.
        """
        return asdict(self)


# ----------------------------
# Repository
# ----------------------------
class AdministrativeRegionsRepository:
    """
    SQLite-backed repository for hierarchical administrative regions with OCR-noise-resilient fuzzy searching.

    Storage model (single table, denormalized features for speed):
        regions(
            region_id    TEXT PRIMARY KEY,     -- e.g., "11", "11.01", "11.01.01", "11.01.01.2001"
            level        TEXT NOT NULL,        -- 'province'|'city'|'district'|'village'
            parent_id    TEXT,                 -- parent region_id (NULL for provinces)
            name_raw     TEXT NOT NULL,        -- raw CSV name
            name_off     TEXT NOT NULL,        -- standardized official name (UPPER)
            norm_name    TEXT NOT NULL,        -- normalized generic (OCR-fixed, punctuation-folded, UPPER)
            norm_nospace TEXT NOT NULL,        -- norm_name without spaces
            tokens       TEXT NOT NULL,        -- space-separated tokens of norm_name
            trigrams     TEXT NOT NULL         -- space-separated char trigrams of norm_nospace, padded with spaces
        )

    Indices:
        - (level, norm_name)
        - (level, norm_nospace)
        - (parent_id, level)
        - (level, trigrams)
    """

    # OCR digit->letter fix for typical confusions (helps "JAKARTABAR4T" -> "JAKARTABARAT")
    _OCR_DIGIT_TO_LETTER = str.maketrans({
        '0': 'O', '1': 'I', '2': 'Z', '3': 'E', '4': 'A', '5': 'S', '6': 'G', '7': 'T', '8': 'B', '9': 'G'
    })

    # punctuation map: translate these to space (no regex)
    _PUNCTS = (
        string.punctuation
        + "·•—–‐-‒–—―…“”‘’´`´¨^~¨¸«»‹›•··"  # common unicode puncts/dashes/quotes/ellipses
    )
    _PUNCT_TO_SPACE = str.maketrans({ch: " " for ch in _PUNCTS})

    def __init__(self, csv_file: str, sqlite_path: str | None = None):
        """Initialize the repository and prepare the SQLite store so reads/searches work immediately.

        Args:
            csv_file (str): Path to a 2-column CSV (id,name); header is optional.
            sqlite_path (str | None): SQLite file path to persist data; None keeps it in-memory.
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
        """Create the core table and indexes to persist regions and search features for fast lookups.

        Returns:
            None
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
        """Load the CSV once into SQLite, storing raw labels plus precomputed features for search.

        Args:
            csv_file (str): Path to the id–name CSV to ingest.

        Returns:
            None
        """
        cur = self.conn.cursor()
        cur.execute("SELECT COUNT(1) FROM regions")
        count, = cur.fetchone()
        if count:
            return

        with open(csv_file, newline='', encoding='utf-8-sig') as f:
            reader = csv.reader(f)
            to_insert: list[tuple] = []
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
                norm_name = self._norm_generic(name)
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
    # Normalization & features (NO regex)
    # ----------------------------
    @staticmethod
    def _std_official_name(s: str) -> str:
        """Normalize a display name to an uppercase, stable form for consistent storage.

        Args:
            s (str): Original label as found in the CSV.

        Returns:
            str: Uppercased, whitespace-collapsed canonical name.
        """
        t = unicodedata.normalize("NFKC", s).strip()
        t = " ".join(t.split())
        return t.upper()

    def _norm_generic(self, s: str) -> str:
        """Produce a regex-free normalized string that dampens OCR and punctuation noise for robust search.

        Args:
            s (str): Input text to normalize.

        Returns:
            str: Uppercased, punctuation-folded, whitespace-collapsed string with OCR digit fixes applied.
        """
        if not s:
            return ""
        t = unicodedata.normalize("NFKC", s)
        t = t.translate(self._OCR_DIGIT_TO_LETTER)
        t = t.translate(self._PUNCT_TO_SPACE)
        t = " ".join(t.split())
        t = t.upper()
        return t

    @staticmethod
    def _nospace(s: str) -> str:
        """Create a space-free variant of a normalized string to handle glued tokens.

        Args:
            s (str): Normalized string.

        Returns:
            str: Input with all whitespace removed.
        """
        return "".join(ch for ch in s if not ch.isspace())

    @staticmethod
    def _tokens(norm_name: str) -> str:
        """Expose a tokenized view of a normalized name for simple token comparisons/filters.

        Args:
            norm_name (str): Normalized name.

        Returns:
            str: Space-separated tokens (same as input but with single spaces).
        """
        return " ".join(norm_name.split())

    @staticmethod
    def _trigrams(norm_nospace: str) -> str:
        """Generate pg_trgm-style character trigrams to cheaply narrow search candidates.

        Args:
            norm_nospace (str): Normalized name without spaces.

        Returns:
            str: Space-padded list of trigrams (e.g., " abc bcd cde ").
        """
        s = f"  {norm_nospace}  "
        tris = []
        for i in range(len(s) - 2):
            tris.append(s[i:i+3])
        return " " + " ".join(tris) + " "

    @staticmethod
    def _lev_ratio(a: str, b: str) -> float:
        """Measure character-level similarity to capture small edits from OCR or typos.

        Args:
            a (str): First string.
            b (str): Second string.

        Returns:
            float: Similarity ratio in [0, 1].
        """
        if not a or not b:
            return 0.0
        return difflib.SequenceMatcher(None, a.upper(), b.upper()).ratio()

    def _token_overlap(self, a: str, b: str) -> float:
        """Estimate shared vocabulary so semantically aligned candidates rank higher.

        Args:
            a (str): First string.
            b (str): Second string.

        Returns:
            float: Jaccard-like token overlap in [0, 1].
        """

        A = set(self._norm_generic(a).split())
        B = set(self._norm_generic(b).split())
        if not A or not B:
            return 0.0
        return len(A & B) / max(1, len(A | B))

    def _combo_score(self, a: str, b: str) -> float:
        """Combine spaced/no-space similarity and token overlap into a single robust score.

        Args:
            a (str): Query text.
            b (str): Candidate text.

        Returns:
            float: Composite similarity score in [0, 1] used for ranking.
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
        """Retrieve a small set of likely candidates from SQLite before Python-side scoring.

        Args:
            level (str): Target granularity ('province'|'city'|'district'|'village').
            query (str): Free-form search text.
            parent_id (Optional[str]): Optional parent scope to keep results contextual.
            limit (int): Max rows to fetch for scoring.

        Returns:
            List[sqlite3.Row]: Raw rows containing IDs and normalized fields for scoring.
        """
        self.conn.row_factory = sqlite3.Row
        cur = self.conn.cursor()

        norm = self._norm_generic(query)
        nns = self._nospace(norm)

        p3 = norm[:3]
        q3 = nns[:3]

        tri_text = self._trigrams(nns)
        tri_list = [t for t in tri_text.strip().split()][:4]  # up to 4 trigrams

        where = ["level = ?"]
        params: list[Any] = [level]

        if parent_id:
            where.append("parent_id = ?")
            params.append(parent_id)

        if p3:
            where.append("norm_name LIKE ?")
            params.append(f"{p3}%")
        if q3:
            where.append("norm_nospace LIKE ?")
            params.append(f"{q3}%")

        # OR group for up to 4 trigrams
        if tri_list:
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

        # fallback — just by level (and parent) with LIMIT
        base_where = ["level = ?"]
        base_params: list[Any] = [level]
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
    def _row_to_region(r: sqlite3.Row | Tuple) -> Region:
        """Convert a SQLite row/tuple into a Region entity for typed, ergonomic usage.

        Args:
            r (sqlite3.Row | Tuple): A database row or 4-tuple (region_id, level, parent_id, name_off).

        Returns:
            Region: The corresponding Region model.
        """
        if isinstance(r, sqlite3.Row):
            return Region(r["region_id"], r["level"], r["parent_id"], r["name_off"])
        return Region(r[0], r[1], r[2], r[3])

    # ----------------------------
    # Pure GET APIs (by parent ID) — return Region objects
    # ----------------------------
    def get_region(self, region_id: str) -> Optional[Region]:
        """Fetch a single Region by its exact region_id to anchor lookups to a known node.

        Args:
            region_id (str): Exact hierarchical ID (e.g., '11.01.01').

        Returns:
            Optional[Region]: The Region if found; otherwise None.
        """
        cur = self.conn.cursor()
        row = cur.execute("""
            SELECT region_id, level, parent_id, name_off
            FROM regions WHERE region_id=?
        """, (region_id,)).fetchone()
        return self._row_to_region(row) if row else None

    def get_provinces(self) -> List[Region]:
        """List all provinces as Region entities to enumerate the hierarchy root.

        Returns:
            List[Region]: Provinces ordered by official name.
        """
        cur = self.conn.cursor()
        rows = cur.execute("""
            SELECT region_id, level, parent_id, name_off
            FROM regions WHERE level='province' ORDER BY name_off
        """).fetchall()
        return [self._row_to_region(r) for r in rows]

    def get_cities(self, province_id: str) -> List[Region]:
        """List cities/kab under a province so traversal remains explicit and deterministic.

        Args:
            province_id (str): Province region_id acting as the parent scope.

        Returns:
            List[Region]: Cities/kab ordered by official name.
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
        """List districts under a city to drill one level deeper in the hierarchy.

        Args:
            city_id (str): City/kab region_id acting as the parent scope.

        Returns:
            List[Region]: Districts ordered by official name.
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
        """List villages under a district to complete the last hop of the hierarchy.

        Args:
            district_id (str): District region_id acting as the parent scope.

        Returns:
            List[Region]: Villages ordered by official name.
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
        """Search for provinces most likely matching a free-form query for suggestions/autocomplete.

        Args:
            q (str): User query text.
            k (int): Number of top results to return.

        Returns:
            List[Tuple[Region, float]]: Ranked (Region, score) pairs.
        """
        return self._search_level('province', q, k)

    def search_cities(self, q: str, province_id: Optional[str] = None, k: int = 10) -> List[Tuple[Region, float]]:
        """Search for cities/kab matching a query, optionally scoped to a province for relevance.

        Args:
            q (str): User query text.
            province_id (Optional[str]): Optional province region_id to constrain matches.
            k (int): Number of top results to return.

        Returns:
            List[Tuple[Region, float]]: Ranked (Region, score) pairs.
        """
        return self._search_level('city', q, k, parent_id=province_id)

    def search_districts(self, q: str, city_id: Optional[str] = None, k: int = 10) -> List[Tuple[Region, float]]:
        """Search for districts matching a query, optionally scoped to a city for precision.

        Args:
            q (str): User query text.
            city_id (Optional[str]): Optional city/kab region_id to constrain matches.
            k (int): Number of top results to return.

        Returns:
            List[Tuple[Region, float]]: Ranked (Region, score) pairs.
        """
        return self._search_level('district', q, k, parent_id=city_id)

    def search_villages(self, q: str, district_id: Optional[str] = None, k: int = 10) -> List[Tuple[Region, float]]:
        """Search for villages matching a query, optionally scoped to a district for precision.

        Args:
            q (str): User query text.
            district_id (Optional[str]): Optional district region_id to constrain matches.
            k (int): Number of top results to return.

        Returns:
            List[Tuple[Region, float]]: Ranked (Region, score) pairs.
        """
        return self._search_level('village', q, k, parent_id=district_id)

    def _search_level(self, level: str, q: str, k: int, parent_id: Optional[str] = None) -> List[Tuple[Region, float]]:
        """Rank and return the top-k candidates at a given level so callers get useful suggestions fast.

        Args:
            level (str): Target granularity ('province'|'city'|'district'|'village').
            q (str): User query text.
            k (int): Number of top results to return.
            parent_id (Optional[str]): Optional parent scope to narrow candidates.

        Returns:
            List[Tuple[Region, float]]: Top-k (Region, score) pairs sorted by score desc.
        """
        cands = self._candidate_query(level, q, parent_id, limit=max(200, k * 12))
        scored: List[Tuple[Region, float]] = []
        for row in cands:
            sc = self._combo_score(q, row["name_off"])
            scored.append((self._row_to_region(row), sc))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:k]

if __name__ == "__main__":
    # Optional: nicer printout for Region
    try:
        from dataclasses import replace
    except Exception:
        pass

    def p_region_list(title: str, items: list[Region]) -> None:
        """Pretty-print a list of Regions for quick manual inspection during runs.

        Args:
            title (str): Section title to print.
            items (list[Region]): Regions to display.
        """
        print(f"\n=== {title} (count={len(items)}) ===")
        for r in items:
            print(f"- [{r.level}] {r.name_off}  (id={r.region_id}, parent={r.parent_id})")

    def p_hits(title: str, hits: list[tuple[Region, float]]) -> None:
        """Pretty-print ranked search results to visually assess match quality.

        Args:
            title (str): Section title to print.
            hits (list[tuple[Region, float]]): Ranked results as (Region, score).
        """
        print(f"\n=== {title} (count={len(hits)}) ===")
        for r, score in hits:
            print(f"- [{r.level}] {r.name_off}  (id={r.region_id}, parent={r.parent_id})  score={score:.4f}")

    repo = AdministrativeRegionsRepository(csv_file="kode_wilayah.csv", sqlite_path="wilayah.db")

    # Pure GET by parent id
    provinces: list[Region] = repo.get_provinces()
    cities: list[Region] = repo.get_cities(province_id="11")
    districts: list[Region] = repo.get_districts(city_id="11.01")
    villages: list[Region] = repo.get_villages(district_id="11.01.01")

    # Search (returns (Region, score))
    prov_hits = repo.search_provinces("dki jkart4", k=5)
    city_hits = repo.search_cities("bandar lampoeng", province_id=None, k=10)
    dist_hits = repo.search_districts("coblong", city_id="32.73")
    vill_hits = repo.search_villages("kr4ng anyar", district_id=None)

    # Print everything
    p_region_list("Provinces", provinces)
    p_region_list("Cities of province_id=11", cities)
    p_region_list("Districts of city_id=11.01", districts)
    p_region_list("Villages of district_id=11.01.01", villages)

    p_hits('Search provinces: "dki jkart4"', prov_hits)
    p_hits('Search cities: "bandar lampoeng"', city_hits)
    p_hits('Search districts in city_id=32.73: "coblong"', dist_hits)
    p_hits('Search villages: "kr4ng anyar"', vill_hits)


# Design notes — AdministrativeRegionsRepository search strategy (no AI/ML)

# Purpose:
#     Provide reliable, SQLite-backed fuzzy search over Indonesian administrative hierarchy
#     (Province → City/Kab → District → Village) using deterministic string features.

# Candidate retrieval (SQLite-assisted):
#     • Narrow with:
#         - level filter (required).
#         - optional parent_id scope (strongly recommended).
#         - prefixes on normalized name and nospace (first 3 chars).
#         - up to 3–4 trigram LIKE filters.
#         - small LIMIT (≤ 200) to bound Python-side scoring.
#     • Precomputed features per row:
#         - norm_name, norm_nospace, tokens, and pg_trgm-style trigrams (space-padded).

# Scoring (Python-side, deterministic):
#     • Composite score = 0.35 * Levenshtein(spaced) + 0.45 * Levenshtein(nospace) + 0.20 * token overlap.
#     • Return (Region, score) so callers can blend with additional heuristics.

# Hierarchy-aware resolution (recommended in OCRParser):
#     • Top-down beam search:
#         1) Take top-2/3 provinces from search over the flattened OCR text.
#         2) For each province, search cities *within that province*.
#         3) For each city, search districts; then villages.
#         4) Overall path score = product/sum of local scores plus hierarchy consistency bonus.
#     • Whole-text backstop:
#         - If level-specific labels are weak/missing, search using entire flattened text and
#           blend global score (e.g., 0.7 local + 0.3 global).
#     • Conflict repair rules:
#         - If a city does not belong to the chosen province, either:
#             (a) switch to the city's true province if city score ≫ province score, or
#             (b) keep province and re-search city constrained to it.

# Performance:
#     • WAL mode for concurrent reads; synchronous=NORMAL is fine.
#     • Cache repeated queries and normalizations (LRU).
#     • Deduplicate candidate phrases before hitting SQLite.
#     • Early stop: once a level is high-confidence, skip broader/global searches.

# Threshold guidance (starting points):
#     • province ≥ 0.58; city ≥ 0.60; district ≥ 0.60; village ≥ 0.62.
#     • hierarchy bonus +0.05 per valid parent link; mismatch penalty −0.08.
