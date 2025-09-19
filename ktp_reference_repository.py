# ktp_reference_repository.py

from __future__ import annotations

import sqlite3
import unicodedata
import difflib
import string
from dataclasses import dataclass, asdict
from typing import List, Tuple, Any, Dict, Optional


# ----------------------------
# Model
# ----------------------------
@dataclass(frozen=True)
class Ref:
    """
    Represent a single reference value inside a category (e.g., 'agama' -> 'ISLAM').

    Why:
        Using a typed model keeps return types consistent across repositories and
        simplifies downstream code (no sqlite rows or bare strings to juggle).

    Fields:
        category (str): The logical bucket of this value (e.g., 'agama').
        value (str): Canonical display label (uppercase, stable form).
    """
    category: str
    value: str

    def to_dict(self) -> Dict[str, Any]:
        """Expose a JSON-serializable mapping for APIs, logs, or tests.

        Returns:
            Dict[str, Any]: {'category': ..., 'value': ...}
        """
        return asdict(self)


class KTPReferenceRepository:
    """
    SQLite-backed repository for non-administrative KTP reference fields with
    pg_trgm-ish fuzzy search and simple 'get all' access.

    Why:
        Centralizes standard sets (gender, blood type, religion, etc.) in one place,
        enabling consistent UI options and resilient fuzzy lookups across the app.

    Covered categories (keys):
      - 'jenis_kelamin'      (LAKI-LAKI, PEREMPUAN)
      - 'golongan_darah'     (A, B, AB, O, -)
      - 'agama'              (ISLAM, KRISTEN, KATHOLIK, HINDU, BUDDHA, KONGHUCU, ...)
      - 'status_perkawinan'  (BELUM KAWIN, KAWIN, CERAI HIDUP, CERAI MATI)
      - 'pekerjaan'          (common Dukcapil-style job labels)
      - 'kewarganegaraan'    (WNI, WNA)
    """

    # OCR digit->letter fix (consistent with admin repo)
    _OCR_DIGIT_TO_LETTER = str.maketrans({
        '0': 'O', '1': 'I', '2': 'Z', '3': 'E', '4': 'A', '5': 'S', '6': 'G', '7': 'T', '8': 'B', '9': 'G'
    })

    # punctuation map: translate to space (no regex)
    _PUNCTS = string.punctuation + "·•—–‐-‒–—―…“”‘’´`´¨^~¨¸«»‹›•··"
    _PUNCT_TO_SPACE = str.maketrans({ch: " " for ch in _PUNCTS})

    # Canonical KTP field names (display labels)
    _REF_FIELDS: List[str] = [
        "Provinsi", "Kota", "NIK", "Nama", "Tempat_Tgl_Lahir", "Jenis_Kelamin",
        "Golongan_Darah", "Alamat", "RT_RW", "Kel_Desa", "Kecamatan", "Agama",
        "Status_Perkawinan", "Pekerjaan", "Kewarganegaraan", "Berlaku_Hingga",
        "Kota_Terbit", "Tanggal_Terbit"
    ]

    def __init__(self, sqlite_path: str | None = None):
        """Initialize the repository and ensure schema/seed so lookups work immediately.

        Args:
            sqlite_path (str | None): SQLite file path for persistence; None keeps data in-memory.

        Returns:
            None
        """
        self.conn = sqlite3.connect(sqlite_path or ":memory:")
        self.conn.execute("PRAGMA journal_mode=WAL;")
        self.conn.execute("PRAGMA synchronous=NORMAL;")
        self._ensure_schema()
        self._seed_defaults_if_empty()
        self._ensure_fields_table()
        self._seed_fields_if_empty()

    # ----------------------------
    # Schema & seeding
    # ----------------------------
    def _ensure_schema(self) -> None:
        """Create the reference-value table and indexes to support fast, scoped searches.

        Returns:
            None
        """
        cur = self.conn.cursor()
        cur.execute("""
        CREATE TABLE IF NOT EXISTS refs (
            category     TEXT NOT NULL,
            value_raw    TEXT NOT NULL,
            value_off    TEXT NOT NULL,
            norm_name    TEXT NOT NULL,
            norm_nospace TEXT NOT NULL,
            tokens       TEXT NOT NULL,
            trigrams     TEXT NOT NULL,
            PRIMARY KEY (category, value_off)
        )
        """)
        cur.execute("CREATE INDEX IF NOT EXISTS idx_refs_cat_norm ON refs(category, norm_name)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_refs_cat_nns  ON refs(category, norm_nospace)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_refs_cat_trg  ON refs(category, trigrams)")
        self.conn.commit()

    def _seed_defaults_if_empty(self) -> None:
        """Insert standard Indonesian sets once so the app has sensible defaults.

        Returns:
            None
        """
        cur = self.conn.cursor()
        cur.execute("SELECT COUNT(1) FROM refs")
        count, = cur.fetchone()
        if count:
            return

        data: Dict[str, List[str]] = {
            "jenis_kelamin": ["LAKI-LAKI", "PEREMPUAN"],
            "golongan_darah": ["A", "B", "AB", "O", "-"],
            "agama": [
                "ISLAM", "KRISTEN", "KATHOLIK", "HINDU", "BUDDHA", "KONGHUCU",
                "KEPERCAYAAN KEPADA TUHAN YME"
            ],
            "status_perkawinan": ["BELUM KAWIN", "KAWIN", "CERAI HIDUP", "CERAI MATI"],
            "pekerjaan": [
                "PELAJAR/MAHASISWA", "KARYAWAN SWASTA", "PNS", "WIRASWASTA",
                "GURU", "NELAYAN", "PETANI/PEKEBUN", "BURUH HARIAN LEPAS",
                "TIDAK BEKERJA", "IBU RUMAH TANGGA", "TNI", "POLRI", "PENSIUNAN"
            ],
            "kewarganegaraan": ["WNI", "WNA"],
        }

        rows = []
        for cat, values in data.items():
            for v in values:
                value_off = self._std_official(v)
                norm = self._norm_generic(v)
                nns = self._nospace(norm)
                tokens = self._tokens(norm)
                trigrams = self._trigrams(nns)
                rows.append((cat, v, value_off, norm, nns, tokens, trigrams))

        cur.executemany("""
            INSERT OR IGNORE INTO refs(category, value_raw, value_off, norm_name, norm_nospace, tokens, trigrams)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, rows)
        self.conn.commit()

    def _ensure_fields_table(self) -> None:
        """Create a tiny table to store KTP field names to enable get/search over fields.

        Returns:
            None
        """
        cur = self.conn.cursor()
        cur.execute("""
        CREATE TABLE IF NOT EXISTS ref_fields (
            field_key    TEXT PRIMARY KEY,
            field_off    TEXT NOT NULL,
            norm_name    TEXT NOT NULL,
            norm_nospace TEXT NOT NULL,
            tokens       TEXT NOT NULL,
            trigrams     TEXT NOT NULL
        )
        """)
        cur.execute("CREATE INDEX IF NOT EXISTS idx_rfields_norm ON ref_fields(norm_name)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_rfields_nns  ON ref_fields(norm_nospace)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_rfields_trg  ON ref_fields(trigrams)")
        self.conn.commit()

    def _seed_fields_if_empty(self) -> None:
        """Seed the canonical KTP field names so they are discoverable and searchable.

        Returns:
            None
        """
        cur = self.conn.cursor()
        cur.execute("SELECT COUNT(1) FROM ref_fields")
        count, = cur.fetchone()
        if count:
            return

        rows = []
        for f in self._REF_FIELDS:
            norm = self._norm_generic(f)
            nns = self._nospace(norm)
            tokens = self._tokens(norm)
            trigrams = self._trigrams(nns)
            rows.append((f, f, norm, nns, tokens, trigrams))

        cur.executemany("""
            INSERT OR IGNORE INTO ref_fields(field_key, field_off, norm_name, norm_nospace, tokens, trigrams)
            VALUES (?, ?, ?, ?, ?, ?)
        """, rows)
        self.conn.commit()

    # ----------------------------
    # Normalization & scoring (regex-free)
    # ----------------------------
    @staticmethod
    def _std_official(s: str) -> str:
        """Return a stable uppercase label so values render uniformly across the app.

        Args:
            s (str): Original text.

        Returns:
            str: Uppercased, whitespace-collapsed display form.
        """
        t = unicodedata.normalize("NFKC", s).strip()
        return " ".join(t.split()).upper()

    def _norm_generic(self, s: str) -> str:
        """Create a robust, regex-free normalized string to withstand OCR/punctuation noise.

        Args:
            s (str): Input text to normalize.

        Returns:
            str: Uppercased, punctuation-folded, whitespace-collapsed text with OCR fixes.
        """
        if not s:
            return ""
        t = unicodedata.normalize("NFKC", s)
        t = t.translate(self._OCR_DIGIT_TO_LETTER)
        t = t.translate(self._PUNCT_TO_SPACE)
        t = " ".join(t.split())
        return t.upper()

    @staticmethod
    def _nospace(s: str) -> str:
        """Drop whitespace from normalized text to make glued-token comparisons resilient.

        Args:
            s (str): Normalized text.

        Returns:
            str: Text with all spaces removed.
        """
        return "".join(ch for ch in s if not ch.isspace())

    @staticmethod
    def _tokens(norm: str) -> str:
        """Expose space-normalized tokens so token-level comparisons stay simple.

        Args:
            norm (str): Normalized text.

        Returns:
            str: Space-separated tokens (single spaces).
        """
        return " ".join(norm.split())

    @staticmethod
    def _trigrams(nns: str) -> str:
        """Produce pg_trgm-like character trigrams for cheap candidate narrowing.

        Args:
            nns (str): Normalized text without spaces.

        Returns:
            str: Space-padded trigram list, e.g. " abc bcd cde ".
        """
        s = f"  {nns}  "
        tris = [s[i:i+3] for i in range(len(s) - 2)]
        return " " + " ".join(tris) + " "

    @staticmethod
    def _lev(a: str, b: str) -> float:
        """Measure character-level similarity so near-duplicates rank higher.

        Args:
            a (str): First string.
            b (str): Second string.

        Returns:
            float: Ratio in [0, 1].
        """
        if not a or not b:
            return 0.0
        return difflib.SequenceMatcher(None, a.upper(), b.upper()).ratio()

    def _token_overlap(self, a: str, b: str) -> float:
        """Estimate shared vocabulary so semantically aligned values win ties.

        Args:
            a (str): First string.
            b (str): Second string.

        Returns:
            float: Overlap score in [0, 1].
        """
        A = set(self._norm_generic(a).split())
        B = set(self._norm_generic(b).split())
        if not A or not B:
            return 0.0
        return len(A & B) / max(1, len(A | B))

    def _score(self, q: str, cand: str) -> float:
        """Blend spaced/no-space similarity and token overlap into one robust score.

        Args:
            q (str): Query text.
            cand (str): Candidate text.

        Returns:
            float: Composite similarity score in [0, 1].
        """
        a1, b1 = self._norm_generic(q), self._norm_generic(cand)
        a2, b2 = self._nospace(a1), self._nospace(b1)
        return 0.35 * self._lev(a1, b1) + 0.45 * self._lev(a2, b2) + 0.20 * self._token_overlap(q, cand)

    # ----------------------------
    # Core helpers
    # ----------------------------
    def _row_to_ref(self, category: str, value_off: str) -> Ref:
        """Create a typed Ref from raw DB values so callers never touch rows.

        Args:
            category (str): Category key (e.g., 'agama').
            value_off (str): Canonical display value.

        Returns:
            Ref: Strongly-typed reference value.
        """
        return Ref(category=category, value=value_off)

    def _get_all(self, category: str) -> List[Ref]:
        """Return every value in a category so UIs can render full dropdowns.

        Args:
            category (str): Category key (e.g., 'pekerjaan').

        Returns:
            List[Ref]: All values as typed models, ordered by display label.
        """
        cur = self.conn.cursor()
        rows = cur.execute("""
            SELECT value_off FROM refs
            WHERE category=?
            ORDER BY value_off
        """, (category,)).fetchall()
        return [self._row_to_ref(category, r[0]) for r in rows]

    def _candidates(self, category: str, q: str, limit: int) -> list[sqlite3.Row]:
        """Fetch a small, likely set of values in a category to be scored precisely.

        Args:
            category (str): Category key (e.g., 'agama').
            q (str): Free-form user query.
            limit (int): Upper bound on candidates fetched for scoring.

        Returns:
            list[sqlite3.Row]: Raw rows restricted to the category and basic trigram/prefix filters.
        """
        self.conn.row_factory = sqlite3.Row
        cur = self.conn.cursor()

        norm = self._norm_generic(q)
        nns = self._nospace(norm)
        p3, q3 = norm[:3], nns[:3]

        tri_text = self._trigrams(nns)
        tri_list = tri_text.strip().split()[:4]

        where = ["category = ?"]
        params: list[Any] = [category]

        if p3:
            where.append("norm_name LIKE ?")
            params.append(f"{p3}%")
        if q3:
            where.append("norm_nospace LIKE ?")
            params.append(f"{q3}%")

        if tri_list:
            parts = []
            for tri in tri_list:
                parts.append("trigrams LIKE ?")
                params.append(f"% {tri} %")
            where.append("(" + " OR ".join(parts) + ")")

        sql = f"""
            SELECT value_off, norm_name, norm_nospace
            FROM refs
            WHERE {' AND '.join(where)}
            LIMIT {int(limit)}
        """
        rows = cur.execute(sql, tuple(params)).fetchall()

        if rows:
            return rows

        # Fallback: category-only with LIMIT
        sql2 = f"""
            SELECT value_off, norm_name, norm_nospace
            FROM refs
            WHERE category=?
            LIMIT {int(limit)}
        """
        return cur.execute(sql2, (category,)).fetchall()

    def _search(self, category: str, q: str, k: int) -> List[Tuple[Ref, float]]:
        """Rank and return the top-k matches in a category for suggestions/autocomplete.

        Args:
            category (str): Category key (e.g., 'status_perkawinan').
            q (str): Free-form user query.
            k (int): Number of top results to return.

        Returns:
            List[Tuple[Ref, float]]: Ranked (Ref, score) pairs, score in [0, 1].
        """
        rows = self._candidates(category, q, limit=max(200, k * 12))
        scored: List[Tuple[Ref, float]] = []
        for r in rows:
            val = r["value_off"]
            scored.append((self._row_to_ref(category, val), self._score(q, val)))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:k]

    # ----------------------------
    # Public API (reference field names)
    # ----------------------------
    def get_refs(self) -> List[str]:
        """Return all possible KTP field names so callers can present/validate inputs.

        Returns:
            List[str]: Field labels (e.g., "Nama", "Agama", "Kota_Terbit") in canonical form.
        """
        cur = self.conn.cursor()
        rows = cur.execute("""
            SELECT field_off FROM ref_fields
            ORDER BY field_off
        """).fetchall()
        return [r[0] for r in rows]

    def search_refs(self, q: str, k: int = 10) -> List[Tuple[str, float]]:
        """Search across KTP field names so UIs can quickly jump to the right field.

        Args:
            q (str): Free-form query (e.g., 'tempat lahir').
            k (int): Number of top results to return.

        Returns:
            List[Tuple[str, float]]: Ranked (field_name, score) pairs.
        """
        self.conn.row_factory = sqlite3.Row
        cur = self.conn.cursor()

        norm = self._norm_generic(q)
        nns = self._nospace(norm)
        p3, q3 = norm[:3], nns[:3]

        tri_text = self._trigrams(nns)
        tri_list = tri_text.strip().split()[:4]

        where = []
        params: list[Any] = []

        if p3:
            where.append("norm_name LIKE ?")
            params.append(f"{p3}%")
        if q3:
            where.append("norm_nospace LIKE ?")
            params.append(f"{q3}%")

        if tri_list:
            parts = []
            for tri in tri_list:
                parts.append("trigrams LIKE ?")
                params.append(f"% {tri} %")
            where.append("(" + " OR ".join(parts) + ")")

        sql = "SELECT field_off FROM ref_fields"
        if where:
            sql += f" WHERE {' AND '.join(where)}"
        sql += f" LIMIT {int(max(200, k * 12))}"

        rows = cur.execute(sql, tuple(params)).fetchall()
        values = [r["field_off"] for r in rows] if rows else self.get_refs()

        scored = [(v, self._score(q, v)) for v in values]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:k]

    # ----------------------------
    # Public API (category-specific wrappers)
    # ----------------------------
    def get_genders(self) -> List[Ref]:
        """All values for Jenis_Kelamin so dropdowns remain consistent.

        Returns:
            List[Ref]: Gender values as typed models.
        """
        return self._get_all("jenis_kelamin")

    def search_genders(self, q: str, k: int = 10) -> List[Tuple[Ref, float]]:
        """Find the closest Jenis_Kelamin values for a free-form query.

        Args:
            q (str): User query (e.g., 'prmpn').
            k (int): Top results to return.

        Returns:
            List[Tuple[Ref, float]]: Ranked (Ref, score) pairs.
        """
        return self._search("jenis_kelamin", q, k)

    def get_blood_types(self) -> List[Ref]:
        """All values for Golongan_Darah so input choices stay standardized.

        Returns:
            List[Ref]: Blood types as typed models.
        """
        return self._get_all("golongan_darah")

    def search_blood_types(self, q: str, k: int = 10) -> List[Tuple[Ref, float]]:
        """Find the closest Golongan_Darah values for a free-form query.

        Args:
            q (str): User query (e.g., 'ab').
            k (int): Top results to return.

        Returns:
            List[Tuple[Ref, float]]: Ranked (Ref, score) pairs.
        """
        return self._search("golongan_darah", q, k)

    def get_religions(self) -> List[Ref]:
        """All values for Agama to keep religious labels uniform.

        Returns:
            List[Ref]: Religions as typed models.
        """
        return self._get_all("agama")

    def search_religions(self, q: str, k: int = 10) -> List[Tuple[Ref, float]]:
        """Find the closest Agama values for a free-form query.

        Args:
            q (str): User query (e.g., 'katolik').
            k (int): Top results to return.

        Returns:
            List[Tuple[Ref, float]]: Ranked (Ref, score) pairs.
        """
        return self._search("agama", q, k)

    def get_marital_statuses(self) -> List[Ref]:
        """All values for Status_Perkawinan so forms align with Dukcapil wording.

        Returns:
            List[Ref]: Marital statuses as typed models.
        """
        return self._get_all("status_perkawinan")

    def search_marital_statuses(self, q: str, k: int = 10) -> List[Tuple[Ref, float]]:
        """Find the closest Status_Perkawinan values for a free-form query.

        Args:
            q (str): User query (e.g., 'blm kawin').
            k (int): Top results to return.

        Returns:
            List[Tuple[Ref, float]]: Ranked (Ref, score) pairs.
        """
        return self._search("status_perkawinan", q, k)

    def get_jobs(self) -> List[Ref]:
        """All values for Pekerjaan so occupation inputs remain normalized.

        Returns:
            List[Ref]: Occupation labels as typed models.
        """
        return self._get_all("pekerjaan")

    def search_jobs(self, q: str, k: int = 10) -> List[Tuple[Ref, float]]:
        """Find the closest Pekerjaan values for a free-form query.

        Args:
            q (str): User query (e.g., 'wira usaha').
            k (int): Top results to return.

        Returns:
            List[Tuple[Ref, float]]: Ranked (Ref, score) pairs.
        """
        return self._search("pekerjaan", q, k)

    def get_citizenships(self) -> List[Ref]:
        """All values for Kewarganegaraan so nationality stays consistent.

        Returns:
            List[Ref]: Citizenship labels as typed models.
        """
        return self._get_all("kewarganegaraan")

    def search_citizenships(self, q: str, k: int = 10) -> List[Tuple[Ref, float]]:
        """Find the closest Kewarganegaraan values for a free-form query.

        Args:
            q (str): User query (e.g., 'wnl').
            k (int): Top results to return.

        Returns:
            List[Tuple[Ref, float]]: Ranked (Ref, score) pairs.
        """
        return self._search("kewarganegaraan", q, k)


if __name__ == "__main__":
    repo = KTPReferenceRepository(sqlite_path="ktp_refs.db")

    print("\n== REF FIELDS ==")
    print("All refs:", repo.get_refs())
    print('Search refs ~ "tempat lahir":', repo.search_refs("tempat lahir"))

    print("\n== GET ALL ==")
    print("Genders:", [r.to_dict() for r in repo.get_genders()])
    print("Blood Types:", [r.to_dict() for r in repo.get_blood_types()])
    print("Religions:", [r.to_dict() for r in repo.get_religions()])
    print("Marital Statuses:", [r.to_dict() for r in repo.get_marital_statuses()])
    print("Jobs:", [r.to_dict() for r in repo.get_jobs()])
    print("Citizenships:", [r.to_dict() for r in repo.get_citizenships()])

    print("\n== SEARCH EXAMPLES ==")
    print('Religions ~ "katolik":', [(r.value, s) for r, s in repo.search_religions("katolik")])
    print('Genders ~ "prmpn":', [(r.value, s) for r, s in repo.search_genders("prmpn")])
    print('Blood Types ~ "ab":', [(r.value, s) for r, s in repo.search_blood_types("ab")])
    print('Marital Statuses ~ "blm kawin":', [(r.value, s) for r, s in repo.search_marital_statuses("blm kawin")])
    print('Jobs ~ "wira usaha":', [(r.value, s) for r, s in repo.search_jobs("wira usaha")])
    print('Citizenship ~ "wnl":', [(r.value, s) for r, s in repo.search_citizenships("wnl")])



# Design notes — KTPReferenceRepository usage with OCRParser (no AI/ML)

# Purpose:
#     Centralize non-administrative reference sets (gender, blood type, religion, marital status,
#     job, citizenship, and field labels) in SQLite to support fast "get all" and fuzzy search.

# Integration patterns:
#     • Dual-pass lookups from OCRParser:
#         1) Label-guided: when a label is detected, search only the next 1–8 tokens.
#         2) Global: if label is missing/uncertain, search the entire flattened token stream.
#            Accept top results only if score ≥ τ and tokens are not already used by another field.
#     • Alias dictionary:
#         - Maintain a small, curated alias table for common OCR confusions and abbreviations:
#           e.g., "PERMPUAN"→"PEREMPUAN", "KAWlN"→"KAWIN", "WNl"→"WNI", "AGM"→"AGAMA".
#           Normalize inputs with aliases before searching to raise hit rates.

# Search strategy (pg_trgm-ish without extensions):
#     • Candidate narrowing:
#         - Category-scoped prefix filters (first 3 chars) on normalized name and nospace.
#         - Up to 3–4 trigram LIKE filters.
#         - Limit candidates (≤ 50 ideal for enums) before scoring.
#     • Scoring (deterministic):
#         - Composite of Levenshtein(spaced) + Levenshtein(nospace) + token overlap.
#         - Keep values and scores for UI confidence or audit.

# Performance:
#     • Cache normalization and previous queries (LRU).
#     • Deduplicate identical phrases before searching.
#     • Keep tables small and indexed:
#         - (category, norm_name), (category, norm_nospace), (category, trigrams).

# UX/Output:
#     • Public getters return typed Ref models (not raw rows) for consistency with Region model.
#     • Provide get_refs()/search_refs() to help map free-form UI actions to the canonical field names.

# Thresholds (starting points):
#     • label-window acceptance ≥ 0.66; global acceptance ≥ 0.60.
#     • When multiple hits are close, prefer the one adjacent to its label; otherwise pick top score.
