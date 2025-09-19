# ktp_reference_repository.py

from __future__ import annotations

import sqlite3
import unicodedata
import difflib
import string
from dataclasses import dataclass, asdict
from functools import lru_cache
from typing import List, Tuple, Any, Dict, Optional


# ----------------------------
# Model
# ----------------------------
@dataclass(frozen=True)
class Ref:
    """Typed reference value inside a category (e.g., 'agama' -> 'ISLAM').

    What:
        Represents one canonical option in a logical KTP category.

    Why:
        Typed values make APIs consistent and discourage direct sqlite row handling in callers.
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
    """SQLite-backed repository for non-administrative KTP reference fields.

    What:
        Centralized store of small, finite vocabularies (gender, blood type, religion,
        marital status, jobs, citizenship, and canonical field labels) with fast fuzzy search.

    Why:
        Keeps enumerations uniform across the app and provides resilient matching for OCR noise.
    """

    # OCR digit->letter fix (consistent with admin repo)
    _OCR_DIGIT_TO_LETTER = str.maketrans({
        '0': 'O', '1': 'I', '2': 'Z', '3': 'E', '4': 'A', '5': 'S', '6': 'G', '7': 'T', '8': 'B'
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
        """Initialize store and seed defaults.

        Why:
            Reference sets must be available immediately for both dropdowns and fuzzy lookups.

        Args:
            sqlite_path: SQLite file path (persistent) or None (in-memory).
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
        """Create the reference-value table and indices.

        Why:
            Category + normalized feature indices keep searches deterministic and fast.
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
        """Insert standard Indonesian sets once.

        Why:
            Real KTPs use a small, known vocabulary—seeding unlocks immediate matching.
        """
        cur = self.conn.cursor()
        cur.execute("SELECT COUNT(1) FROM refs")
        count, = cur.fetchone()
        if count:
            return

        # Expanded jobs for better recall (common Dukcapil-style labels and synonyms)
        jobs = [
            "PELAJAR/MAHASISWA", "KARYAWAN SWASTA", "PEGAWAI SWASTA", "SWASTA",
            "PNS", "PEGAWAI NEGERI SIPIL", "HONORER", "WIRASWASTA",
            "GURU", "DOSEN", "NELAYAN", "PETANI/PEKEBUN", "BURUH HARIAN LEPAS",
            "SOPIR", "PERAWAT", "TNI", "POLRI", "PENSIUNAN", "IBU RUMAH TANGGA",
            "TIDAK BEKERJA"
        ]

        data: Dict[str, List[str]] = {
            "jenis_kelamin": ["LAKI-LAKI", "PEREMPUAN"],
            "golongan_darah": ["A", "B", "AB", "O", "-"],
            # Keep canonical religion labels compact; parser may map synonyms to these.
            "agama": [
                "ISLAM", "KRISTEN", "KATHOLIK", "HINDU", "BUDDHA", "KONGHUCU",
                "KEPERCAYAAN KEPADA TUHAN YME"
            ],
            "status_perkawinan": ["BELUM KAWIN", "KAWIN", "CERAI HIDUP", "CERAI MATI"],
            "pekerjaan": jobs,
            "kewarganegaraan": ["WNI", "WNA"],
        }

        rows: List[Tuple[str, str, str, str, str, str, str]] = []
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
        """Create the field-names table for KTP schema discovery.

        Why:
            Lets callers search/display canonical field labels consistently.
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
        """Seed canonical KTP field names.

        Why:
            Enables search and validation against the app’s field vocabulary.
        """
        cur = self.conn.cursor()
        cur.execute("SELECT COUNT(1) FROM ref_fields")
        count, = cur.fetchone()
        if count:
            return

        rows: List[Tuple[str, str, str, str, str, str]] = []
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
    @lru_cache(maxsize=8192)
    def _std_official(s: str) -> str:
        """Return a stable uppercase label for consistent display.

        Args:
            s: Original text.

        Returns:
            Uppercased, whitespace-collapsed text.
        """
        t = unicodedata.normalize("NFKC", s).strip()
        return " ".join(t.split()).upper()

    @classmethod
    @lru_cache(maxsize=65536)
    def _norm_generic(cls, s: str) -> str:
        """Normalize a string for robust, deterministic search.

        Why:
            Dampens OCR/punctuation noise without relying on regex-heavy pipelines.

        Args:
            s: Input text.

        Returns:
            Uppercased, punctuation-folded, whitespace-collapsed string with OCR fixes.
        """
        if not s:
            return ""
        t = unicodedata.normalize("NFKC", s)
        t = t.translate(cls._OCR_DIGIT_TO_LETTER)
        t = t.translate(cls._PUNCT_TO_SPACE)
        t = " ".join(t.split())
        return t.upper()

    @staticmethod
    @lru_cache(maxsize=65536)
    def _nospace(s: str) -> str:
        """Drop whitespace from normalized text to tolerate glued tokens."""
        return "".join(ch for ch in s if not ch.isspace())

    @staticmethod
    @lru_cache(maxsize=65536)
    def _tokens(norm: str) -> str:
        """Expose space-normalized tokens for simple token-level comparisons."""
        return " ".join(norm.split())

    @staticmethod
    @lru_cache(maxsize=65536)
    def _trigrams(nns: str) -> str:
        """Produce pg_trgm-like trigrams for cheap candidate narrowing."""
        s = f"  {nns}  "
        tris = [s[i:i+3] for i in range(len(s) - 2)]
        return " " + " ".join(tris) + " "

    @staticmethod
    @lru_cache(maxsize=65536)
    def _lev(a: str, b: str) -> float:
        """Character-level similarity ratio in [0, 1]."""
        if not a or not b:
            return 0.0
        return difflib.SequenceMatcher(None, a.upper(), b.upper()).ratio()

    def _token_overlap(self, a: str, b: str) -> float:
        """Estimate shared vocabulary to encourage semantically aligned values.

        Args:
            a: First string.
            b: Second string.

        Returns:
            Overlap score in [0, 1].
        """
        A = set(self._norm_generic(a).split())
        B = set(self._norm_generic(b).split())
        if not A or not B:
            return 0.0
        return len(A & B) / max(1, len(A | B))

    def _score(self, q: str, cand: str) -> float:
        """Blend spaced/no-space similarity and token overlap into one score.

        Why:
            Balances letter-level noise tolerance with semantic hints.

        Args:
            q: Query text.
            cand: Candidate text.

        Returns:
            Composite similarity score in [0, 1].
        """
        a1, b1 = self._norm_generic(q), self._norm_generic(cand)
        a2, b2 = self._nospace(a1), self._nospace(b1)
        return 0.35 * self._lev(a1, b1) + 0.45 * self._lev(a2, b2) + 0.20 * self._token_overlap(q, cand)

    # ----------------------------
    # Core helpers
    # ----------------------------
    def _row_to_ref(self, category: str, value_off: str) -> Ref:
        """Create a typed Ref from canonical DB value."""
        return Ref(category=category, value=value_off)

    def _get_all(self, category: str) -> List[Ref]:
        """Return every value in a category.

        Why:
            Ideal for building dropdowns and validating inputs.

        Args:
            category: Category key (e.g., 'pekerjaan').

        Returns:
            All values as typed Ref models, ordered by display label.
        """
        cur = self.conn.cursor()
        rows = cur.execute("""
            SELECT value_off FROM refs
            WHERE category=?
            ORDER BY value_off
        """, (category,)).fetchall()
        return [self._row_to_ref(category, r[0]) for r in rows]

    def _candidates(self, category: str, q: str, limit: int) -> List[sqlite3.Row]:
        """Fetch a small, likely set of values in a category to be scored precisely.

        Why:
            Narrowing at SQL level keeps the Python ranking stage small and quick.

        Args:
            category: Category key (e.g., 'agama').
            q: Free-form user query.
            limit: Upper bound on candidates fetched for scoring.

        Returns:
            Raw rows restricted to the category with basic trigram/prefix filters.
        """
        self.conn.row_factory = sqlite3.Row
        cur = self.conn.cursor()

        norm = self._norm_generic(q)
        nns = self._nospace(norm)
        qlen = len(nns)
        p3, q3 = norm[:3], nns[:3]

        tri_text = self._trigrams(nns)
        tri_list = tri_text.strip().split()[:4]

        where = ["category = ?"]
        params: List[Any] = [category]

        if p3:
            where.append("norm_name LIKE ?")
            params.append(f"{p3}%")
        # Loosen for very short queries (≤ 3 chars): skip nospace and trigram predicates
        if q3 and qlen > 3:
            where.append("norm_nospace LIKE ?")
            params.append(f"{q3}%")

        if tri_list and qlen > 3:
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
        """Rank and return the top-k matches in a category.

        Why:
            Keeps matching robust to OCR noise while staying explainable.

        Args:
            category: Category key (e.g., 'status_perkawinan').
            q: Free-form query.
            k: Number of top results.

        Returns:
            Ranked (Ref, score) pairs in [0, 1].
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
        """Return all canonical KTP field names.

        Why:
            Allows UIs and parsers to align on a single vocabulary.
        """
        cur = self.conn.cursor()
        rows = cur.execute("""
            SELECT field_off FROM ref_fields
            ORDER BY field_off
        """).fetchall()
        return [r[0] for r in rows]

    def search_refs(self, q: str, k: int = 10) -> List[Tuple[str, float]]:
        """Search across KTP field names.

        Args:
            q: Free-form query (e.g., 'tempat lahir').
            k: Number of results.

        Returns:
            Ranked (field_name, score) pairs.
        """
        self.conn.row_factory = sqlite3.Row
        cur = self.conn.cursor()

        norm = self._norm_generic(q)
        nns = self._nospace(norm)
        qlen = len(nns)
        p3, q3 = norm[:3], nns[:3]

        tri_text = self._trigrams(nns)
        tri_list = tri_text.strip().split()[:4]

        where = []
        params: List[Any] = []

        if p3:
            where.append("norm_name LIKE ?")
            params.append(f"{p3}%")
        if q3 and qlen > 3:
            where.append("norm_nospace LIKE ?")
            params.append(f"{q3}%")

        if tri_list and qlen > 3:
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
        """All values for Jenis_Kelamin."""
        return self._get_all("jenis_kelamin")

    def search_genders(self, q: str, k: int = 10) -> List[Tuple[Ref, float]]:
        """Find the closest Jenis_Kelamin values for a free-form query."""
        return self._search("jenis_kelamin", q, k)

    def get_blood_types(self) -> List[Ref]:
        """All values for Golongan_Darah."""
        return self._get_all("golongan_darah")

    def search_blood_types(self, q: str, k: int = 10) -> List[Tuple[Ref, float]]:
        """Find the closest Golongan_Darah values for a free-form query."""
        return self._search("golongan_darah", q, k)

    def get_religions(self) -> List[Ref]:
        """All values for Agama."""
        return self._get_all("agama")

    def search_religions(self, q: str, k: int = 10) -> List[Tuple[Ref, float]]:
        """Find the closest Agama values for a free-form query."""
        return self._search("agama", q, k)

    def get_marital_statuses(self) -> List[Ref]:
        """All values for Status_Perkawinan."""
        return self._get_all("status_perkawinan")

    def search_marital_statuses(self, q: str, k: int = 10) -> List[Tuple[Ref, float]]:
        """Find the closest Status_Perkawinan values for a free-form query."""
        return self._search("status_perkawinan", q, k)

    def get_jobs(self) -> List[Ref]:
        """All values for Pekerjaan."""
        return self._get_all("pekerjaan")

    def search_jobs(self, q: str, k: int = 10) -> List[Tuple[Ref, float]]:
        """Find the closest Pekerjaan values for a free-form query."""
        return self._search("pekerjaan", q, k)

    def get_citizenships(self) -> List[Ref]:
        """All values for Kewarganegaraan."""
        return self._get_all("kewarganegaraan")

    def search_citizenships(self, q: str, k: int = 10) -> List[Tuple[Ref, float]]:
        """Find the closest Kewarganegaraan values for a free-form query."""
        return self._search("kewarganegaraan", q, k)


if __name__ == "__main__":
    # Quick standalone sanity check for the refs repository.
    # Run this file directly to verify seeding and fuzzy search behavior.
    repo = KTPReferenceRepository(sqlite_path="ktp_refs.db")

    print("\n== REF FIELDS ==")
    print("All refs:", repo.get_refs())
    print('Search refs ~ "tempat lahir":', repo.search_refs("tempat lahir"))

    print("\n== GET ALL ==")
    print("Genders:", [r.to_dict() for r in repo.get_genders()])
    print("Blood Types:", [r.to_dict() for r in repo.get_blood_types()])
    print("Religions:", [r.to_dict() for r in repo.get_religions()])
    print("Marital Statuses:", [r.to_dict() for r in repo.get_marital_statuses()])
    print("Jobs:", [r.to_dict() for r in repo.get_jobs()][:8], "... (total:", len(repo.get_jobs()), ")")
    print("Citizenships:", [r.to_dict() for r in repo.get_citizenships()])

    print("\n== SEARCH EXAMPLES ==")
    print('Religions ~ "katolik":', [(r.value, f"{s:.3f}") for r, s in repo.search_religions("katolik")])
    print('Genders ~ "prmpn":', [(r.value, f"{s:.3f}") for r, s in repo.search_genders("prmpn")])
    print('Blood Types ~ "ab":', [(r.value, f"{s:.3f}") for r, s in repo.search_blood_types("ab")])
    print('Marital ~ "blm kawin":', [(r.value, f"{s:.3f}") for r, s in repo.search_marital_statuses("blm kawin")])
    print('Jobs ~ "wira usaha":', [(r.value, f"{s:.3f}") for r, s in repo.search_jobs("wira usaha")][:5])
    print('Citizenship ~ "wnl":', [(r.value, f"{s:.3f}") for r, s in repo.search_citizenships("wnl")])
