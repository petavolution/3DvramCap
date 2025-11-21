#!/usr/bin/env python3
"""
Asset Database
==============
SQLite-based tracking for captures, meshes, and exports.

Features:
    - Track all captures and their processing status
    - Mesh deduplication via geometry hash lookup
    - Export history and validation results
    - Search and query assets
    - Session and job tracking

Usage:
    from core_database import AssetDatabase, get_database

    db = get_database()

    # Add a capture
    capture_id = db.add_capture("scene.rdc", "/path/to/scene.rdc")

    # Add extracted meshes
    mesh_id = db.add_mesh("mesh_001", capture_id, geometry_hash="abc123")

    # Query
    meshes = db.find_meshes_by_hash("abc123")
    captures = db.list_captures(status="completed")

Author: Capture Pipeline
"""

import sqlite3
import json
import hashlib
from contextlib import contextmanager
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple
from enum import Enum


class CaptureStatus(Enum):
    """Status of a capture in the pipeline."""
    PENDING = "pending"
    EXTRACTING = "extracting"
    EXTRACTED = "extracted"
    PROCESSING = "processing"
    PROCESSED = "processed"
    EXPORTING = "exporting"
    EXPORTED = "exported"
    VALIDATED = "validated"
    FAILED = "failed"


class MeshStatus(Enum):
    """Status of a mesh."""
    RAW = "raw"
    DEDUPLICATED = "deduplicated"
    OPTIMIZED = "optimized"
    EXPORTED = "exported"


@dataclass
class CaptureRecord:
    """Database record for a capture."""
    id: int
    name: str
    filepath: str
    status: str
    created_at: str
    updated_at: str
    mesh_count: int = 0
    texture_count: int = 0
    metadata: Dict = None

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> 'CaptureRecord':
        return cls(
            id=row['id'],
            name=row['name'],
            filepath=row['filepath'],
            status=row['status'],
            created_at=row['created_at'],
            updated_at=row['updated_at'],
            mesh_count=row['mesh_count'],
            texture_count=row['texture_count'],
            metadata=json.loads(row['metadata']) if row['metadata'] else {}
        )


@dataclass
class MeshRecord:
    """Database record for a mesh."""
    id: int
    name: str
    capture_id: int
    filepath: str
    geometry_hash: str
    vertex_count: int
    face_count: int
    status: str
    is_duplicate: bool
    canonical_id: Optional[int]
    created_at: str
    metadata: Dict = None

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> 'MeshRecord':
        return cls(
            id=row['id'],
            name=row['name'],
            capture_id=row['capture_id'],
            filepath=row['filepath'],
            geometry_hash=row['geometry_hash'],
            vertex_count=row['vertex_count'],
            face_count=row['face_count'],
            status=row['status'],
            is_duplicate=bool(row['is_duplicate']),
            canonical_id=row['canonical_id'],
            created_at=row['created_at'],
            metadata=json.loads(row['metadata']) if row['metadata'] else {}
        )


@dataclass
class ExportRecord:
    """Database record for an export."""
    id: int
    capture_id: int
    format: str
    filepath: str
    status: str
    validation_score: Optional[float]
    created_at: str
    metadata: Dict = None

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> 'ExportRecord':
        return cls(
            id=row['id'],
            capture_id=row['capture_id'],
            format=row['format'],
            filepath=row['filepath'],
            status=row['status'],
            validation_score=row['validation_score'],
            created_at=row['created_at'],
            metadata=json.loads(row['metadata']) if row['metadata'] else {}
        )


class AssetDatabase:
    """SQLite database for asset tracking."""

    SCHEMA_VERSION = 1

    def __init__(self, db_path: str = "pipeline.db"):
        self.db_path = db_path
        self._conn: Optional[sqlite3.Connection] = None
        self._initialize()

    def _initialize(self):
        """Initialize database schema."""
        with self._connection() as conn:
            conn.executescript('''
                -- Captures table
                CREATE TABLE IF NOT EXISTS captures (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    filepath TEXT UNIQUE NOT NULL,
                    status TEXT DEFAULT 'pending',
                    mesh_count INTEGER DEFAULT 0,
                    texture_count INTEGER DEFAULT 0,
                    metadata TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
                );

                -- Meshes table
                CREATE TABLE IF NOT EXISTS meshes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    capture_id INTEGER,
                    filepath TEXT,
                    geometry_hash TEXT,
                    vertex_count INTEGER DEFAULT 0,
                    face_count INTEGER DEFAULT 0,
                    status TEXT DEFAULT 'raw',
                    is_duplicate INTEGER DEFAULT 0,
                    canonical_id INTEGER,
                    metadata TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (capture_id) REFERENCES captures(id),
                    FOREIGN KEY (canonical_id) REFERENCES meshes(id)
                );

                -- Exports table
                CREATE TABLE IF NOT EXISTS exports (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    capture_id INTEGER,
                    format TEXT NOT NULL,
                    filepath TEXT NOT NULL,
                    status TEXT DEFAULT 'pending',
                    validation_score REAL,
                    metadata TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (capture_id) REFERENCES captures(id)
                );

                -- Jobs table for tracking processing jobs
                CREATE TABLE IF NOT EXISTS jobs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    capture_id INTEGER,
                    job_type TEXT NOT NULL,
                    status TEXT DEFAULT 'pending',
                    started_at TEXT,
                    completed_at TEXT,
                    error_message TEXT,
                    metadata TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (capture_id) REFERENCES captures(id)
                );

                -- Sessions table for tracking pipeline sessions
                CREATE TABLE IF NOT EXISTS sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT UNIQUE NOT NULL,
                    started_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    ended_at TEXT,
                    captures_processed INTEGER DEFAULT 0,
                    status TEXT DEFAULT 'active',
                    config TEXT
                );

                -- Geometry hash index for fast deduplication
                CREATE INDEX IF NOT EXISTS idx_meshes_geometry_hash ON meshes(geometry_hash);

                -- Status indexes
                CREATE INDEX IF NOT EXISTS idx_captures_status ON captures(status);
                CREATE INDEX IF NOT EXISTS idx_meshes_status ON meshes(status);

                -- Schema version
                CREATE TABLE IF NOT EXISTS schema_version (
                    version INTEGER PRIMARY KEY
                );
            ''')

            # Check/update schema version
            cursor = conn.execute('SELECT version FROM schema_version LIMIT 1')
            row = cursor.fetchone()
            if row is None:
                conn.execute('INSERT INTO schema_version (version) VALUES (?)',
                           (self.SCHEMA_VERSION,))

    @contextmanager
    def _connection(self):
        """Get database connection with row factory."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    # =========================================================================
    # Captures
    # =========================================================================

    def add_capture(self, name: str, filepath: str,
                    metadata: Optional[Dict] = None) -> int:
        """Add a new capture to the database."""
        with self._connection() as conn:
            cursor = conn.execute('''
                INSERT INTO captures (name, filepath, metadata)
                VALUES (?, ?, ?)
            ''', (name, filepath, json.dumps(metadata) if metadata else None))
            return cursor.lastrowid

    def get_capture(self, capture_id: int) -> Optional[CaptureRecord]:
        """Get capture by ID."""
        with self._connection() as conn:
            cursor = conn.execute(
                'SELECT * FROM captures WHERE id = ?', (capture_id,))
            row = cursor.fetchone()
            return CaptureRecord.from_row(row) if row else None

    def get_capture_by_path(self, filepath: str) -> Optional[CaptureRecord]:
        """Get capture by file path."""
        with self._connection() as conn:
            cursor = conn.execute(
                'SELECT * FROM captures WHERE filepath = ?', (filepath,))
            row = cursor.fetchone()
            return CaptureRecord.from_row(row) if row else None

    def update_capture_status(self, capture_id: int, status: CaptureStatus,
                              mesh_count: int = None, texture_count: int = None):
        """Update capture status."""
        with self._connection() as conn:
            updates = ['status = ?', 'updated_at = CURRENT_TIMESTAMP']
            params = [status.value]

            if mesh_count is not None:
                updates.append('mesh_count = ?')
                params.append(mesh_count)
            if texture_count is not None:
                updates.append('texture_count = ?')
                params.append(texture_count)

            params.append(capture_id)
            conn.execute(f'''
                UPDATE captures SET {', '.join(updates)} WHERE id = ?
            ''', params)

    def list_captures(self, status: str = None, limit: int = 100) -> List[CaptureRecord]:
        """List captures with optional status filter."""
        with self._connection() as conn:
            if status:
                cursor = conn.execute(
                    'SELECT * FROM captures WHERE status = ? ORDER BY created_at DESC LIMIT ?',
                    (status, limit))
            else:
                cursor = conn.execute(
                    'SELECT * FROM captures ORDER BY created_at DESC LIMIT ?',
                    (limit,))
            return [CaptureRecord.from_row(row) for row in cursor.fetchall()]

    def delete_capture(self, capture_id: int):
        """Delete a capture and its related records."""
        with self._connection() as conn:
            conn.execute('DELETE FROM meshes WHERE capture_id = ?', (capture_id,))
            conn.execute('DELETE FROM exports WHERE capture_id = ?', (capture_id,))
            conn.execute('DELETE FROM jobs WHERE capture_id = ?', (capture_id,))
            conn.execute('DELETE FROM captures WHERE id = ?', (capture_id,))

    # =========================================================================
    # Meshes
    # =========================================================================

    def add_mesh(self, name: str, capture_id: int,
                 filepath: str = "",
                 geometry_hash: str = "",
                 vertex_count: int = 0,
                 face_count: int = 0,
                 metadata: Optional[Dict] = None) -> int:
        """Add a mesh to the database."""
        with self._connection() as conn:
            cursor = conn.execute('''
                INSERT INTO meshes (name, capture_id, filepath, geometry_hash,
                                   vertex_count, face_count, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (name, capture_id, filepath, geometry_hash,
                  vertex_count, face_count, json.dumps(metadata) if metadata else None))
            return cursor.lastrowid

    def get_mesh(self, mesh_id: int) -> Optional[MeshRecord]:
        """Get mesh by ID."""
        with self._connection() as conn:
            cursor = conn.execute('SELECT * FROM meshes WHERE id = ?', (mesh_id,))
            row = cursor.fetchone()
            return MeshRecord.from_row(row) if row else None

    def find_meshes_by_hash(self, geometry_hash: str) -> List[MeshRecord]:
        """Find all meshes with a given geometry hash."""
        with self._connection() as conn:
            cursor = conn.execute(
                'SELECT * FROM meshes WHERE geometry_hash = ?', (geometry_hash,))
            return [MeshRecord.from_row(row) for row in cursor.fetchall()]

    def find_canonical_mesh(self, geometry_hash: str) -> Optional[MeshRecord]:
        """Find the canonical (non-duplicate) mesh for a hash."""
        with self._connection() as conn:
            cursor = conn.execute('''
                SELECT * FROM meshes
                WHERE geometry_hash = ? AND is_duplicate = 0
                ORDER BY vertex_count DESC
                LIMIT 1
            ''', (geometry_hash,))
            row = cursor.fetchone()
            return MeshRecord.from_row(row) if row else None

    def mark_as_duplicate(self, mesh_id: int, canonical_id: int):
        """Mark a mesh as duplicate of another."""
        with self._connection() as conn:
            conn.execute('''
                UPDATE meshes SET is_duplicate = 1, canonical_id = ?
                WHERE id = ?
            ''', (canonical_id, mesh_id))

    def update_mesh_status(self, mesh_id: int, status: MeshStatus):
        """Update mesh status."""
        with self._connection() as conn:
            conn.execute(
                'UPDATE meshes SET status = ? WHERE id = ?',
                (status.value, mesh_id))

    def list_meshes(self, capture_id: int = None,
                    include_duplicates: bool = False,
                    limit: int = 1000) -> List[MeshRecord]:
        """List meshes with optional filters."""
        with self._connection() as conn:
            conditions = []
            params = []

            if capture_id is not None:
                conditions.append('capture_id = ?')
                params.append(capture_id)

            if not include_duplicates:
                conditions.append('is_duplicate = 0')

            where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
            params.append(limit)

            cursor = conn.execute(f'''
                SELECT * FROM meshes {where}
                ORDER BY created_at DESC LIMIT ?
            ''', params)
            return [MeshRecord.from_row(row) for row in cursor.fetchall()]

    def get_unique_mesh_count(self) -> int:
        """Get count of unique (non-duplicate) meshes."""
        with self._connection() as conn:
            cursor = conn.execute(
                'SELECT COUNT(*) FROM meshes WHERE is_duplicate = 0')
            return cursor.fetchone()[0]

    # =========================================================================
    # Exports
    # =========================================================================

    def add_export(self, capture_id: int, format: str, filepath: str,
                   metadata: Optional[Dict] = None) -> int:
        """Add an export record."""
        with self._connection() as conn:
            cursor = conn.execute('''
                INSERT INTO exports (capture_id, format, filepath, metadata)
                VALUES (?, ?, ?, ?)
            ''', (capture_id, format, filepath, json.dumps(metadata) if metadata else None))
            return cursor.lastrowid

    def update_export_status(self, export_id: int, status: str,
                            validation_score: float = None):
        """Update export status and validation score."""
        with self._connection() as conn:
            if validation_score is not None:
                conn.execute('''
                    UPDATE exports SET status = ?, validation_score = ?
                    WHERE id = ?
                ''', (status, validation_score, export_id))
            else:
                conn.execute(
                    'UPDATE exports SET status = ? WHERE id = ?',
                    (status, export_id))

    def get_exports(self, capture_id: int) -> List[ExportRecord]:
        """Get all exports for a capture."""
        with self._connection() as conn:
            cursor = conn.execute(
                'SELECT * FROM exports WHERE capture_id = ?', (capture_id,))
            return [ExportRecord.from_row(row) for row in cursor.fetchall()]

    # =========================================================================
    # Jobs
    # =========================================================================

    def create_job(self, capture_id: int, job_type: str,
                   metadata: Optional[Dict] = None) -> int:
        """Create a processing job."""
        with self._connection() as conn:
            cursor = conn.execute('''
                INSERT INTO jobs (capture_id, job_type, metadata)
                VALUES (?, ?, ?)
            ''', (capture_id, job_type, json.dumps(metadata) if metadata else None))
            return cursor.lastrowid

    def start_job(self, job_id: int):
        """Mark job as started."""
        with self._connection() as conn:
            conn.execute('''
                UPDATE jobs SET status = 'running', started_at = CURRENT_TIMESTAMP
                WHERE id = ?
            ''', (job_id,))

    def complete_job(self, job_id: int, success: bool = True, error: str = None):
        """Mark job as completed."""
        status = 'completed' if success else 'failed'
        with self._connection() as conn:
            conn.execute('''
                UPDATE jobs SET status = ?, completed_at = CURRENT_TIMESTAMP,
                error_message = ?
                WHERE id = ?
            ''', (status, error, job_id))

    # =========================================================================
    # Sessions
    # =========================================================================

    def create_session(self, session_id: str, config: Dict = None) -> int:
        """Create a pipeline session."""
        with self._connection() as conn:
            cursor = conn.execute('''
                INSERT INTO sessions (session_id, config)
                VALUES (?, ?)
            ''', (session_id, json.dumps(config) if config else None))
            return cursor.lastrowid

    def end_session(self, session_id: str, captures_processed: int):
        """End a pipeline session."""
        with self._connection() as conn:
            conn.execute('''
                UPDATE sessions SET status = 'completed',
                ended_at = CURRENT_TIMESTAMP, captures_processed = ?
                WHERE session_id = ?
            ''', (captures_processed, session_id))

    # =========================================================================
    # Statistics
    # =========================================================================

    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        with self._connection() as conn:
            stats = {}

            # Capture counts by status
            cursor = conn.execute('''
                SELECT status, COUNT(*) FROM captures GROUP BY status
            ''')
            stats['captures_by_status'] = dict(cursor.fetchall())

            # Total counts
            cursor = conn.execute('SELECT COUNT(*) FROM captures')
            stats['total_captures'] = cursor.fetchone()[0]

            cursor = conn.execute('SELECT COUNT(*) FROM meshes')
            stats['total_meshes'] = cursor.fetchone()[0]

            cursor = conn.execute('SELECT COUNT(*) FROM meshes WHERE is_duplicate = 0')
            stats['unique_meshes'] = cursor.fetchone()[0]

            cursor = conn.execute('SELECT COUNT(*) FROM exports')
            stats['total_exports'] = cursor.fetchone()[0]

            # Vertex/face totals
            cursor = conn.execute('''
                SELECT SUM(vertex_count), SUM(face_count)
                FROM meshes WHERE is_duplicate = 0
            ''')
            row = cursor.fetchone()
            stats['total_vertices'] = row[0] or 0
            stats['total_faces'] = row[1] or 0

            return stats


# Global database instance
_global_db: Optional[AssetDatabase] = None


def get_database(db_path: str = None) -> AssetDatabase:
    """Get global database instance."""
    global _global_db

    if _global_db is None or (db_path and db_path != _global_db.db_path):
        _global_db = AssetDatabase(db_path or "pipeline.db")

    return _global_db


def init_database(db_path: str = "pipeline.db") -> AssetDatabase:
    """Initialize and return database."""
    global _global_db
    _global_db = AssetDatabase(db_path)
    return _global_db


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Asset database management")
    parser.add_argument('--db', default='pipeline.db', help="Database path")
    parser.add_argument('--stats', action='store_true', help="Show statistics")
    parser.add_argument('--list-captures', action='store_true', help="List captures")
    parser.add_argument('--status', help="Filter by status")

    args = parser.parse_args()

    db = AssetDatabase(args.db)

    if args.stats:
        stats = db.get_stats()
        print(json.dumps(stats, indent=2))

    elif args.list_captures:
        captures = db.list_captures(status=args.status)
        for c in captures:
            print(f"[{c.id}] {c.name} - {c.status} ({c.mesh_count} meshes)")
