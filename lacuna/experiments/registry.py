"""
Run registry: data model and persistence for experiment tracking.

Provides RunEntry (dataclass) and RunRegistry (CRUD + atomic JSON persistence).
Registry path is always passed explicitly -- no global state (Rule 5).
"""

import json
import os
import re
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional


# Valid status values -- ordered by progression.
VALID_STATUSES = ("training", "evaluated", "calibrated")

# Pattern for run IDs: RUN- followed by 3 or more digits.
_RUN_ID_PATTERN = re.compile(r"^RUN-\d{3,}$")


class RegistryError(Exception):
    """Raised on registry lookup failures or validation errors."""


@dataclass
class RunEntry:
    """A single experiment run record."""

    run_id: str
    folder_name: str
    folder_path: str
    timestamp: str
    config_path: str
    status: str
    description: str
    metrics: Dict[str, float] = field(default_factory=dict)
    mnar_variants: List[str] = field(default_factory=list)
    n_experts: int = 0
    tags: List[str] = field(default_factory=list)

    def __post_init__(self):
        """Validate fields on construction -- fail fast (Rule 1)."""
        if not _RUN_ID_PATTERN.match(self.run_id):
            raise RegistryError(
                f"Invalid run_id '{self.run_id}': must match RUN-\\d{{3,}}"
            )
        if self.status not in VALID_STATUSES:
            raise RegistryError(
                f"Invalid status '{self.status}': must be one of {VALID_STATUSES}"
            )
        if not self.folder_name:
            raise RegistryError("folder_name must not be empty")
        if not self.folder_path:
            raise RegistryError("folder_path must not be empty")

    def to_dict(self) -> dict:
        """Serialize to a plain dict suitable for JSON."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "RunEntry":
        """Deserialize from a dict.  Raises RegistryError on invalid data."""
        required_keys = {"run_id", "folder_name", "folder_path", "timestamp",
                         "config_path", "status", "description"}
        missing = required_keys - set(data.keys())
        if missing:
            raise RegistryError(f"Missing required keys: {missing}")
        return cls(
            run_id=data["run_id"],
            folder_name=data["folder_name"],
            folder_path=data["folder_path"],
            timestamp=data["timestamp"],
            config_path=data["config_path"],
            status=data["status"],
            description=data["description"],
            metrics=data.get("metrics", {}),
            mnar_variants=data.get("mnar_variants", []),
            n_experts=data.get("n_experts", 0),
            tags=data.get("tags", []),
        )


class RunRegistry:
    """
    Persistent registry of experiment runs backed by a JSON file.

    Parameters
    ----------
    registry_path : Path
        Explicit path to the JSON file (no global state -- Rule 5).
    """

    def __init__(self, registry_path: Path):
        if not isinstance(registry_path, Path):
            registry_path = Path(registry_path)
        self._path = registry_path
        self._entries: Dict[str, RunEntry] = {}

    # -- persistence --------------------------------------------------------

    def load(self) -> None:
        """Load entries from disk.  Creates empty registry if file absent."""
        if not self._path.exists():
            self._entries = {}
            return
        try:
            text = self._path.read_text(encoding="utf-8")
            raw = json.loads(text)
        except (json.JSONDecodeError, OSError) as exc:
            raise RegistryError(
                f"Failed to read registry at {self._path}: {exc}"
            ) from exc
        if not isinstance(raw, dict) or "runs" not in raw:
            raise RegistryError(
                f"Invalid registry format in {self._path}: "
                "expected top-level 'runs' key"
            )
        self._entries = {}
        for item in raw["runs"]:
            entry = RunEntry.from_dict(item)
            self._entries[entry.run_id] = entry

    def save(self) -> None:
        """Atomic write: write to .tmp then os.replace (Rule 1 -- no silent data loss)."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self._path.with_suffix(".json.tmp")
        payload = {
            "runs": [e.to_dict() for e in self.all_entries()]
        }
        try:
            tmp_path.write_text(
                json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            os.replace(str(tmp_path), str(self._path))
        except OSError as exc:
            raise RegistryError(f"Failed to save registry: {exc}") from exc

    # -- id generation ------------------------------------------------------

    def next_id(self) -> str:
        """Return the next sequential run ID (e.g. RUN-001, RUN-002, ...)."""
        if not self._entries:
            return "RUN-001"
        max_num = max(
            int(rid.split("-")[1]) for rid in self._entries
        )
        return f"RUN-{max_num + 1:03d}"

    # -- CRUD ---------------------------------------------------------------

    def register(
        self,
        folder_path: str,
        timestamp: str,
        config_path: str = "",
        status: str = "training",
        description: str = "",
        metrics: Optional[Dict[str, float]] = None,
        mnar_variants: Optional[List[str]] = None,
        n_experts: int = 0,
        tags: Optional[List[str]] = None,
    ) -> RunEntry:
        """Create and persist a new run entry.  Returns the entry."""
        run_id = self.next_id()
        folder = Path(folder_path)
        entry = RunEntry(
            run_id=run_id,
            folder_name=folder.name,
            folder_path=str(folder),
            timestamp=timestamp,
            config_path=config_path,
            status=status,
            description=description,
            metrics=metrics or {},
            mnar_variants=mnar_variants or [],
            n_experts=n_experts,
            tags=tags or [],
        )
        self._entries[run_id] = entry
        self.save()
        return entry

    def update(self, run_id: str, **kwargs) -> RunEntry:
        """
        Update fields on an existing entry and persist.

        Raises RegistryError if run_id not found or invalid values supplied.
        """
        entry = self.get(run_id)
        for key, value in kwargs.items():
            if not hasattr(entry, key):
                raise RegistryError(
                    f"RunEntry has no attribute '{key}'"
                )
            setattr(entry, key, value)
        # Re-validate after mutation.
        entry.__post_init__()
        self._entries[run_id] = entry
        self.save()
        return entry

    def get(self, run_id: str) -> RunEntry:
        """Retrieve by run_id.  Raises RegistryError if missing (Rule 1)."""
        if run_id not in self._entries:
            raise RegistryError(f"Run '{run_id}' not found in registry")
        return self._entries[run_id]

    def find_by_folder(self, folder_name: str) -> Optional[RunEntry]:
        """Find an entry whose folder_name matches.  Returns None if absent."""
        for entry in self._entries.values():
            if entry.folder_name == folder_name:
                return entry
        return None

    def all_entries(self) -> List[RunEntry]:
        """Return all entries sorted by run_id."""
        return sorted(self._entries.values(), key=lambda e: e.run_id)

    def __len__(self) -> int:
        return len(self._entries)

    @property
    def path(self) -> Path:
        """Expose the registry file path (read-only)."""
        return self._path
