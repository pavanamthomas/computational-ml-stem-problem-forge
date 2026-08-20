"""Discover and import problem laboratories.

The registry is path-based. Adding a directory under ``problems/<domain>/<id>/``
with a valid ``problem.yaml`` is sufficient for listing, validation, and runs.
The scan is recursive so the same loader can hold 100+ problems without a
hand-maintained index.
"""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

import yaml

from problemforge.schema import ProblemSpec

REQUIRED_FILES = (
    "problem.yaml",
    "README.md",
    "reference_solution.py",
    "independent_verifier.py",
    "invariants.py",
    "failure_modes.md",
    "expected_schema.json",
)


def repository_root(start: Path | None = None) -> Path:
    """Locate the repository root that contains ``problems/``."""
    if start is None:
        start = Path(__file__).resolve()
        candidates = [start.parents[2], Path.cwd()]
    else:
        candidates = [start, *start.parents, Path.cwd()]
    for cand in candidates:
        if (cand / "problems").is_dir() and (cand / "src").is_dir():
            return cand
    cwd = Path.cwd()
    if (cwd / "problems").is_dir():
        return cwd
    raise FileNotFoundError(
        "Could not locate repository root (expected a problems/ directory)."
    )


def problems_root(start: Path | None = None) -> Path:
    return repository_root(start) / "problems"


@dataclass(frozen=True)
class ProblemRecord:
    """Index entry for one problem laboratory."""

    problem_id: str
    qualified_id: str
    path: Path
    domain: str
    difficulty: str
    title: str

    @property
    def keys(self) -> tuple[str, str]:
        return (self.problem_id, self.qualified_id)


@dataclass
class ProblemModules:
    spec: ProblemSpec
    record: ProblemRecord
    reference: ModuleType
    verifier: ModuleType
    invariants: ModuleType


class Registry:
    """In-memory index of every ``problem.yaml`` under ``problems/``."""

    def __init__(self, root: Path | None = None) -> None:
        self.root = problems_root(root) if root is None or root.name != "problems" else root
        if root is not None and (root / "problems").is_dir():
            self.root = root / "problems"
        self._by_key: dict[str, ProblemRecord] = {}
        self._scan()

    def _scan(self) -> None:
        if not self.root.is_dir():
            raise FileNotFoundError(f"problems directory missing: {self.root}")
        for yaml_path in sorted(self.root.rglob("problem.yaml")):
            problem_dir = yaml_path.parent
            domain = problem_dir.parent.name
            short_id = problem_dir.name
            qualified = f"{domain}/{short_id}"
            raw = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
            if not isinstance(raw, dict):
                raise ValueError(f"problem.yaml is not a mapping: {yaml_path}")
            title = str(raw.get("title", short_id))
            difficulty = str(raw.get("difficulty", "unknown"))
            yaml_id = str(raw.get("id", short_id))
            record = ProblemRecord(
                problem_id=yaml_id,
                qualified_id=qualified,
                path=problem_dir,
                domain=domain,
                difficulty=difficulty,
                title=title,
            )
            for key in (yaml_id, short_id, qualified):
                existing = self._by_key.get(key)
                if existing is not None and existing.path != record.path:
                    raise ValueError(
                        f"duplicate problem key {key!r}: {existing.path} and {record.path}"
                    )
                self._by_key[key] = record

    def get(self, problem_id: str) -> ProblemRecord:
        try:
            return self._by_key[problem_id]
        except KeyError as exc:
            known = ", ".join(sorted({r.qualified_id for r in self}))
            raise KeyError(
                f"unknown problem {problem_id!r}. Known: {known}"
            ) from exc

    def __iter__(self):
        seen: set[Path] = set()
        for record in sorted(self._by_key.values(), key=lambda r: r.qualified_id):
            if record.path in seen:
                continue
            seen.add(record.path)
            yield record

    def __len__(self) -> int:
        return len({r.path for r in self._by_key.values()})


def load_yaml_spec(problem_dir: Path) -> ProblemSpec:
    path = problem_dir / "problem.yaml"
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return ProblemSpec.model_validate(raw)


def import_module_from_path(path: Path, module_name: str) -> ModuleType:
    """Import a problem-local module without polluting ``sys.modules`` globally.

    Each problem has the same filenames (``reference_solution.py``). Importing
    them as top-level ``reference_solution`` would collide under pytest. The
    module name is therefore qualified by the problem directory.
    """
    if not path.is_file():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def load_problem_modules(
    problem_id: str, registry: Registry | None = None
) -> ProblemModules:
    registry = registry or Registry()
    record = registry.get(problem_id)
    spec = load_yaml_spec(record.path)
    stem = record.qualified_id.replace("/", ".")
    reference = import_module_from_path(
        record.path / "reference_solution.py",
        f"problemforge_problems.{stem}.reference_solution",
    )
    verifier = import_module_from_path(
        record.path / "independent_verifier.py",
        f"problemforge_problems.{stem}.independent_verifier",
    )
    invariants = import_module_from_path(
        record.path / "invariants.py",
        f"problemforge_problems.{stem}.invariants",
    )
    return ProblemModules(
        spec=spec,
        record=record,
        reference=reference,
        verifier=verifier,
        invariants=invariants,
    )


def required_files_present(problem_dir: Path) -> list[str]:
    missing = [name for name in REQUIRED_FILES if not (problem_dir / name).is_file()]
    tests_dir = problem_dir / "tests"
    if not tests_dir.is_dir() or not any(tests_dir.glob("test_*.py")):
        missing.append("tests/test_*.py")
    return missing
