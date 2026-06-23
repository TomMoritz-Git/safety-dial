"""Dataset model and loaders for ladders, anchors, and the judge gold set.

The on-disk JSON schemas are documented on the loaders. Everything is validated
on load so a malformed dataset fails loudly and early rather than halfway
through a multi-hour run.
"""

import json
from dataclasses import dataclass
from pathlib import Path

from . import config


@dataclass(frozen=True)
class PromptItem:
    """A single graded request (one cell of one ladder)."""

    safeguard: str
    scenario_id: str
    surface: str
    level: int
    prompt: str

    @property
    def uid(self) -> str:
        """Stable unique id, e.g. ``privacy/privacy_03/L2``."""
        return f"{self.safeguard}/{self.scenario_id}/L{self.level}"


@dataclass(frozen=True)
class Scenario:
    """One topic escalated across ``config.N_LEVELS`` severity levels."""

    safeguard: str
    scenario_id: str
    surface: str
    levels: tuple[str, ...]

    def items(self) -> list[PromptItem]:
        """Expand into one :class:`PromptItem` per level."""
        return [
            PromptItem(self.safeguard, self.scenario_id, self.surface, lvl, text)
            for lvl, text in enumerate(self.levels)
        ]


@dataclass(frozen=True)
class Safeguard:
    """A safeguard domain: an escalation description plus its scenarios."""

    name: str
    escalation: str
    scenarios: tuple[Scenario, ...]


@dataclass(frozen=True)
class Anchors:
    """Topic-disjoint anchor requests defining the refusal direction."""

    benign: tuple[str, ...]
    harmful: tuple[str, ...]


@dataclass(frozen=True)
class GoldItem:
    """A hand-labeled (request, response) pair validating the judge."""

    id: str
    request: str
    response: str
    label: str
    note: str = ""


def load_ladders(path: Path | None = None) -> dict[str, Safeguard]:
    """Load and validate ``ladders.json``.

    Expected schema::

        {
          "<safeguard>": {
            "escalation": "<str>",
            "scenarios": [
              {"id": "<safeguard>_NN", "surface": "<str>",
               "levels": ["<L0>", ..., "<L4>"]},
              ...
            ]
          },
          ...
        }

    Args:
        path: Override path; defaults to ``config.LADDERS_PATH``.

    Returns:
        Mapping from safeguard name to :class:`Safeguard`.

    Raises:
        ValueError: On any structural problem (missing keys, wrong level count,
            unexpected safeguard names).
    """
    raw = json.loads(Path(path or config.LADDERS_PATH).read_text())
    if set(raw) != set(config.SAFEGUARDS):
        raise ValueError(
            f"ladders safeguards {sorted(raw)} != configured {sorted(config.SAFEGUARDS)}"
        )
    out: dict[str, Safeguard] = {}
    for name in config.SAFEGUARDS:
        block = raw[name]
        scenarios: list[Scenario] = []
        for sc in block["scenarios"]:
            levels = tuple(sc["levels"])
            if len(levels) != config.N_LEVELS:
                raise ValueError(
                    f"{sc['id']}: expected {config.N_LEVELS} levels, got {len(levels)}"
                )
            scenarios.append(
                Scenario(
                    safeguard=name,
                    scenario_id=sc["id"],
                    surface=sc.get("surface", ""),
                    levels=levels,
                )
            )
        out[name] = Safeguard(
            name=name, escalation=block.get("escalation", ""), scenarios=tuple(scenarios)
        )
    return out


def all_items(ladders: dict[str, Safeguard]) -> list[PromptItem]:
    """Flatten every scenario of every safeguard into prompt items."""
    return [item for sg in ladders.values() for sc in sg.scenarios for item in sc.items()]


def load_anchors(path: Path | None = None) -> Anchors:
    """Load and validate ``anchors.json`` (keys ``benign`` and ``harmful``)."""
    raw = json.loads(Path(path or config.ANCHORS_PATH).read_text())
    benign, harmful = tuple(raw["benign"]), tuple(raw["harmful"])
    if not benign or not harmful:
        raise ValueError("anchors must contain non-empty benign and harmful lists")
    if len(benign) != len(harmful):
        raise ValueError(
            f"expected balanced anchors, got {len(benign)} benign / {len(harmful)} harmful"
        )
    return Anchors(benign=benign, harmful=harmful)


def load_gold(path: Path | None = None) -> list[GoldItem]:
    """Load and validate ``gold.json`` (list of labeled response pairs)."""
    raw = json.loads(Path(path or config.GOLD_PATH).read_text())
    items: list[GoldItem] = []
    for row in raw:
        if row["label"] not in config.JUDGE_LABELS:
            raise ValueError(f"{row['id']}: bad label {row['label']!r}")
        items.append(
            GoldItem(
                id=row["id"],
                request=row["request"],
                response=row["response"],
                label=row["label"],
                note=row.get("note", ""),
            )
        )
    return items
