"""Backend interface: prompts in, answer-letter logits out."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, Sequence

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


class PromptTooLongError(ValueError):
    """The prompt without its truncatable state already exceeds the token budget."""


@dataclass(frozen=True)
class Prompt:
    """One instruction, split so a backend knows what it may truncate.

    The full instruction text is `prefix + state + suffix`.
    """

    prefix: str  # few-shot demos etc.; never truncated ("" if none)
    state: str  # the state text; the only part that may be truncated (from the left)
    suffix: str  # question block ending in "Answer:"; never truncated
    n_labels: int  # read logits over the first n_labels letters A, B, C, ...

    @property
    def text(self) -> str:
        return self.prefix + self.state + self.suffix


@dataclass
class Readout:
    """Next-token logits over the answer letters for one prompt."""

    logits: list[float]  # final-cycle raw logits over the n_labels letter tokens
    prompt_tokens: int
    truncated: bool = False
    # one row per H cycle, last row == logits; None for non-recurrent backends
    cycle_logits: list[list[float]] | None = None


class LabelBackend(Protocol):
    """A model that scores answer letters in one forward pass per prompt."""

    name: str
    max_labels: int

    def info(self) -> dict[str, Any]: ...

    def read(self, prompts: Sequence[Prompt]) -> list[Readout]:
        """Return one `Readout` per prompt, in input order."""
        ...
