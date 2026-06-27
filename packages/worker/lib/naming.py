"""
Human-readable name generator for graph entities (wandb-style).

Generates memorable two-word names like "rxn-bright-copper" for reactions,
"min-amber-prism" for minima, "ts-swift-ketone" for transition states.
"""

import random

ADJECTIVES = [
    "amber", "azure", "bold", "bright", "calm",
    "clear", "cool", "crisp", "dark", "deep",
    "dry", "dull", "fast", "faint", "fierce",
    "firm", "flat", "fresh", "glad", "gold",
    "grand", "green", "harsh", "keen", "light",
    "mild", "neat", "pale", "plain", "prime",
    "pure", "quick", "rare", "rich", "rough",
    "sharp", "sleek", "slim", "soft", "stark",
    "steep", "still", "stout", "swift", "tart",
    "thin", "tiny", "true", "vast", "vivid",
    "warm", "weak", "wet", "wide", "wild",
]

NOUNS = [
    "argon", "beryl", "boron", "brass", "cedar",
    "chalk", "chrome", "clay", "cobalt", "coral",
    "copper", "crystal", "delta", "ether", "ferrite",
    "field", "flame", "flint", "forge", "frost",
    "glass", "grain", "helix", "iron", "jade",
    "ketal", "ketone", "larch", "lattice", "lime",
    "lunar", "maple", "marsh", "mica", "neon",
    "nickel", "node", "oak", "onyx", "oxide",
    "pearl", "phase", "pine", "plasma", "prism",
    "pulse", "quartz", "ridge", "salt", "sigma",
    "slate", "spark", "spire", "steel", "stone",
    "tide", "torch", "vapor", "zinc", "zonal",
]


class NameGenerator:
    """Generates unique human-readable names with a given prefix.

    Names follow the pattern: "{prefix}-{adjective}-{noun}"
    e.g. "rxn-bright-copper"

    A random offset shifts the starting point in the word lists so that
    different NameGenerator instances (e.g. per PESGraph) produce
    different name sequences.

    Counter increments per call, ensuring uniqueness within one generator.
    """

    def __init__(self, adjectives: list[str] = None, nouns: list[str] = None):
        self._adjectives = adjectives or ADJECTIVES
        self._nouns = nouns or NOUNS
        self._counter: int = 0
        self._offset: int = random.randint(0, len(self._adjectives) * len(self._nouns) - 1)

    def generate(self, prefix: str) -> str:
        """Generate the next unique name with the given prefix."""
        idx = self._counter + self._offset
        n_adj = len(self._adjectives)
        n_noun = len(self._nouns)

        adj = self._adjectives[idx % n_adj]
        noun = self._nouns[(idx // n_adj) % n_noun]

        name = f"{prefix}-{adj}-{noun}"
        self._counter += 1
        return name
