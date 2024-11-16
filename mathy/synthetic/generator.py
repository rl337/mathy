
import random
import sys
from typing import List, TypeVar

import babel

T = TypeVar('T')

class SeededRandom:
    seed: int
    _uses: int
    _random: random.Random

    def __init__(self, seed: int):
        self.seed = seed
        self._random = random.Random(seed)
        self._uses = 0

    def next_int(self, min: int = -sys.maxsize - 1, max: int = sys.maxsize - 1) -> int:
        self._uses += 1
        return self._random.randint(min, max)

    def next_float(self) -> float:
        self._uses += 1
        return self._random.random()
    
    def next_choice(self, choices: List[T]) -> T:
        self._uses += 1
        return self._random.choice(choices)

    def clone(self):
        # start with our seed and use the same number of uses
        result = SeededRandom(self.seed)
        for _ in range(self._uses):
            result.next_float()
        return result


class DataGenerator:
    random: SeededRandom
    locale: babel.Locale

    def __init__(self, locale: str = "en_US", seed: int = 0):
        self.random = SeededRandom(seed)
        self.locale = babel.Locale(locale)
