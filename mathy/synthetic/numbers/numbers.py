import json
from typing import List

import babel
import babel.numbers
import num2words

from mathy.synthetic.generator import DataGenerator


class Number:
    value: float
    rendered: str

    def __init__(self, value: int, rendered: str):
        self.value = value
        self.rendered = rendered

    def as_training_example(self) -> str:
        return json.dumps({
            'value': self.rendered,
            'solution': self.value
        })
    
class NumberRenderer:
    def render(self, value: int) -> str:
        raise NotImplementedError()
    
class RawNumberRenderer(NumberRenderer):
    def render(self, value: int) -> str:
        return str(value)
    
class FormattedNumberRenderer(NumberRenderer):
    def __init__(self, locale: babel.Locale):
        self.locale = locale

    def render(self, value: int) -> str:
        return babel.numbers.format_number(value, locale=self.locale)
    
class SpelledOutNumberRenderer(NumberRenderer):
    def __init__(self, locale: babel.Locale):
        self.locale = locale

    def render(self, value: int) -> str:
        return num2words.num2words(value, lang=self.locale.language)


class NaturalNumberGenerator(DataGenerator):
    renderers: List[NumberRenderer]
    def __init__(self, seed: int = 0, locale: str = "en_US"):
        super().__init__(locale, seed)
        self.renderers = [
            RawNumberRenderer(),
            FormattedNumberRenderer(self.locale),
            SpelledOutNumberRenderer(self.locale)
        ]
    
    def generate(self) -> Number:
        # range distribution is skewed towards smaller numbers
        # 50% of the time, the number will be less than 1000
        # 90% of the time, the number will be less than 1000000
        # 99% of the time, the number will be less than 1000000000
        # 99.999999999% of the time, the number will be less than 1000000000000
        x = self.random.next_float()
        if x < 0.5:
            value = self.random.next_int(0, 999)
        elif x < 0.9:
            value = self.random.next_int(1000, 999999)
        elif x < 0.99:
            value = self.random.next_int(1000000, 999999999)
        else:
            value = self.random.next_int(1000000, 1000000000000)

        renderer = self.random.next_choice(self.renderers)
        rendered = renderer.render(value)

        return Number(value, rendered)
    
    def generate_batch(self, count: int) -> List[Number]:
        # generate a batch of count numbers avoiding duplicates
        seen = set()
        numbers = []
        while len(numbers) < count:
            number = self.generate()
            if number.rendered not in seen:
                seen.add(number.rendered)
                numbers.append(number)
        return numbers
