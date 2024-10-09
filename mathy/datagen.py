import argparse
import random
import json

import mathy.command

class Number:
    value: float
    is_integer: bool
    spelled_out: str

    def __init__(self, value: int, spelled_out: str):
        self.value = value
        self.spelled_out = spelled_out

    def as_training_example(self) -> str:
        return json.dumps({
            'value': self.spelled_out, 
            'solution': self.value
        })

class NumberGenerator:
    def __init__(self, seed: int):
        self.random = random.Random(seed)

    def _get_spelled_out(self, value: int) -> str:
        # hard code all numbers less than 20
        if value < 20:
            return [
                "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
                "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"
            ][value]
        # hard code all tens
        if value < 100:
            return [
                "", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"
            ][value // 10] + (" " + self._get_spelled_out(value % 10) if value % 10 != 0 else "")
        # hard code all hundreds
        if value < 1000:
            return self._get_spelled_out(value // 100) + " hundred" + (" and " + self._get_spelled_out(value % 100) if value % 100 != 0 else "")
        # hard code all thousands
        if value < 1000000:
            return self._get_spelled_out(value // 1000) + " thousand" + (" " + self._get_spelled_out(value % 1000) if value % 1000 != 0 else "")
        # hard code all millions
        if value < 1000000000:
            return self._get_spelled_out(value // 1000000) + " million" + (" " + self._get_spelled_out(value % 1000000) if value % 1000000 != 0 else "")
        # hard code all billions
        if value < 1000000000000:
            return self._get_spelled_out(value // 1000000000) + " billion" + (" " + self._get_spelled_out(value % 1000000000) if value % 1000000000 != 0 else "")
        # hard code all trillions
        if value < 1000000000000000:
            return self._get_spelled_out(value // 1000000000000) + " trillion" + (" " + self._get_spelled_out(value % 1000000000000) if value % 1000000000000 != 0 else "")

        return "unknown"
    def generate(self) -> Number:
        # range distribution is skewed towards smaller numbers
        # 50% of the time, the number will be less than 1000
        # 90% of the time, the number will be less than 1000000
        # 99% of the time, the number will be less than 1000000000
        # 99.999999999% of the time, the number will be less than 1000000000000
        x = self.random.random()
        if x < 0.5:
            value = self.random.randint(0, 999)
        elif x < 0.9:
            value = self.random.randint(1000, 999999)
        elif x < 0.99:
            value = self.random.randint(1000000, 999999999)
        else:
            value = self.random.randint(1000000000, 1000000000000)
        spelled_out = self._get_spelled_out(value)
        return Number(value, spelled_out)


class DataGen(mathy.command.Command):
    """
    Generate data in the form of json objects that have a numeric expression as the
    "value" field and a "solution" field that is the result of evaluating the expression.
    The solution will be a integer or fixed point decimal number.
    """
    def __init__(self):
        super().__init__()

    def execute(self):
        seen = set()
        generator = NumberGenerator(self.args.seed)
        with open(self.args.output_file, "w") as f:
            for _ in range(self.args.example_count):
                while True:
                    number = generator.generate()
                    if number.spelled_out not in seen:
                        break
                
                seen.add(number.spelled_out)
                f.write(number.as_training_example() + "\n")

    @classmethod
    def add_args(cls, parser: argparse.ArgumentParser):
        parser.add_argument("--output_file", required=True, type=str, help="output file")
        parser.add_argument("--example_count", required=True, type=int, help="number of samples to generate")
        parser.add_argument("--seed", default=0, type=int, help="seed for the random number generator")
