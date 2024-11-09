import argparse

import mathy.command
from mathy.synthetic.numbers import NaturalNumberGenerator

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
        generator = NaturalNumberGenerator(self.args.seed)
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
