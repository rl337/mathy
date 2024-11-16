import argparse

# create a decorator that will skip a Command class if set on that class
def not_a_command(cls):
    cls.not_a_command = True
    return cls

class Command:
    args: argparse.Namespace

    @classmethod
    def add_args(cls, parser: argparse.ArgumentParser) -> None:
        pass

    def initialize(self, args: argparse.Namespace) -> None:
        self.args = args

    def execute(self) -> None:
        self.action()   