import argparse

class Command:
    args: argparse.Namespace

    @classmethod
    def add_args(cls, parser: argparse.ArgumentParser):
        pass

    def initialize(self, args: argparse.Namespace):
        self.args = args

    def execute(self):
        self.action()   