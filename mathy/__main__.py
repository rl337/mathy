import sys
import mathy
import logging
import argparse
import types
import importlib
import pkgutil
from inspect import getmembers, isclass
from typing import List, Type
import mathy.command


class SubcommandHelpFormatter(argparse.RawDescriptionHelpFormatter):
    def _format_action(self, action):
        parts = super()._format_action(action)
        if action.nargs == argparse.PARSER:
            parts = "\n".join([line for line in parts.split("\n")[1:]])
        return parts


def find_subclasses_in_module(m: types.ModuleType, base_class: Type, visited=None) -> List[Type[mathy.command.Command]]:
    if visited is None:
        visited = set()

    subclasses = []

    # Avoid processing the same module twice
    if m in visited:
        return subclasses
    visited.add(m)

    logging.debug(f"Scanning module: {m.__name__}")

    # Iterate over module members
    for name, obj in getmembers(m):
        if name.startswith("_"):
            continue

        # Check if obj is a class and is a subclass of base_class
        if isclass(obj) and issubclass(obj, base_class) and obj is not base_class:
            logging.debug(f"Found subclass: {obj.__name__}")
            subclasses.append(obj)

    # Check if the module is actually a package
    if hasattr(m, '__path__'):
        # Explicitly check for and import submodules
        for _, name, is_pkg in pkgutil.iter_modules(m.__path__):
            full_name = f"{m.__name__}.{name}"
            if full_name not in sys.modules:
                logging.debug(f"Importing module: {full_name}")
                importlib.import_module(full_name)
            
            submodule = sys.modules[full_name]
            logging.debug(f"Recursing into submodule: {full_name}")
            subclasses.extend(find_subclasses_in_module(submodule, base_class, visited))

    return subclasses


def main():

    parent_argparser = argparse.ArgumentParser(add_help=False)
    parent_argparser.add_argument("--loglevel", type=str, choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], default="INFO", help="output loglevel")
    loglevel_args, remaining_args = parent_argparser.parse_known_args()

    root_logger = logging.getLogger()
    root_logger.setLevel(loglevel_args.loglevel)
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    root_logger.handlers = [handler]

    command_classes = find_subclasses_in_module(mathy, mathy.command.Command)
    command_dict = {cls.__name__: cls for cls in command_classes if "not_a_command" not in cls.__dict__}

    loglevel_parser = argparse.ArgumentParser(add_help=False)
    loglevel_parser.add_argument("--loglevel", type=str, choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], default="INFO", help="output loglevel")
    loglevel_args, remaining_args = loglevel_parser.parse_known_args()

    # Set up logging based on the parsed loglevel
    logging.basicConfig(level=getattr(logging, loglevel_args.loglevel),
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        stream=sys.stderr)

    # Create the main argument parser with custom formatter
    argparser = argparse.ArgumentParser(
        parents=[loglevel_parser],
        formatter_class=SubcommandHelpFormatter,
        description="Mathy CLI\n\nAvailable subcommands:",
        epilog="For more information about a command, run: mathy <command> --help"
    )
    subparsers = argparser.add_subparsers(dest="command", help="The command to execute")

    # Add subcommands with their descriptions
    for command_name, command_class in command_dict.items():
        command_parser = subparsers.add_parser(command_name, help=command_class.__doc__)
        command_class.add_args(command_parser)
        # Add the command description to the main help
        argparser.description += f"\n  {command_name:<15} {command_class.__doc__ or ''}"

    args = argparser.parse_args(remaining_args)

    command = args.command

    if command not in command_dict:
        argparser.print_help()
        return

    # Log the chosen command
    logging.info(f"Executing command: {command}")

    # Execute the command
    command_class = command_dict[command]
    command_instance = command_class()
    command_instance.initialize(args)
    command_instance.execute()

if __name__ == "__main__":
    main()