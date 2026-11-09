"""
nanollm command-line entry point: 
    `nanollm <command> [options]`

Each command is a module under nanollm/ with a main(argv) function. 
Modules are imported only when their command is chosen, 
so `nanollm prepare_fineweb` does not import torch and 
`nanollm --help` imports nothing heavy.
"""

import importlib
import sys

COMMANDS = {
    "prepare_fineweb":              "nanollm.dataset.prepare_fineweb",
    "prepare_tinyshakespeare":      "nanollm.dataset.prepare_tinyshakespeare",
    "check_data":                   "nanollm.dataset.check_data",
    # later: "train": "nanollm.train", "generate": "nanollm.generate", ...
}


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if not argv or argv[0] in ("-h", "--help"):
        print("usage: nanollm <command> [options]\n\ncommands:")
        for name in COMMANDS:
            print(f"  {name}")
        print("\nRun `nanollm <command> --help` for a command's own options.")
        return
    command, rest = argv[0], argv[1:]
    if command not in COMMANDS:
        sys.exit(f"nanollm: unknown command '{command}'. Choose from: {', '.join(COMMANDS)}")
    importlib.import_module(COMMANDS[command]).main(rest)