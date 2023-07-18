"""Main module of AI-SOFIA project"""
import inspect
from collections.abc import Generator
import sys
import re
import pathlib as path
from termcolor import colored
import hotreload

cmds = hotreload.Loader("commands.py")

module = cmds.CommandModule()

parsers_table = {
    "int": int,
    "str": str,
}

#pylint: disable=R0903
class Command():
    """Represents a console command"""
    def __init__(self, method_name: str, except_types: list[str]):
        self.method_name = method_name
        self.except_types = except_types

    def __str__(self) -> str:
        return f"{self.method_name}{self.except_types}"
#pylint: enable=R0903

attributes = dir(module)
attributes = filter(lambda x: not x.startswith("__"), attributes) # remove private and special
attributes = filter(lambda x: callable(getattr(module, x)), attributes) # methods only

commands: dict[str, Command] = {}

for attr_name in attributes:
    method = getattr(module, attr_name)

    commandExceptTypes = [
        para.annotation.__name__ for para in inspect.signature(method).parameters.values()
    ]

    command = Command(attr_name, commandExceptTypes)
    commands[attr_name] = command

def execute_command(command_name: str, raw_args: list[str]):
    """Executes command by name and string arguments"""
    command_inst = commands[command_name]

    parsed_args = [
        parsers_table[except_type](arg)
        for except_type, arg in zip(command_inst.except_types, raw_args)
    ]

    result = getattr(module, command_inst.method_name)(*parsed_args)

    if result is not None:
        if isinstance(result, Generator):
            print("\n".join(list(map(str, result))))
        else: print(result)

def execute_script(script_path: str):
    """Executes script by path"""
    with open(script_path, 'r', encoding='utf-8') as script:
        commands_r = script.read()

    cwd = str(path.Path(script_path).parent.absolute())

    for command_r in commands_r.splitlines():
        if command_r == "":
            continue

        if command_r.startswith('@SCRIPT '):
            execute_script(cwd + '/' + command_r[len('@SCRIPT '):])
        elif command_r.startswith('@USERCHOICE'):
            split = re.split(r' ?@', command_r)[1:]

            match = re.match(r'USERCHOICE:([\w]+):([\w]+)', split[0])
            default_case = match.group(2)
            promt = match.group(1).replace('_', ' ')

            cases = {}
            for split_unit in split[1:]:
                match = re.match(r'CASE:([\w]+) (.+)', split_unit)
                if match is None:
                    match = re.match(r'ALLOWNONE:([\w]+)', split_unit)
                    cases[match.group(1)] = None
                else:
                    cases[match.group(1)] = match.group(2)

            while True:
                input_promt = f"{promt} [{'|'.join(cases.keys())} DEFAULT:{default_case}]: "
                input_promt = colored(input_promt, 'white', attrs = ['bold'])
                choose = input(input_promt)
                if choose == "":
                    choose = default_case
                elif choose not in cases:
                    print(colored(f"No case {choose}, try again", 'red'))
                    continue

                cmd = cases[choose]
                if cmd is not None:
                    splited_cmd = cmd.split(' ')
                    print(colored("SOFIA AI SCRIPT> ", 'yellow') + colored(cmd, 'white'))
                    execute_command(splited_cmd[0], splited_cmd[1:])
                break
        else:
            command_parts = command_r.split(' ')
            for command_part_i, command_part in enumerate(command_parts):
                match = re.match(r'@INPUT:([!\w]+)', command_part)
                if match is not None:
                    input_promt = match.group(1)
                    input_promt = colored(input_promt.replace('_', ' ') + ": ", 'white', attrs = ['bold'])
                    command_parts[command_part_i] = input(input_promt)

            print(colored("SOFIA AI SCRIPT> ", 'yellow') + colored(" ".join(command_parts), 'white'))
            execute_command(command_parts[0], command_parts[1:])




args = sys.argv[1:]

if len(args) != 0:
    if args == 1:
        print(colored("Invalid command line args", 'red'))
        sys.exit(1)

    if args[0] == '--command':
        execute_command(args[1], args[2:])
    elif args[0] == '--script':
        execute_script(" ".join(args[1:]))
else:
    while True:
        fullCommand = input(colored("SOFIA AI> ", "yellow"))
        if fullCommand == "exit":
            break

        if fullCommand == "help":
            print("\n".join(map(str, commands.values())))
        elif fullCommand in ["reimport", ".r"]:
            if not cmds.has_changed():
                print(colored("No changed detected in commands.py", 'red'))
            module = cmds.CommandModule()
            print(colored("Reimport ok", "green"))
        elif not fullCommand:
            continue
        else:
            try:
                splited = fullCommand.split()
                command = splited[0]
                cargs = splited[1:]
                execute_command(command, cargs)
            #pylint: disable=W0718
            except Exception as ex:
            #pylint: enable=W0718
                print(colored("Command finished with exception!", "red"))
                print(colored(ex, "red"))
            except KeyboardInterrupt:
                print(colored("CTRL + C. Keyboard interrupt", 'red'))
