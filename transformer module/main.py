import inspect
from collections.abc import *
import sys
from termcolor import colored
import hotreload

cmds = hotreload.Loader("commands.py")

module = cmds.CommandModule()

parsers_table = {
    "int": int,
    "str": str,
}

class Command():
    def __init__(self, method_name, exceptTypes:list):
        self.method_name = method_name
        self.exceptTypes = exceptTypes

    def __str__(self) -> str:
        return f"{self.method_name}{self.exceptTypes}"

attributes = dir(module)
attributes = filter(lambda x: not x.startswith("__"), attributes) # remove private and special
attributes = filter(lambda x: callable(getattr(module, x)), attributes) # methods only

commands = {}

for method_name in attributes:
    method = getattr(module, method_name)
    commandExceptTypes = [para.annotation.__name__ for para in inspect.signature(method).parameters.values()]
    command = Command(method_name, commandExceptTypes)
    commands[method_name] = command

def execute_command(command_name:str, args:list):
    command = commands[command_name]

    parsed_args = [parsers_table[exceptType](arg) for exceptType, arg in zip(command.exceptTypes, args)]

    result = getattr(module, command.method_name)(*parsed_args)

    if result is not None:
        if isinstance(result, Generator):
            print("\n".join(list(map(str, result))))
        else: print(result)


args = sys.argv[1:]

if len(args) != 0:
    execute_command(args[0], args[1:])
else:
    while True:
        fullCommand = input(colored("SOFIA AI> ", "yellow"))
        if fullCommand == "exit":
            break
        elif fullCommand == "help":
            print("\n".join(map(str, commands.values())))
        elif fullCommand == "reimport" or fullCommand == ".r":
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
            except Exception as ex:
                print(colored("Command finished with exception!", "red"))
                print(colored(ex, "red"))
            except KeyboardInterrupt:
                print(colored("CTRL + C. Keyboard interrupt", 'red'))
