"""The tdb module provides support for reading and writing databases in
Thermo-Calc TDB format.
"""

from itertools import tee

def partition(pred, iterable):
    'Use a predicate to partition entries into false entries and true entries'
    # partition(is_odd, range(10)) --> 0 2 4 6 8   and  1 3 5 7 9
    tee1, tee2 = tee(iterable)
    return [i for i in tee1 if not pred(i)], [i for i in tee2 if pred(i)]

def tdbread(targetdb, lines):
    """
    Parse a TDB file into a pycalphad Database object.

    Parameters
    ----------
    targetdb : Database
        A pycalphad Database.
    lines : string
        A raw TDB file.
    """
    lines = lines.replace('\t', ' ')
    lines = lines.strip()
    # Split the string by newlines
    splitlines = lines.split('\n')
    # Remove extra whitespace inside line
    splitlines = [' '.join(k.split()) for k in splitlines]
    # Remove comments
    splitlines = [k for k in splitlines if not k.startswith("$")]
    # Combine everything back together
    lines = ' '.join(splitlines)
    # Now split by the command delimeter
    commands = lines.split('!')
    # Filter out comments one more time
    # It's possible they were at the end of a command
    commands = [k for k in commands if not k.startswith("$")]
    # Separate out all PARAMETER commands; to be handled last
    commands, para_commands = partition(
        lambda cmd: cmd.upper().startswith("PARA"),
        commands)
    # Separate out all FUNCTION commands
    commands, func_commands = partition(
        lambda cmd: cmd.upper().startswith("FUNC"),
        commands)

if __name__ == "__main__":
    pass