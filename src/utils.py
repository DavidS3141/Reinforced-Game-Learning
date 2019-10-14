#!/usr/bin/env python3

import time


def get_time_stamp(with_date=True, with_delims=False):
    if with_date:
        if with_delims:
            return time.strftime("%Y/%m/%d-%H:%M:%S")
        else:
            return time.strftime("%Y%m%d-%H%M%S")
    else:
        if with_delims:
            return time.strftime("%H:%M:%S")
        else:
            return time.strftime("%H%M%S")


def ask_yn(question, default=-1, timeout=0):
    """Ask interactively a yes/no-question and wait for an answer.
    Parameters
    ----------
    question : string
        Question asked to the user printed in the terminal.
    default : int
        Default answer can be one of (-1, 0, 1) corresponding to no default
        (requires an user response), No, Yes.
    timeout : float
        Timeout (in seconds) after which the default answer is returned. This
        raises an error if there is no default provided (default = -1).
    Returns
    -------
    bool
        Answer to the question trough user or default. (Yes=True, No=False)
    """
    import sys
    import select

    answers = "[y/n]"
    if default == 0:
        answers = "[N/y]"
    elif default == 1:
        answers = "[Y/n]"
    elif default != -1:
        raise Exception("Wrong default parameter (%d) to ask_yn!" % default)

    if timeout > 0:
        if default == -1:
            raise Exception("When using timeout, specify a default answer!")
        answers += " (%.1fs time to answer!)" % timeout
    print(question + " " + answers)

    if timeout == 0:
        ans = input()
    else:
        i, o, e = select.select([sys.stdin], [], [], timeout)
        if i:
            ans = sys.stdin.readline().strip()
        else:
            ans = ""

    if ans == "y" or ans == "Y":
        return True
    elif ans == "n" or ans == "N":
        return False
    elif len(ans) == 0:
        if default == 0:
            return False
        elif default == 1:
            return True
        elif default == -1:
            print("There is no default option given to this y/n-question!")
            return ask_yn(question, default=default, timeout=timeout)
        else:
            raise Exception("Logical error in ask_yn function!")
    else:
        print("Wrong answer to y/n-question! Answer was %s!" % ans)
        return ask_yn(question, default=default, timeout=timeout)
    raise Exception("Logical error in ask_yn function!")
