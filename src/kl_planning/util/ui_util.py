from datetime import datetime


class colors:
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    ENDCOLOR = '\033[0m'

class accents:
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

    
def print_warning(msg, bold=False):
    if bold:
        print(f"{accents.BOLD}{colors.YELLOW}{msg}{colors.ENDCOLOR}")
    else:
        print(f"{colors.YELLOW}{msg}{colors.ENDCOLOR}")


def print_error(msg, bold=False):
    if bold:
        print(f"{accents.BOLD}{colors.RED}{msg}{colors.ENDCOLOR}")
    else:
        print(f"{colors.RED}{msg}{colors.ENDCOLOR}")


def print_happy(msg, bold=True):
    if bold:
        print(f"{accents.BOLD}{colors.GREEN}{msg}{colors.ENDCOLOR}")
    else:
        print(f"{colors.GREEN}{msg}{colors.ENDCOLOR}")


def print_query(msg, bold=True):
    if bold:
        print(f"{accents.BOLD}{colors.BLUE}{msg}{colors.ENDCOLOR}", end='')
    else:
        print(f"{colors.BLUE}{msg}{colors.ENDCOLOR}", end='')


def print_info(msg, bold=True):
    if bold:
        print(f"{accents.BOLD}{colors.CYAN}{msg}{colors.ENDCOLOR}", end='')
    else:
        print(f"{colors.CYAN}{msg}{colors.ENDCOLOR}", end='')
    

def query_yes_no(question, default="no"):
    """
    Ask a yes/no question via raw_input() and return their answer.

    Args:
        question (str): Question presented to user.
        default (str): Default answer if the user just hits <Enter>.
    Returns:
        answer (bool): User's response to question.

    Credit: http://code.activestate.com/recipes/577058/
    """
    valid = {"yes": True, "y": True, "ye": True, "no": False, "n": False}
    if default is None:
        prompt = " [y/n] "
    elif default == "yes":
        prompt = " [Y/n] "
    elif default == "no":
        prompt = " [y/N] "
    else:
        raise ValueError("invalid default answer: '%s'" % default)

    while True:
        print_query(question + prompt)
        choice = input().lower()
        if default is not None and choice == '':
            return valid[default]
        elif choice in valid:
            return valid[choice]
        else:
            print_warning("Please respond with 'yes' or 'no' (or 'y' or 'n').")


def get_time():
    return datetime.now().strftime("%Y%m%d-%H%M%S")
