import logging

class Bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    ENDC = '\033[0m'

    # Method that returns a message with the desired color
    # usage:
    #    print(bcolor.colored("My colored message", bcolor.OKBLUE))
    @staticmethod
    def colored(message, color):
        return color + message + Bcolors.ENDC

    # Method that returns a yellow warning
    # usage:
    #   print(Bcolors.warning("What you are about to do is potentially dangerous. Continue?"))
    @staticmethod
    def warning(message):
        return Bcolors.WARNING + message + Bcolors.ENDC

    # Method that returns a red fail
    # usage:
    #   print(Bcolors.fail("What you did just failed massively. Bummer"))
    #   or:
    #   sys.exit(Bcolors.fail("Not a valid date"))
    @staticmethod
    def fail(message):
        return Bcolors.FAIL + message + Bcolors.ENDC

    # Method that returns a green ok
    # usage:
    #   print(Bcolors.ok("What you did just ok-ed massively. Yay!"))
    @staticmethod
    def ok(message):
        return Bcolors.OKGREEN + message + Bcolors.ENDC

    # Method that returns a blue ok
    # usage:
    #   print(Bcolors.okblue("What you did just ok-ed into the blue. Wow!"))
    @staticmethod
    def okblue(message):
        return Bcolors.OKBLUE + message + Bcolors.ENDC

    # Method that returns a header in some purple-ish color
    # usage:
    #   print(Bcolors.header("This is great"))
    @staticmethod
    def header(message):
        return Bcolors.HEADER + message + Bcolors.ENDC


class CustomFormatter(logging.Formatter):
    grey = "\x1b[38;21m"
    yellow = "\x1b[33;21m"
    red = "\x1b[31;21m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)"

    FORMATS = {
        logging.DEBUG: grey + format + reset,
        logging.INFO: grey + format + reset,
        logging.WARNING: yellow + format + reset,
        logging.ERROR: red + format + reset,
        logging.CRITICAL: bold_red + format + reset
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)