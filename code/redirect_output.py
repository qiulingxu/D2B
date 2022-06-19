# from https://stackoverflow.com/questions/19425736/how-to-redirect-stdout-and-stderr-to-logger-in-python Cameron Gagnon
import logging
import sys

class LoggerWriter:
    def __init__(self, level, std_out):
        # self.level is really like using log.debug(message)
        # at least in my case
        self.level = level
        self.std_out = std_out
        self.cache = ""

    def _write(self, message):
        self.level(message)

    def write(self, message):
        # if statement reduces the amount of newlines that are
        # printed to the logger
        if message != '\n':
            self.cache += message
            if message.find("\n") >= 0:
                self._write(self.cache)
                self.cache = ""

        self.std_out.write(message)

    def flush(self):
        # create a flush method so things can be flushed when
        # the system wants to. Not sure if simply 'printing'
        # sys.stderr is the correct way to do it, but it seemed
        # to work properly for me.
        self.level(sys.stderr)
        

def overide_write(file_name, filemode = 'a'):

    if filemode == "w":
        with open(file_name, "w"):
            pass
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s:%(levelname)s:%(name)s:%(message)s',
        filename=file_name,
        filemode='a'
    )
    std_out = sys.stdout
    log = logging.getLogger()
    sys.stdout = LoggerWriter(log.debug, std_out)
    sys.stderr = LoggerWriter(log.warning, std_out)
