import os, re
import tkinter
import tkinter.filedialog as filedialog
from operator import itemgetter

class FileManager:
    def __init__(self):
        pass

    def Select(self, initialdir=None, extspec='*'):
        # Open file selection dialouge and let user select files, store file properties in object
        self.initialdir = initialdir # start up directory

        self.window = tkinter.Tk() # initialize tkinter
        self.window.withdraw() # close unecessary window

        # Open file selection GUI
        self.files = filedialog.askopenfilenames(initialdir=self.initialdir, title="Select file", filetypes=(
            ("all files", "*.*"),
            ("mat files", '*.mat'),
            ("avi files", '*.avi'),
            ("mp4 files", '*.mp4'),
            ("mov files", '*.mov')))

        self.files = list(self.files) # convert to list
        self.files.sort(key=self.natural_keys) # sort files
        self.nfile = len(self.files)

        pathfile        = list(map(os.path.split, self.files))
        self.path       = list(map(itemgetter(0), pathfile)) # paths to files
        self.basename   = list(map(itemgetter(1), pathfile)) # filenames with extensions
        filext          = list(map(os.path.splitext, self.basename))
        self.fname      = list(map(itemgetter(0), filext)) # filenames without extensions
        self.ext        = list(map(itemgetter(1), filext)) # file extensions

    def TargetDir(self, targetdir, ext=[None]):
        # Create target paths
        self.targetdir  = [targetdir]*self.nfile # target directory
        self.targetext  = [ext] * self.nfile # target extension
        self.targetfile = [i + j for i, j in zip(self.targetdir, self.fname)] # target full file path

        if ext[0] is not None:
            self.targetfile = [i + j for i, j in zip(self.targetfile, self.targetext)] # add extension to full file path if specified

    def atof(self,text):
        try:
            retval = float(text)
        except ValueError:
            retval = text
        return retval

    def natural_keys(self,text):
        return [ self.atof(c) for c in re.split(r'[+-]?([0-9]+(?:[.][0-9]*)?|[.][0-9]+)', text) ]
