import tempfile
import os
import shutil
import functools

class TempfileManager(object):
    """
    Ensure temporary files/paths get removed, but
    only at the very end of the calculation.
    This mitigates some issues with multiprocessing.
    """
    def __init__(self, oldworkdir):
        self.temptrees = []
        self.preserve_on_error = []
        self.oldworkdir = oldworkdir

    def create_tree(self, fdir, prefix="_sympy_compile"):
        if fdir is None:
            fdir = tempfile.mkdtemp(prefix)
            #print('created', fdir)
            self.temptrees.append(fdir)
        else:
            if not os.access(fdir, os.F_OK):
                os.mkdir(fdir)
        return fdir

    def create_logfile(self, prefix="", suffix=".log"):
        fd, fname = tempfile.mkstemp(suffix=suffix, prefix=prefix)
        self.preserve_on_error.append(fname)
        return fd

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        os.chdir(self.oldworkdir)
        for tree in self.temptrees:
            try:
                # Could be some issues on Windows
                #print('removing', tree)
                shutil.rmtree(tree)
            except OSError:
                pass
        if exc_tb is None:
            # Remove the log files (if they exist) if no exceptions were raised
            for fpath in self.preserve_on_error:
                try:
                    os.unlink(fpath)
                except OSError:
                    pass
        else:
            print('Preserved Files: ', self.preserve_on_error)


    def __call__(self, func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            #print('calling', func.__name__)
            tmpman = kwargs.pop('tmpman', None)
            # Use passed-in tmpman if available, otherwise use self
            if tmpman is None:
                with self:
                    kwargs['tmpman'] = self
                    return func(*args, **kwargs)
            else:
                kwargs['tmpman'] = tmpman
                return func(*args, **kwargs)
        return wrapper