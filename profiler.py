import cProfile as Profile
import pstats
import os
import sys


def profile(function, *args, **kwargs):
    """ Returns performance statistics (as a string) for the given function.
        """
    def _run():
        function(*args, **kwargs)

    sys.modules['__main__'].__profile_run__ = _run
    _id = function.__name__ + '()'
    Profile.run('__profile_run__()', _id)
    p = pstats.Stats(_id)
    p.stream = open(_id, 'w')
    p.sort_stats('time').print_stats(20)
    p.stream.close()
    s = open(_id).read()
    os.remove(_id)
    return s
