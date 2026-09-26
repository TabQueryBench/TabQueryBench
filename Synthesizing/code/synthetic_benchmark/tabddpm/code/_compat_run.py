import collections, collections.abc
for _a in ('Sequence','MutableSequence','MutableMapping','Mapping','MutableSet','Set','Callable','Iterable','Iterator'):
    if not hasattr(collections, _a): setattr(collections, _a, getattr(collections.abc, _a, None))
import sys, runpy
sys.argv = sys.argv[1:]
runpy.run_path(sys.argv[0], run_name='__main__')
