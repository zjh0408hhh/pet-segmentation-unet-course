# Added/adapted for this course project, 2026-11-01.
# License: GPL-3.0; upstream attribution and modification details in SOURCES.md/CHANGES.md.
"""OS-specific locking and optional Windows sleep inhibition."""
import os

def keep_awake():
    if os.name == 'nt':
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(ctypes.c_uint(0x80000001))

def acquire_lock(path):
    stream = path.open('a+b')
    stream.seek(0, 2)
    if stream.tell() == 0:
        stream.write(b'0')
        stream.flush()
    stream.seek(0)
    try:
        if os.name == 'nt':
            import msvcrt
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as error:
        stream.close()
        raise RuntimeError('This experiment is already running.') from error
    return stream
