# Added/adapted for this course project, 2026-11-01.
# License: GPL-3.0; see 部署文档/许可与来源.txt at the project root.
# Project paths reorganized, 2026-11-03.
"""Prevent concurrent training processes from overwriting experiment files."""
import os

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
