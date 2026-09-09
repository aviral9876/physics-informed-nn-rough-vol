r"""
Hold the system awake for the duration of a long compute run.

Measured need: on this machine roughly 6 of 11 overnight hours were spent in
Modern Standby (Windows Kernel-Power events 506/507), which suspends -- but does
not kill -- running jobs. Two full-budget calibrations and the seed sweep each
accumulated a ~5-20% CPU duty cycle overnight as a result.

This asserts ES_SYSTEM_REQUIRED | ES_CONTINUOUS, which is the documented way for
a long-running process to say "do not idle-sleep the system". The assertion is
scoped to THIS process: it is released the moment the process exits or is
killed, and it changes no persistent power setting. The display is deliberately
left free to sleep (no ES_DISPLAY_REQUIRED).

Usage:
    python keepawake.py            # hold until killed
    python keepawake.py 43200      # hold for at most 12 h, then release
"""
import ctypes
import sys
import time

ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001


def acquire():
    rc = ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
    if rc == 0:
        raise OSError("SetThreadExecutionState failed")
    return rc


def release():
    ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS)


if __name__ == "__main__":
    limit = float(sys.argv[1]) if len(sys.argv) > 1 else float("inf")
    acquire()
    t0 = time.time()
    print("keep-awake asserted (system only; display may still sleep). "
          "Released when this process exits.", flush=True)
    try:
        while time.time() - t0 < limit:
            time.sleep(60)
            # Re-assert periodically: some power-policy transitions clear it.
            acquire()
            if int(time.time() - t0) % 1800 < 60:
                print("  still awake, %.1f h" % ((time.time() - t0) / 3600), flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        release()
        print("keep-awake released after %.1f h" % ((time.time() - t0) / 3600), flush=True)
