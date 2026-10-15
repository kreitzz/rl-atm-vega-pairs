"""Bound one numerical worker to 50% of one core and 384 MiB aggregate RSS.

Uses a process-group duty cycle, not an OS CPU quota. On macOS also stops on
critical memory pressure, low headroom, or more than 64 MiB additional swap.
No browser or unrelated process is touched.
"""
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time


def swap_usage():
    if sys.platform != "darwin":
        return 0.
    output = subprocess.check_output(["sysctl", "vm.swapusage"], text=True)
    match = re.search(r"used\s*=\s*([\d.]+)([MG])", output)
    if match is None:
        raise RuntimeError("Cannot read system swap usage")
    return float(match[1]) * (1024 if match[2] == "G" else 1)


def memory_check(pid, initial_swap):
    output = subprocess.check_output(["ps", "-axo", "pid=,ppid=,rss="], text=True)
    rows = [tuple(map(int, line.split())) for line in output.splitlines() if line.strip()]
    descendants = {pid}
    while True:
        expanded = descendants | {p for p, parent, _ in rows if parent in descendants}
        if expanded == descendants:
            break
        descendants = expanded
    rss = sum(size for p, _, size in rows if p in descendants) / 1024
    if rss > 384:
        raise MemoryError(f"Worker RSS {rss:.1f} MiB exceeds 384 MiB")
    duty = .5
    if sys.platform == "darwin":
        output = subprocess.check_output(["/usr/bin/memory_pressure", "-Q"], text=True)
        match = re.search(r"System-wide memory free percentage:\s*(\d+)%", output)
        level = subprocess.check_output(
            ["sysctl", "-n", "kern.memorystatus_vm_pressure_level"], text=True).strip()
        if match is None or int(match[1]) < 35 or level not in ("1", "2"):
            raise MemoryError("System memory headroom/pressure stop")
        if swap_usage() - initial_swap > 64:
            raise MemoryError("System swap grew by more than 64 MiB")
        if level == "2":
            duty = .2
    return rss, duty


def run(arguments, root, timeout=1800):
    """Launch the package's worker and monitor only its process group."""
    initial_swap = swap_usage()
    env = dict(os.environ, PAPER_CODE_WORKER="1")
    env["PAPER_CODE_PARENT"] = str(os.getpid())
    bootstrap = (
        "import os,runpy,threading,time; os.nice(15)\n"
        "def watch():\n"
        " while True:\n"
        "  if os.getppid()!=int(os.environ['PAPER_CODE_PARENT']): os._exit(143)\n"
        "  time.sleep(.2)\n"
        "threading.Thread(target=watch,daemon=True).start()\n"
        "runpy.run_module('paper_code',run_name='__main__')\n"
    )
    child = subprocess.Popen([sys.executable, "-c", bootstrap, *arguments],
                             cwd=root, env=env, start_new_session=True)
    started, checked, peak, duty = time.monotonic(), 0., 0., .5

    def send(sig):
        try:
            os.killpg(child.pid, sig)
        except ProcessLookupError:
            child.wait()
        except PermissionError as error:
            # macOS can report EPERM during process-group teardown. Suppress
            # only if waitpid confirms the worker has exited.
            try:
                child.wait(timeout=1)
            except subprocess.TimeoutExpired:
                raise error

    try:
        while child.poll() is None:
            now = time.monotonic()
            if now - started > timeout or (Path(root) / "artifacts/paper_code/STOP").exists():
                raise InterruptedError("Timeout or artifacts/paper_code/STOP")
            if now - checked >= 2:
                rss, duty = memory_check(child.pid, initial_swap)
                peak, checked = max(peak, rss), now
            send(signal.SIGCONT)
            time.sleep(.2 * duty)
            if child.poll() is not None:
                break
            send(signal.SIGSTOP)
            time.sleep(.2 * (1 - duty))
        code = child.wait()
        print(f"Resource guard: peak worker RSS {peak:.1f} MiB", flush=True)
        return code
    finally:
        if child.poll() is None:
            send(signal.SIGCONT)
            send(signal.SIGTERM)
            try:
                child.wait(timeout=3)
            except subprocess.TimeoutExpired:
                send(signal.SIGKILL)
                child.wait()
