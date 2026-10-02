"""Start the fixed, resource-limited pipeline independently of the chat session."""
import fcntl
from pathlib import Path
import subprocess
import sys
import time


def main():
    root = Path(__file__).resolve().parents[1]
    output = root/'artifacts/contracts/curve90'
    output.mkdir(parents=True, exist_ok=True)
    if (root/'data/curve90/STOP').exists():
        raise RuntimeError('STOP file present; pipeline not launched')
    with (output/'launch.lock').open('a') as launch_lock:
        fcntl.flock(launch_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with (output/'runner.lock').open('a') as probe:
            try:
                fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                print('Pipeline already running; no duplicate launched')
                return
        with (output/'pipeline.log').open('a') as log:
            log.write('\nStarting checkpointed pipeline via persistent launcher\n')
            log.flush()
            child = subprocess.Popen(
                [sys.executable, '-m', 'scripts.finish_curve90', '--max-jobs', '1200'],
                cwd=root, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                start_new_session=True)
        (output/'supervisor.pid').write_text(str(child.pid)+'\n')
        # Detect startup errors and hold the launcher lock until the child has
        # had time to acquire its own exclusive pipeline lock.
        time.sleep(1)
        if child.poll() is not None:
            raise RuntimeError(f'Pipeline exited at startup ({child.returncode}); see pipeline.log')
        print(f'Pipeline supervisor {child.pid}; log: {output / "pipeline.log"}')


if __name__ == '__main__':
    main()
