"""Helpers to run external estimation/recording commands used by the burger scripts.

Provides `execute_watching` and `execute_smoking` which invoke an external
`lerobot-record` CLI for a fixed duration and then terminate it cleanly.

These functions are intentionally small wrappers around subprocess so the
caller (the state machine in `main.py`) can run blocking actions without
duplicating process-management logic.
"""

from typing import Sequence
import subprocess
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _run_command_for_seconds(cmd: Sequence[str], seconds: int) -> int:
    """Run `cmd` as a subprocess for `seconds`, then terminate it.

    The subprocess is started and allowed to run for `seconds` seconds. After
    that sleep the subprocess is terminated (SIGTERM) and, if it doesn't exit
    within a short timeout, it is killed (SIGKILL).

    Returns the process return code (may be None until process terminates).
    """
    logger.info("Running command for %d seconds: %s", seconds, " ".join(cmd))
    # Redirect output to avoid filling pipes in long-running processes.
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    try:
        time.sleep(seconds)
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt received; terminating child process")
    finally:
        # Try graceful termination first
        try:
            proc.terminate()
        except Exception:
            pass

        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            logger.info("Process did not exit after SIGTERM; killing it")
            try:
                proc.kill()
            except Exception:
                pass
            proc.wait()

    logger.info("Process finished with return code: %s", proc.returncode)
    return proc.returncode


def execute_working(duration: int = 20) -> None:
    """Execute the watching CLI for `duration` seconds.

    This runs the `lerobot-record` command with parameters used by the burger
    robot policy and waits `duration` seconds before terminating the process.
    """
    cmd = [
        "lerobot-record",
        "--robot.type=bi_so100_follower",
        "--robot.left_arm_port=/dev/ttyACM2",
        "--robot.right_arm_port=/dev/ttyACM0",
        "--robot.id=bimanual_follower",
        "--policy.path=Mozgi512/act_burger_merged2_6000",
        "--robot.cameras={ top: {type: opencv, index_or_path: 6, width: 640, height: 480, fps: 30},front: {type: opencv, index_or_path: 8, width: 640, height: 480, fps: 30}}",
        "--display_data=false",
        "--dataset.push_to_hub=false",
        "--dataset.single_task=Burger",
        "--dataset.episode_time_s=15",
        "--dataset.num_episodes=1",
        "--dataset.repo_id=Mozgi512/eval_hoge1",
    ]

    logger.info("Starting watching action (duration=%ds)", duration)
    _run_command_for_seconds(cmd, duration)
    logger.info("Watching action completed")


def execute_smoking(duration: int = 20) -> None:
    """Execute the smoking action.

    Currently this function uses the same command/structure as
    `execute_watching` as a placeholder. Replace the command contents here
    when the actual smoking CLI is available.
    """
    cmd = [
        "lerobot-record",
        "--robot.type=bi_so100_follower",
        "--robot.left_arm_port=/dev/ttyACM2",
        "--robot.right_arm_port=/dev/ttyACM0",
        "--robot.id=bimanual_follower",
        "--policy.path=Mozgi512/act_burger_merged2_6000",
        "--robot.cameras={ top: {type: opencv, index_or_path: 6, width: 640, height: 480, fps: 30},front: {type: opencv, index_or_path: 8, width: 640, height: 480, fps: 30}}",
        "--display_data=false",
        "--dataset.push_to_hub=false",
        "--dataset.single_task=Burger",
        "--dataset.episode_time_s=15",
        "--dataset.num_episodes=1",
        "--dataset.repo_id=Mozgi512/eval_hoge1",
    ]

    logger.info("Starting smoking action (duration=%ds)", duration)
    _run_command_for_seconds(cmd, duration)
    logger.info("Smoking action completed")


__all__ = ["execute_watching", "execute_smoking"]
