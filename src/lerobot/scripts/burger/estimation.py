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
import threading
import os
import shutil
from datetime import datetime
from return_home import return_watching_home, return_working_home

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# グローバルキャンセルフラグと保護用ロック
_action_cancel_flag = False
_cancel_lock = threading.Lock()


def set_action_cancel():
    """動作をキャンセルするフラグをセット"""
    global _action_cancel_flag
    with _cancel_lock:
        _action_cancel_flag = True
        logger.info("Action cancel flag set")


def reset_action_cancel():
    """キャンセルフラグをリセット"""
    global _action_cancel_flag
    with _cancel_lock:
        _action_cancel_flag = False


def is_action_cancelled():
    """キャンセルフラグがセットされているかチェック"""
    global _action_cancel_flag
    with _cancel_lock:
        return _action_cancel_flag


def _run_command_for_seconds(cmd: Sequence[str], seconds: int) -> int:
    """Run `cmd` as a subprocess for `seconds`, then terminate it.

    The subprocess is started and allowed to run for `seconds` seconds. After
    that sleep the subprocess is terminated (SIGTERM) and, if it doesn't exit
    within a short timeout, it is killed (SIGKILL).

    キャンセルフラグがセットされた場合は即座に終了する。

    Returns the process return code (may be None until process terminates).
    """
    logger.info("Running command for %d seconds: %s", seconds, " ".join(cmd))
    # Capture output for error diagnostics, provide stdin to auto-respond to prompts
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, stdin=subprocess.PIPE, text=True)
    
    # キャリブレーションプロンプトに自動的にENTERを送信
    try:
        proc.stdin.write("\n")
        proc.stdin.flush()
    except Exception as e:
        logger.warning("Failed to send ENTER to stdin: %s", e)
    
    start_time = time.time()
    early_exit = False
    try:
        while time.time() - start_time < seconds:
            # キャンセルフラグをチェック
            if is_action_cancelled():
                logger.info("Action cancelled by detection; terminating process")
                break
            # プロセスが早期終了していないかチェック
            if proc.poll() is not None:
                elapsed = time.time() - start_time
                logger.warning("Process terminated early after %.2f seconds with code: %s", elapsed, proc.returncode)
                early_exit = True
                break
            time.sleep(0.1)  # 定期的にチェック
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt received; terminating child process")
    finally:
        # Try graceful termination first
        if not early_exit:
            try:
                proc.terminate()
            except Exception:
                pass

        try:
            stdout, stderr = proc.communicate(timeout=5)
            if proc.returncode != 0 or early_exit:
                if stderr:
                    logger.error("STDERR (last 3000 chars): %s", stderr[-3000:])
                if stdout:
                    logger.info("STDOUT (last 3000 chars): %s", stdout[-3000:])
        except subprocess.TimeoutExpired:
            logger.info("Process did not exit after SIGTERM; killing it")
            try:
                proc.kill()
            except Exception:
                pass
            proc.wait()

    logger.info("Process finished with return code: %s", proc.returncode)
    return proc.returncode


def execute_working(duration: int = 30) -> None:
    """Execute the watching CLI for `duration` seconds.

    This runs the `lerobot-record` command with parameters used by the burger
    robot policy and waits `duration` seconds before terminating the process.
    """
    reset_action_cancel()
    
    # キャッシュディレクトリが存在する場合は削除
    cache_dir = "/home/amddemo/.cache/huggingface/lerobot/Mozgi512/eval_hoge1"
    if os.path.exists(cache_dir):
        logger.info("Removing existing cache directory: %s", cache_dir)
        try:
            shutil.rmtree(cache_dir)
        except Exception as e:
            logger.error("Failed to remove cache directory: %s", e)
    
    cmd = [
        "lerobot-record",
        "--robot.type=bi_so100_follower",
        "--robot.left_arm_port=/dev/ttyACM2",
        "--robot.right_arm_port=/dev/ttyACM0",
        "--robot.id=bimanual_follower",
        "--policy.path=Mozgi512/act_burger_final_8000",
        "--robot.cameras={ top: {type: opencv, index_or_path: 6, width: 640, height: 480, fps: 30},front: {type: opencv, index_or_path: 8, width: 640, height: 480, fps: 30}}",
        "--display_data=false",
        "--dataset.push_to_hub=false",
        "--dataset.single_task=Burger",
        "--dataset.episode_time_s=30",
        "--dataset.num_episodes=1",
        "--dataset.repo_id=Mozgi512/eval_hoge1",
    ]

    logger.info("Starting watching action (duration=%ds)", duration)
    _run_command_for_seconds(cmd, duration)
    logger.info("Watching action completed")
    time.sleep(1.0)
    return_working_home()
    time.sleep(1.0)

def execute_smoking(duration: int = 20) -> None:
    """Execute the smoking action.

    Currently this function uses the same command/structure as
    `execute_watching` as a placeholder. Replace the command contents here
    when the actual smoking CLI is available.
    """
    reset_action_cancel()

    # キャッシュディレクトリが存在する場合は削除
    cache_dir = "/home/amddemo/.cache/huggingface/lerobot/Mozgi512/eval_smoking_2"
    if os.path.exists(cache_dir):
        logger.info("Removing existing cache directory: %s", cache_dir)
        try:
            shutil.rmtree(cache_dir)
        except Exception as e:
            logger.error("Failed to remove cache directory: %s", e)
    
    cmd = [
        "lerobot-record",
        "--robot.type=so101_follower",
        "--robot.port=/dev/ttyACM0",
        "--robot.id=tsu_follower_arm",
        "--robot.cameras={ top: {type: opencv, index_or_path: 6, width: 640, height: 480, fps: 30},front: {type: opencv, index_or_path: 8, width: 640, height: 480, fps: 30}}",
        "--display_data=false",
        "--dataset.repo_id=Mozgi512/eval_smoking_2",
        "--dataset.single_task=Smoking",
        "--policy.path=Mozgi512/act_smoking_ckpt_1"
    ]

    logger.info("Starting smoking action (duration=%ds)", duration)
    _run_command_for_seconds(cmd, duration)
    logger.info("Smoking action completed")


__all__ = ["execute_watching", "execute_smoking"]
