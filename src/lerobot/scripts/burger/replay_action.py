import time
import threading

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.robots.so101_follower import SO101FollowerConfig, SO101Follower  
from lerobot.utils.robot_utils import busy_wait
from lerobot.utils.utils import log_say
from return_home import return_watching_home, return_working_home

# グローバルキャンセルフラグと保護用ロック
_action_cancel_flag = False
_cancel_lock = threading.Lock()


def set_action_cancel():
    """動作をキャンセルするフラグをセット"""
    global _action_cancel_flag
    with _cancel_lock:
        _action_cancel_flag = True
        print("[Action] Cancel flag set")


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


def execute_watching():
    """watching動作を実行"""
    reset_action_cancel()
    
    left_follower_config = SO101FollowerConfig(  
        port="/dev/ttyACM2",  
        id="den_follower_arm"  
    )  

    left_follower = SO101Follower(left_follower_config)  
    left_follower.connect()  

    dataset = LeRobotDataset("Mozgi512/record_watching_2", episodes=[4])
    actions = dataset.hf_dataset.select_columns("action")

    log_say("replay watching")
    try:
        for idx in range(dataset.num_frames):
            # キャンセルフラグをチェック
            if is_action_cancelled():
                print("[Action] Watching cancelled by detection")
                break
            
            t0 = time.perf_counter()

            action = {
                name: float(actions[idx]["action"][i]) for i, name in enumerate(dataset.features["action"]["names"])
            }
            left_follower.send_action(action)

            # 待機時間を計算
            sleep_time = 1.0 / dataset.fps - (time.perf_counter() - t0)
            # 待機時間が正の値の場合のみsleepを実行
            if sleep_time > 0:
                time.sleep(sleep_time)
    finally:
        # 動作完了後、watching_homeに戻る（安全のため）
        watching_home = {  
            "shoulder_pan.pos": 0,  
            "shoulder_lift.pos": -70,  
            "elbow_flex.pos": 60,  
            "wrist_flex.pos": 30.0,  
            "wrist_roll.pos": 0.0,  
            "gripper.pos": 20  
        }
        left_follower.send_action(watching_home)
        time.sleep(3.0)  # ホームポジションに戻るまで少し待機
        left_follower.disconnect()


def execute_apologize():
    """apologize動作を実行"""
    reset_action_cancel()
    
    left_follower_config = SO101FollowerConfig(  
        port="/dev/ttyACM2",  
        id="den_follower_arm"  
    )  

    left_follower = SO101Follower(left_follower_config)  
    left_follower.connect()  

    dataset = LeRobotDataset("Mozgi512/record_apologizing_1", episodes=[4])
    actions = dataset.hf_dataset.select_columns("action")

    log_say("replay apologizing")
    try:
        for idx in range(dataset.num_frames//5):
            # キャンセルフラグをチェック
            if is_action_cancelled():
                print("[Action] Apologizing cancelled by detection")
                break
            
            t0 = time.perf_counter()

            action = {
                name: float(actions[idx]["action"][i]) for i, name in enumerate(dataset.features["action"]["names"])
            }
            left_follower.send_action(action)

            elapsed = time.perf_counter() - t0
            sleep_time = 1.0 / dataset.fps - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
    finally:
        # 動作完了後、watching_homeに戻る（安全のため）
        watching_home = {  
            "shoulder_pan.pos": 0,  
            "shoulder_lift.pos": -70,  
            "elbow_flex.pos": 60,  
            "wrist_flex.pos": 30.0,  
            "wrist_roll.pos": 0.0,  
            "gripper.pos": 20  
        }
        left_follower.send_action(watching_home)
        time.sleep(1.0)  # ホームポジションに戻るまで少し待機
        left_follower.disconnect()