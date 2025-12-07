"""
右手と左手のロボットアームを制御する中心部
状態マシンを使用して、各状態間の遷移を管理
"""

import time
from enum import Enum
from typing import Tuple
from dataclasses import dataclass
import threading

# インポート\
from return_home import return_watching_home, return_working_home
from detection import detect_person
from replay_action import execute_watching, execute_apologize, set_action_cancel as set_replay_cancel
from estimation import execute_smoking, execute_working, set_action_cancel as set_estimation_cancel

class RightHandState(Enum):
    """右手の状態"""
    IDLE = "idle"
    SMOKING = "smoking"
    WORKING = "working"


class LeftHandState(Enum):
    """左手の状態"""
    WATCHING = "watching"
    APOLOGIZE = "apologize"
    WORKING = "working"


@dataclass
class RobotState:
    """ロボットの現在の状態を保持"""
    right_hand: RightHandState = RightHandState.IDLE
    left_hand: LeftHandState = LeftHandState.WATCHING
    current_scenario: str = "scenario_1_sabori"  # scenario_1_sabori, scenario_2_ayamaru, scenario_3_work
    
    def __str__(self):
        return f"Right: {self.right_hand.value}, Left: {self.left_hand.value}, Scenario: {self.current_scenario}"


class BurgerRobotController:
    """バーガーロボット制御の中心部"""
    
    def __init__(self):
        self.state = RobotState()
        self.right_hand_idle_start_time = None
        self.idle_threshold_sec = 5  # 3秒でsmoking状態に遷移
        self.person_detected = False
        self.detection_thread = None
        self.detection_running = False
        
        # 右手・左手スレッド用フラグ
        self.right_hand_thread = None
        self.left_hand_thread = None
        self.right_hand_running = False
        self.left_hand_running = False
        
    def update_detection(self) -> bool:
        """
        人検知の情報を更新
        Returns:
            bool: 人が検知されたかどうか
        """
        self.person_detected = detect_person()
        return self.person_detected
    
    def _background_detection_loop(self):
        """バックグラウンドで人検知を常に更新"""
        while self.detection_running:
            result = detect_person()
            
            # 人が検知された場合、フラグをセットしてプロセスを中断
            if result:
                print(f"[Detection] Person detected! Cancelling actions.")
                self.person_detected = True
                # 実行中のプロセスをキャンセル
                set_replay_cancel()
                set_estimation_cancel()
            else:
                self.person_detected = False
            
            time.sleep(0.1)  # 適度な間隔で更新
    
    def _background_right_hand_loop(self):
        """右手のバックグラウンドループ"""
        smoking_transitioned = False  # SMOKING状態への遷移が完了したかを記録
        
        while self.right_hand_running:
            # 経過時間を計算
            elapsed = time.time() - self.right_hand_idle_start_time if self.right_hand_idle_start_time else 0
            
            # 3秒未満: IDLE状態を継続
            if elapsed < self.idle_threshold_sec:
                self.state.right_hand = RightHandState.IDLE
                smoking_transitioned = False
            else:
                # 3秒以上: SMOKING状態に遷移（一度だけ）
                if not smoking_transitioned:
                    print(f"[Right Hand] Transitioned to SMOKING after {elapsed:.2f}s")
                    smoking_transitioned = True
                self.state.right_hand = RightHandState.SMOKING
                
                # SMOKING状態に遷移した後、smoking動作を実行
                if smoking_transitioned:
                    print("[Execute] Smoking action started")
                    execute_smoking()
                    print("[Execute] Smoking action completed")
                    # smoking動作が完了後、ループを抜ける
                    if self.person_detected:
                        break
            
            time.sleep(0.05)  # 定期的に状態を更新
    
    def _background_left_hand_loop(self):
        """左手のバックグラウンドループ"""
        while self.left_hand_running:
            # 左手は常にWATCHING状態で見渡す
            self.state.left_hand = LeftHandState.WATCHING
            
            # watching動作を実行（キャンセルフラグをチェック）
            print("[Execute] Watching action started")
            execute_watching()
            print("[Execute] Watching action completed")
            
            # キャンセルフラグがセットされたら終了
            if self.person_detected:
                break
            
            time.sleep(0.05)
    
    def execute_scenario_1_sabori(self) -> Tuple[bool, str]:
        """
        シナリオ1：さぼる
        
        ループフロー:
        1. ループ開始時: watching_home に移動、右手=IDLE、左手=WATCHING、スレッド開始
        2. バックグラウンド：
           - 右手スレッド: 3秒後にIDLE→SMOKING状態に遷移
           - 左手スレッド: 常にWATCHING状態でexecute_watching()を実行
           - 検知スレッド: 常に人検知をチェック
        3. 人検知時：全スレッドをキャンセルして状態遷移
        
        Returns:
            Tuple[bool, str]: (状態遷移があったか, 次の状態)
        """
        # ループの最初（初回エントリー時）
        if self.state.left_hand != LeftHandState.WATCHING or self.right_hand_idle_start_time is None:
            self.state.current_scenario = "scenario_1_sabori"
            print(f"\n[Scenario 1: Sabori] {self.state}")
            # watching_home位置に移動
            print("[Return] Moving to watching home")
            return_watching_home()
            # 状態を初期化
            self.state.right_hand = RightHandState.IDLE
            self.state.left_hand = LeftHandState.WATCHING
            # タイマーを開始
            self.right_hand_idle_start_time = time.time()
            
            # バックグラウンドスレッドを開始
            if not self.detection_running:
                self.detection_running = True
                self.detection_thread = threading.Thread(target=self._background_detection_loop, daemon=True)
                self.detection_thread.start()
            
            if not self.right_hand_running:
                self.right_hand_running = True
                self.right_hand_thread = threading.Thread(target=self._background_right_hand_loop, daemon=True)
                self.right_hand_thread.start()
            
            if not self.left_hand_running:
                self.left_hand_running = True
                self.left_hand_thread = threading.Thread(target=self._background_left_hand_loop, daemon=True)
                self.left_hand_thread.start()
            
            time.sleep(0.1)
            return False, "scenario_1_sabori"
        
        # 人検知時のみ出力
        
        if self.person_detected:
            # 全スレッドを停止
            self.right_hand_running = False
            self.left_hand_running = False
            self.detection_running = False
            
            # スレッドが完全に停止するまで待機
            time.sleep(0.5)
            
            # 人を検知したら常にWorkingに遷移
            print("[Return] Moving to working home")
            return_working_home()
            print("\n→ Transition to Scenario 3 (Working)")
            self.right_hand_idle_start_time = None
            return True, "scenario_3_work"
        
        # スレッド実行中、静かにループを継続
        time.sleep(0.5)
        return False, "scenario_1_sabori"
    
    def execute_scenario_2_ayamaru(self) -> Tuple[bool, str]:
        """
        シナリオ2：謝る
        右手：idle
        左手：apologize（一回のみ実行）
        
        Returns:
            Tuple[bool, str]: (状態遷移があったか, 次の状態)
        """
        print(f"\n[Scenario 2: Ayamaru] {self.state}")
        
        # 検知スレッドを停止（シナリオ2と3では人検知不要）
        self.detection_running = False
        
        # 状態を更新
        self.state.current_scenario = "scenario_2_ayamaru"
        self.state.right_hand = RightHandState.IDLE
        self.state.left_hand = LeftHandState.APOLOGIZE
        
        # apologize動作を実行（一回のみ）
        # execute_apologize()
        
        # watching_home位置に戻る
        print("[Return] Moving to working home")
        return_working_home()        
        # working シナリオに遷移
        print("\n→ Transition to Scenario 3 (Working)")
        return True, "scenario_3_work"
    
    def execute_scenario_3_work(self) -> Tuple[bool, str]:
        """
        シナリオ3：働く
        右手：working
        左手：working
        
        Returns:
            Tuple[bool, str]: (状態遷移があったか, 次の状態)
        """
        print(f"\n[Scenario 3: Working] {self.state}")
        
        # 状態を更新
        self.state.current_scenario = "scenario_3_work"
        self.state.right_hand = RightHandState.WORKING
        self.state.left_hand = LeftHandState.WORKING
        
        # working動作を実行
        print("[Execute] Working action started")
        execute_working()
        print("[Execute] Working action completed")
        
        # watching_home位置に戻る
        print("[Return] Moving to watching home")
        return_watching_home()
        print("\n→ Transition to Scenario 1 (Sabori)")
        return True, "scenario_1_sabori"
        
        # リセット
        self.state.right_hand = RightHandState.IDLE
        self.state.left_hand = LeftHandState.WATCHING
        self.right_hand_idle_start_time = None
        
        return True, "scenario_1_sabori"
    
    def run(self, max_cycles: int = None):
        """
        状態マシンのメインループ
        
        Args:
            max_cycles: 最大サイクル数（Noneの場合は無制限）
        """
        cycle_count = 0
        current_scenario = "scenario_1_sabori"
        
        try:
            print("=" * 60)
            print("Burger Robot Control System Started")
            print("=" * 60)
            
            while max_cycles is None or cycle_count < max_cycles:
                cycle_count += 1
                
                # 現在のシナリオを実行
                if current_scenario == "scenario_1_sabori":
                    transition, next_scenario = self.execute_scenario_1_sabori()
                    if transition:
                        current_scenario = next_scenario
                
                elif current_scenario == "scenario_2_ayamaru":
                    transition, next_scenario = self.execute_scenario_2_ayamaru()
                    if transition:
                        current_scenario = next_scenario
                
                elif current_scenario == "scenario_3_work":
                    transition, next_scenario = self.execute_scenario_3_work()
                    if transition:
                        current_scenario = next_scenario
                
        except KeyboardInterrupt:
            print("\n\n[INFO] Control interrupted by user")
        except Exception as e:
            print(f"\n[ERROR] An error occurred: {e}")
            raise
        finally:
            print("\n" + "=" * 60)
            print("Burger Robot Control System Stopped")
            print("=" * 60)


def main():
    """メインエントリーポイント"""
    controller = BurgerRobotController()
    
    # max_cyclesを指定して実行制限、またはNoneで無制限
    controller.run(max_cycles=None)


if __name__ == "__main__":
    main()
