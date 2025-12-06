"""
右手と左手のロボットアームを制御する中心部
状態マシンを使用して、各状態間の遷移を管理
"""

import time
from enum import Enum
from typing import Tuple
from dataclasses import dataclass

# インポート
from smoking import execute_smoking
from apologize import execute_apologize
from return_home import execute_return_home
from detection import detect_person
from watching import execute_watching
from task import execute_working


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
        self.idle_threshold_sec = 3  # 3秒でsmoking状態に遷移
        self.person_detected = False
        
    def update_detection(self) -> bool:
        """
        人検知の情報を更新
        Returns:
            bool: 人が検知されたかどうか
        """
        self.person_detected = detect_person()
        return self.person_detected
    
    def execute_scenario_1_sabori(self) -> Tuple[bool, str]:
        """
        シナリオ1：さぼる
        右手：idle → smoking
        左手：watching（ループ）
        
        Returns:
            Tuple[bool, str]: (状態遷移があったか, 次の状態)
        """
        print(f"\n[Scenario 1: Sabori] {self.state}")
        
        # 左手がwatchingで人検知をチェック
        self.update_detection()
        
        if self.person_detected:
            print("⚠ Person detected during sabori! Transitioning to Scenario 2 (Ayamaru)")
            return True, "scenario_2_ayamaru"
        
        # 右手がidleからsmokingへの遷移
        if self.state.right_hand == RightHandState.IDLE:
            if self.right_hand_idle_start_time is None:
                self.right_hand_idle_start_time = time.time()
            
            elapsed = time.time() - self.right_hand_idle_start_time
            
            if elapsed >= self.idle_threshold_sec:
                # smoking動作を開始
                self.state.right_hand = RightHandState.SMOKING
                print(f"✓ Right hand transitioned to SMOKING (after {elapsed:.1f}s)")
                
                # smoking動作を実行
                execute_smoking()
                
                # smoking動作が終わったのでworkingシナリオに遷移
                print("✓ Smoking action completed. Transitioning to Scenario 3 (Working)")
                
                # home位置に戻る
                execute_return_home()
                
                return True, "scenario_3_work"
            else:
                # まだ3秒未満、waiting状態を保つ
                print(f"Waiting for idle threshold... ({elapsed:.1f}s / {self.idle_threshold_sec}s)")
                time.sleep(0.1)
                return False, "scenario_1_sabori"
        
        # ループを続ける
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
        
        # 状態を更新
        self.state.right_hand = RightHandState.IDLE
        self.state.left_hand = LeftHandState.APOLOGIZE
        
        # apologize動作を実行（一回のみ）
        execute_apologize()
        print("✓ Apologize action completed")
        
        # home位置に戻る
        execute_return_home()
        
        # working シナリオに遷移
        print("✓ Transitioning to Scenario 3 (Working)")
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
        self.state.right_hand = RightHandState.WORKING
        self.state.left_hand = LeftHandState.WORKING
        
        # working動作を実行
        execute_working()
        print("✓ Working action completed")
        
        # home位置に戻る
        execute_return_home()
        
        # シナリオ1に戻る
        print("✓ Transitioning back to Scenario 1 (Sabori)")
        
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
                print(f"\n--- Cycle {cycle_count} ---")
                
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
