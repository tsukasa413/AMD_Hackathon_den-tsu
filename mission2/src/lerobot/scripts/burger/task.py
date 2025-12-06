"""
task モジュール
右手と左手の両方がworking状態で働く動作を実行
"""

import time


def execute_working() -> None:
    """
    右手と左手の両方がworking状態で働く動作を実行
    この動作は一回実行されてから状態1に戻る
    """
    print("[Working] Executing working motion (both hands)...")
    
    try:
        # 働く動作の実装
        # ACTモデルまたはポリシーを使用して働く動作を生成
        
        # シミュレーション用の時間
        print("  Right hand: Working...")
        print("  Left hand: Working...")
        
        for i in range(3):
            print(f"  Progress: {(i+1)/3*100:.0f}%")
            time.sleep(1.0)
        
        print("[Working] Working motion completed")
    
    except Exception as e:
        print(f"[Error] Working error: {e}")
        raise