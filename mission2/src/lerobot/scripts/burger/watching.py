"""
watching モジュール
左手がwatchingの状態で周囲を監視
"""

import time


def execute_watching(duration: float = 1.0) -> None:
    """
    左手がwatching状態で周囲を監視する動作
    
    Args:
        duration: 監視の継続時間（秒）
    """
    print("[Watching] Left hand is watching surroundings...")
    
    try:
        # 監視動作の実装（ここではシミュレーション）
        start_time = time.time()
        
        while time.time() - start_time < duration:
            # 周囲を監視する動作
            elapsed = time.time() - start_time
            print(f"  Watching... ({elapsed:.1f}s / {duration}s)")
            time.sleep(0.5)
        
        print("[Watching] Watching completed")
    
    except Exception as e:
        print(f"[Error] Watching error: {e}")
        raise
