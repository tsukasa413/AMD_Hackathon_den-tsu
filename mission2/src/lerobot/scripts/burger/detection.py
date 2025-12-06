"""
検出モジュール
左手がwatchingの状態の時、周りに人がいるかを検出
"""

import cv2
from typing import Optional


def detect_person(camera_index: int = 0, frame_count: int = 1) -> bool:
    """
    ビデオキャプチャから人を検出
    
    Args:
        camera_index: カメラのインデックス（デフォルト: 0）
        frame_count: チェックするフレーム数（デフォルト: 1）
    
    Returns:
        bool: 人が検出されたかどうか
    """
    try:
        cap = cv2.VideoCapture(camera_index)
        
        if not cap.isOpened():
            print("[Warning] Could not open camera")
            return False
        
        # 顔検出のカスケード分類器をロード
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        person_detected = False
        
        for _ in range(frame_count):
            ret, frame = cap.read()
            
            if not ret:
                print("[Warning] Could not read frame from camera")
                break
            
            # グレースケール変換
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # 顔を検出
            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30)
            )
            
            if len(faces) > 0:
                person_detected = True
                print(f"[Detection] {len(faces)} person(s) detected")
                break
        
        cap.release()
        return person_detected
    
    except Exception as e:
        print(f"[Error] Detection error: {e}")
        return False


# テスト用
if __name__ == "__main__":
    result = detect_person()
    print(f"Person detected: {result}")
