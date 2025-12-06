"""
検出モジュール
左手がwatchingの状態の時、周りに人がいるかを検出
MediaPipeを使用した人検知と正面向き判定
"""

import cv2
import mediapipe as mp


def detect_person(camera_index: int = 4, frame_count: int = 1) -> bool:
    """
    MediaPipeを使用してビデオキャプチャから正面を向いている人を検出
    
    処理フロー:
    1. フレーム取得
    2. 反時計回り90度回転
    3. 左側2/3、上側2/3の領域を切り取り
    4. MediaPipeで姿勢推定実行
    5. 頭部ランドマークで正面向き判定
    
    Args:
        camera_index: カメラのインデックス（デフォルト: 4 = /dev/video4）
        frame_count: チェックするフレーム数（デフォルト: 1）
    
    Returns:
        bool: 正面を向いている人が検出されたかどうか
    """
    try:
        # MediaPipeの姿勢推定を初期化
        mp_pose = mp.solutions.pose
        pose = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True
        )
        
        cap = cv2.VideoCapture(camera_index)
        
        if not cap.isOpened():
            print("[Warning] Could not open camera")
            return False
        
        person_detected = False
        person_count = 0
        
        for _ in range(frame_count):
            ret, frame = cap.read()
            
            if not ret:
                print("[Warning] Could not read frame from camera")
                break
            
            # 反時計回り90度回転
            rotated_frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
            
            # 回転後のサイズを取得
            height, width = rotated_frame.shape[:2]
            
            # 左側2/3、上側2/3の領域を切り取り
            crop_width = int(width * 2 / 3)
            crop_height = int(height * 2 / 3)
            cropped_frame = rotated_frame[0:crop_height, 0:crop_width]
            
            # MediaPipeで推論実行
            rgb_frame = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb_frame)
            
            # 人を検出し、正面向きか判定
            if results.pose_landmarks:
                # ランドマークから正面向き判定
                is_frontal = _is_frontal_pose(results.pose_landmarks)
                
                if is_frontal:
                    person_detected = True
                    person_count += 1
            
            if person_detected:
                print(f"[Detection] {person_count} frontal person(s) detected")
                break
        
        pose.close()
        cap.release()
        return person_detected
    
    except Exception as e:
        print(f"[Error] Detection error: {e}")
        return False


def _is_frontal_pose(landmarks) -> bool:
    """
    ランドマークから正面向きか判定
    
    判定方法：
    - 左肩と右肩のy座標の差が小さい（肩が水平）
    - 左目と右目のx座標の差が大きい（顔が正面）
    - 鼻がほぼ中央
    
    Args:
        landmarks: MediaPipeのランドマーク
    
    Returns:
        bool: 正面向きと判定されたかどうか
    """
    # 主要なランドマークインデックス
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_EYE = 2
    RIGHT_EYE = 5
    NOSE = 0
    
    # ランドマークが十分な信頼度を持っているか確認
    if landmarks.landmark[LEFT_SHOULDER].visibility < 0.5 or \
       landmarks.landmark[RIGHT_SHOULDER].visibility < 0.5 or \
       landmarks.landmark[LEFT_EYE].visibility < 0.5 or \
       landmarks.landmark[RIGHT_EYE].visibility < 0.5:
        return False
    
    # 肩の高さの差（小さいほど水平 = 正面）
    shoulder_y_diff = abs(
        landmarks.landmark[LEFT_SHOULDER].y - 
        landmarks.landmark[RIGHT_SHOULDER].y
    )
    
    # 目の水平距離（大きいほど正面）
    eye_x_diff = abs(
        landmarks.landmark[LEFT_EYE].x - 
        landmarks.landmark[RIGHT_EYE].x
    )
    
    # 鼻が画像中央付近か（0.3～0.7の範囲）
    nose_x = landmarks.landmark[NOSE].x
    nose_centered = 0.3 < nose_x < 0.7
    
    # 正面向きの判定条件
    is_frontal = (shoulder_y_diff < 0.1 and eye_x_diff > 0.05 and nose_centered)
    
    return is_frontal


# テスト用
if __name__ == "__main__":
    try:
        mp_pose = mp.solutions.pose
        pose = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True
        )
        
        cap = cv2.VideoCapture(6)
        
        if not cap.isOpened():
            print("[Error] Could not open camera")
        else:
            ret, frame = cap.read()
            
            if ret:
                # 反時計回り90度回転
                rotated_frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
                
                # 回転後のサイズを取得
                height, width = rotated_frame.shape[:2]
                
                # 左側2/3、上側2/3の領域を切り取り
                crop_width = int(width * 2 / 3)
                crop_height = int(height * 2 / 3)
                cropped_frame = rotated_frame[0:crop_height, 0:crop_width]
                
                print(f"[Info] Original frame size: {frame.shape}")
                print(f"[Info] Rotated frame size: {rotated_frame.shape}")
                print(f"[Info] Cropped frame size: {cropped_frame.shape}")
                
                # MediaPipeで推論実行
                rgb_frame = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2RGB)
                results = pose.process(rgb_frame)
                
                # 切り取り結果を描画
                visualization_frame = rotated_frame.copy()
                
                # 切り取り領域を矩形で描画（青色）
                cv2.rectangle(visualization_frame, (0, 0), (crop_width, crop_height), (255, 0, 0), 3)
                
                # ランドマークを描画
                if results.pose_landmarks:
                    mp_drawing = mp.solutions.drawing_utils
                    annotated_cropped = cropped_frame.copy()
                    mp_drawing.draw_landmarks(
                        annotated_cropped,
                        results.pose_landmarks,
                        mp_pose.POSE_CONNECTIONS
                    )
                    visualization_frame[0:crop_height, 0:crop_width] = annotated_cropped
                    
                    # 正面向き判定
                    is_frontal = _is_frontal_pose(results.pose_landmarks)
                    print(f"[Info] Frontal pose detected: {is_frontal}")
                
                # 画像をファイルに保存
                output_path = "pose_detection_result.jpg"
                cv2.imwrite(output_path, visualization_frame)
                print(f"[Info] Detection result saved to: {output_path}")
            else:
                print("[Error] Could not read frame from camera")
            
            pose.close()
            cap.release()
    
    except Exception as e:
        print(f"[Error] Test error: {e}")