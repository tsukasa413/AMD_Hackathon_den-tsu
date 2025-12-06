"""
検出モジュール
左手がwatchingの状態の時、周りに人がいるかを検出
YOLOv8を使用した人検知
"""

import cv2
from ultralytics import YOLO


def detect_person(camera_index: int = 4, frame_count: int = 1, model_name: str = "yolov8n.pt") -> bool:
    """
    YOLOv8を使用してビデオキャプチャから人を検出
    
    処理フロー:
    1. フレーム取得
    2. 反時計回り90度回転
    3. 左側2/3、上側2/3の領域を切り取り
    4. YOLOで推論実行
    
    Args:
        camera_index: カメラのインデックス（デフォルト: 4 = /dev/video4）
        frame_count: チェックするフレーム数（デフォルト: 1）
        model_name: YOLOモデル名（デフォルト: yolov8n.pt）
    
    Returns:
        bool: 人が検出されたかどうか
    """
    try:
        # YOLOv8モデルをロード
        model = YOLO(model_name)
        
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
            
            # YOLOで推論実行
            results = model(cropped_frame, verbose=False)
            
            # 人（クラスID=0）を検出
            for result in results:
                for box in result.boxes:
                    class_id = int(box.cls[0])
                    # YOLO では person クラスID = 0
                    if class_id == 0:
                        person_detected = True
                        person_count += 1
            
            if person_detected:
                print(f"[Detection] {person_count} person(s) detected")
                break
        
        cap.release()
        return person_detected
    
    except Exception as e:
        print(f"[Error] Detection error: {e}")
        return False


# テスト用
if __name__ == "__main__":
    try:
        # YOLOv8モデルをロード
        model = YOLO("yolov8n.pt")
        
        cap = cv2.VideoCapture(4)
        
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
                
                # YOLOで推論実行（切り取った画像）
                results = model(cropped_frame, verbose=False)
                
                # 切り取り結果を描画したフレーム（回転画像に切り取り領域の線を描画）
                visualization_frame = rotated_frame.copy()
                
                # 切り取り領域を矩形で描画（青色）
                cv2.rectangle(visualization_frame, (0, 0), (crop_width, crop_height), (255, 0, 0), 3)
                
                # 切り取り領域内にYOLO検出結果を描画
                annotated_cropped = results[0].plot()
                
                # 回転画像に切り取り領域を合成（YOLO結果を含む）
                visualization_frame[0:crop_height, 0:crop_width] = annotated_cropped
                
                # 画像をファイルに保存
                output_path = "yolo_detection_result.jpg"
                cv2.imwrite(output_path, visualization_frame)
                print(f"[Info] Detection result saved to: {output_path}")
                
                # 人が検出されたかを表示
                person_detected = False
                person_count = 0
                for result in results:
                    for box in result.boxes:
                        class_id = int(box.cls[0])
                        if class_id == 0:
                            person_detected = True
                            person_count += 1
                
                print(f"Person detected: {person_detected}")
                if person_detected:
                    print(f"Number of persons: {person_count}")
            else:
                print("[Error] Could not read frame from camera")
            
            cap.release()
    
    except Exception as e:
        print(f"[Error] Test error: {e}")
