"""
検出モジュール
左手がwatchingの状態の時、周りに人がいるかを検出
YOLOv8を使用した人検知
"""

import cv2
from ultralytics import YOLO

# グローバルにモデルとカメラを保持（初回のみロード）
_model = None
_cap = None

def _initialize_detection(camera_index: int = 4, model_name: str = "yolov8s.pt"):
    """検出用のモデルとカメラを初期化"""
    global _model, _cap
    
    if _model is None:
        _model = YOLO(model_name)
    
    if _cap is None or not _cap.isOpened():
        _cap = cv2.VideoCapture(camera_index)
        if not _cap.isOpened():
            return False
    
    return True


def detect_person(camera_index: int = 4, frame_count: int = 1, model_name: str = "yolov8s.pt") -> bool:
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
        model_name: YOLOモデル名（デフォルト: yolov8s.pt）
    
    Returns:
        bool: 人が検出されたかどうか
    """
    global _model, _cap
    
    try:
        # モデルとカメラを初期化
        if not _initialize_detection(camera_index, model_name):
            return False
        
        person_detected = False
        person_count = 0
        
        for _ in range(frame_count):
            ret, frame = _cap.read()
            
            if not ret:
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
            results = _model(cropped_frame, verbose=False)
            
            # 人（クラスID=0）を検出
            # バウンディングボックスの大きさに閾値を設ける
            # 縦横がともに画角の2/3より大きいことを条件とする
            min_box_width = crop_width * 2 / 3
            min_box_height = crop_height * 2 / 3
            min_confidence = 0.6  # 信頼度の閾値
            
            detected_info = []  # 検出された人の情報を保存
            
            for result in results:
                for box in result.boxes:
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    
                    # YOLO では person クラスID = 0
                    if class_id == 0:
                        # バウンディングボックスのサイズを取得
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        box_width = x2 - x1
                        box_height = y2 - y1
                        
                        # 信頼度が0.6以上、かつ縦横がともに画角の2/3より大きい場合のみ検出とする
                        if confidence >= min_confidence and box_width > min_box_width and box_height > min_box_height:
                            person_detected = True
                            person_count += 1
                            detected_info.append({
                                'confidence': confidence,
                                'width': box_width,
                                'height': box_height
                            })
            
            if person_detected:
                # 検出結果を画像として保存（人が検出された時のみ）
                annotated_frame = results[0].plot()
                import datetime
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                filepath = f"/tmp/person_detected_{timestamp}.jpg"
                cv2.imwrite(filepath, annotated_frame)
                
                # 検出情報を出力
                for i, info in enumerate(detected_info, 1):
                    print(f"[Detection] Person {i}: confidence={info['confidence']:.2f}, size={info['width']:.1f}x{info['height']:.1f}")
                print(f"[Detection] Image saved to {filepath}")
                break
        
        return person_detected
    
    except Exception as e:
        print(f"[Error] Detection error: {e}")
        return False


# テスト用
if __name__ == "__main__":
    try:
        # YOLOv8モデルをロード
        model = YOLO("yolov8s.pt")
        
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
                
                # バウンディングボックスの大きさに閾値を設ける
                crop_height, crop_width = cropped_frame.shape[:2]
                min_box_width = crop_width / 3
                min_box_height = crop_height / 3
                
                for result in results:
                    for box in result.boxes:
                        class_id = int(box.cls[0])
                        if class_id == 0:
                            # バウンディングボックスのサイズを取得
                            x1, y1, x2, y2 = box.xyxy[0].tolist()
                            box_width = x2 - x1
                            box_height = y2 - y1
                            
                            # 縦横がともに画角の1/3より大きい場合のみ検出とする
                            if box_width > min_box_width and box_height > min_box_height:
                                person_detected = True
                                person_count += 1
                                print(f"[Info] Box size: {box_width:.1f}x{box_height:.1f} (threshold: {min_box_width:.1f}x{min_box_height:.1f})")
                            else:
                                print(f"[Info] Box too small: {box_width:.1f}x{box_height:.1f} (threshold: {min_box_width:.1f}x{min_box_height:.1f})")
                
                print(f"Person detected: {person_detected}")
                if person_detected:
                    print(f"Number of persons: {person_count}")
            else:
                print("[Error] Could not read frame from camera")
            
            cap.release()
    
    except Exception as e:
        print(f"[Error] Test error: {e}")
