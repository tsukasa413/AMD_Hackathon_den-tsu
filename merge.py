import shutil
from pathlib import Path
from lerobot.datasets.lerobot_dataset import LeRobotDataset

# もしユーザーの環境でこの関数が利用可能ならインポート
# ※注意: LeRobotのバージョンによっては場所が異なる、または存在しない場合があります
try:
    from lerobot.datasets.dataset_tools import merge_datasets
except ImportError:
    print("エラー: merge_datasets 関数が見つかりません。")
    print("lerobotが最新版(source install)であることを確認してください。")
    exit(1)

def main():
    # ---------------------------------------------------------
    # 設定項目
    # ---------------------------------------------------------
    REPO_ID_1 = "Mozgi512/burger_merged"
    REPO_ID_2 = "Mozgi512/burger_3"
    
    # 新しく作るマージ後のリポジトリ名
    TARGET_REPO_ID = "Mozgi512/burger_merged2"
    
    # ローカルの一時保存場所（作業後に削除可能）
    LOCAL_DIR_1 = "./data/burger_merged"
    LOCAL_DIR_2 = "./data/burger_3"
    MERGED_DIR = "./data/burger_merged2"
    # ---------------------------------------------------------

    print(f"1. データセットをロード中: {REPO_ID_1} ...")
    ds1 = LeRobotDataset(REPO_ID_1, root=LOCAL_DIR_1)
    
    print(f"2. データセットをロード中: {REPO_ID_2} ...")
    ds2 = LeRobotDataset(REPO_ID_2, root=LOCAL_DIR_2)

    print("3. データセットをマージしています...")
    # 指定された関数を使用してマージ
    merged_dataset = merge_datasets(
        [ds1, ds2],
        output_repo_id=TARGET_REPO_ID,
        output_dir=MERGED_DIR
    )

    print(f"4. Hubへアップロードを開始します: {TARGET_REPO_ID}")
    # merge_datasetsの実装によっては自動でpushされない場合があるため、念のため明示的にpush
    # (すでにpushされている場合はスキップされます)
    try:
        merged_dataset.push_to_hub()
        print("アップロード完了！")
    except Exception as e:
        print(f"アップロード中に警告またはエラー: {e}")
        print("merge_datasets内で既にアップロードされている可能性があります。")

    print("\n完了しました。以下のコマンドで学習を開始できます：")
    print(f"lerobot-train --dataset.repo_id={TARGET_REPO_ID} ...")

    # お掃除（任意）
    # shutil.rmtree("./data") 

if __name__ == "__main__":
    main()