import time  
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig  
from lerobot.policies.act.modeling_act import ACTPolicy  
from lerobot.policies.factory import make_pre_post_processors  
from lerobot.robots.so101_follower.config_so101_follower import SO101FollowerConfig  
from lerobot.robots.so101_follower.so101_follower import SO101Follower  
from lerobot.utils.control_utils import predict_action, init_keyboard_listener  
from lerobot.utils.utils import get_safe_torch_device, log_say  
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data  
from lerobot.datasets.utils import build_dataset_frame, hw_to_dataset_features  
from lerobot.policies.utils import make_robot_action  
from lerobot.processor import make_default_processors  
from lerobot.utils.constants import OBS_STR  
from lerobot.utils.robot_utils import precise_sleep  


def execute_smoking(
    num_episodes: int = 5,
    fps: int = 30,
    episode_time_sec: int = 60,
    task_description: str = "タスクの説明",
    hf_model_id: str = "<hf_username>/<model_repo_id>",
) -> None:
    """
    右手のさぼる動作を実行
    
    Args:
        num_episodes: 評価エピソード数
        fps: フレームレート
        episode_time_sec: エピソードの継続時間（秒）
        task_description: タスクの説明
        hf_model_id: HuggingFace ModelIDのパス
    """
    # カメラ設定  
    camera_config = {"front": OpenCVCameraConfig(index_or_path=0, width=640, height=480, fps=fps)}  
      
    # ロボット設定  
    robot_config = SO101FollowerConfig(  
        port="/dev/ttyACM0",  
        id="den_follower_arm",  
        cameras=camera_config  
    )  
      
    # ロボットを初期化  
    robot = SO101Follower(robot_config)  
      
    # ポリシーを読み込み  
    policy = ACTPolicy.from_pretrained(hf_model_id)  
      
    # データセット特徴を設定（推論用）  
    action_features = hw_to_dataset_features(robot.action_features, "action")  
    obs_features = hw_to_dataset_features(robot.observation_features, "observation")  
    dataset_features = {**action_features, **obs_features}  
      
    # プリプロセッサとポストプロセッサを作成（統計情報なしで推論用に作成）  
    preprocessor, postprocessor = make_pre_post_processors(  
        policy_cfg=policy.config,  
        pretrained_path=hf_model_id,  
        dataset_stats=None,  # 統計情報がない場合はNone  
    )  
      
    # プロセッサパイプラインを作成  
    _, robot_action_processor, robot_observation_processor = make_default_processors()  
      
    # キーボードリスナーを初期化  
    _, events = init_keyboard_listener()  
      
    # Rerunによる可視化を初期化（オプション）  
    init_rerun(session_name="evaluation")  
      
    # ロボットに接続  
    robot.connect()  
      
    # 評価ループ  
    for episode_idx in range(num_episodes):  
        log_say(f"評価エピソード {episode_idx + 1} / {num_episodes} を実行中")  
          
        # ポリシーとプロセッサをリセット  
        policy.reset()  
        preprocessor.reset()  
        postprocessor.reset()  
          
        start_episode_t = time.perf_counter()  
        timestamp = 0  
          
        while timestamp < episode_time_sec and not events["exit_early"]:  
            start_loop_t = time.perf_counter()  
              
            # ロボットから観測を取得  
            obs = robot.get_observation()  
              
            # 観測を処理  
            obs_processed = robot_observation_processor(obs)  
              
            # データセットフレームを構築  
            observation_frame = build_dataset_frame(dataset_features, obs_processed, prefix=OBS_STR)  
              
            # ポリシーからアクションを予測  
            action_values = predict_action(  
                observation=observation_frame,  
                policy=policy,  
                device=get_safe_torch_device(policy.config.device),  
                preprocessor=preprocessor,  
                postprocessor=postprocessor,  
                use_amp=policy.config.use_amp,  
                task=task_description,  
                robot_type=robot.robot_type,  
            )  
              
            # アクションをロボット形式に変換  
            robot_action = make_robot_action(action_values, dataset_features)  
              
            # アクションを処理  
            robot_action_to_send = robot_action_processor((robot_action, obs))  
              
            # ロボットにアクションを送信  
            robot.send_action(robot_action_to_send)  
              
            # 可視化（オプション）  
            log_rerun_data(observation=obs_processed, action=action_values)  
              
            # FPSを維持するためにスリープ  
            dt_s = time.perf_counter() - start_loop_t  
            precise_sleep(1 / fps - dt_s)  
              
            timestamp = time.perf_counter() - start_episode_t  
          
        log_say(f"エピソード {episode_idx + 1} 完了")  
          
        # 早期終了フラグをリセット  
        if events["exit_early"]:  
            events["exit_early"] = False  
      
    # クリーンアップ  
    log_say("評価完了")  
    robot.disconnect()