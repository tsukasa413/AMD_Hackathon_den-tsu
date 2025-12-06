from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import hw_to_dataset_features
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.robots.bi_so100_follower import BiSO100FollowerConfig, BiSO100Follower  
from lerobot.scripts.lerobot_record import record_loop
from lerobot.utils.control_utils import init_keyboard_listener
from lerobot.utils.utils import log_say
from lerobot.utils.visualization_utils import init_rerun


NUM_EPISODES = 5
FPS = 30
EPISODE_TIME_SEC = 60
TASK_DESCRIPTION = "Burger"
HF_MODEL_ID = "Mozgi512/act_burger_merged2_6000"
#HF_DATASET_ID = "Mozgi512/<eval_dataset_repo_id>"

# Create the robot configuration
camera_config = {"top": OpenCVCameraConfig(index_or_path=6, width=640, height=480, fps=FPS),"front": OpenCVCameraConfig(index_or_path=8, width=640, height=480, fps=FPS)}
robot_config = BiSO100FollowerConfig(
    left_arm_port="/dev/ttyACM2", right_arm_port="/dev/ttyACM0", id="bimanual_follower", cameras=camera_config
)

# Initialize the robot
robot = BiSO100Follower(robot_config)

# Initialize the policy
policy = ACTPolicy.from_pretrained(HF_MODEL_ID)

# Configure the dataset features
action_features = hw_to_dataset_features(robot.action_features, "action")
obs_features = hw_to_dataset_features(robot.observation_features, "observation")
dataset_features = {**action_features, **obs_features}

# Create the dataset
'''
dataset = LeRobotDataset.create(
    repo_id=HF_DATASET_ID,
    fps=FPS,
    features=dataset_features,
    robot_type=robot.name,
    use_videos=True,
    image_writer_threads=4,
)
'''
# Initialize the keyboard listener and rerun visualization
_, events = init_keyboard_listener()
init_rerun(session_name="recording")

# Connect the robot
robot.connect()

preprocessor, postprocessor = make_pre_post_processors(
    policy_cfg=policy,
    pretrained_path=HF_MODEL_ID,
    #dataset_stats=dataset.meta.stats,
)

for episode_idx in range(NUM_EPISODES):
    log_say(f"Running inference, recording eval episode {episode_idx + 1} of {NUM_EPISODES}")

    # Run the policy inference loop
    record_loop(
        robot=robot,
        events=events,
        fps=FPS,
        policy=policy,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        dataset=None,
        control_time_s=EPISODE_TIME_SEC,
        single_task=TASK_DESCRIPTION,
        display_data=True,
    )

    #dataset.save_episode()

# Clean up
robot.disconnect()
#dataset.push_to_hub()