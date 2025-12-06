import time

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.robots.so101_follower import SO101FollowerConfig, SO101Follower  
from lerobot.utils.robot_utils import busy_wait
from lerobot.utils.utils import log_say

left_follower_config = SO101FollowerConfig(  
    port="/dev/ttyACM0",  
    id="den_follower_arm"  
)  

left_follower = SO101Follower(left_follower_config)  
left_follower.connect()  

#dataset = LeRobotDataset("Mozgi512/record_watching_2", episodes=[4])   #watching
dataset = LeRobotDataset("Mozgi512/record_apologizing_1", episodes=[4]) #apologizing
actions = dataset.hf_dataset.select_columns("action")

log_say("replay")
for idx in range(dataset.num_frames):
    t0 = time.perf_counter()

    action = {
        name: float(actions[idx]["action"][i]) for i, name in enumerate(dataset.features["action"]["names"])
    }
    left_follower.send_action(action)

    time.sleep(1.0 / dataset.fps - (time.perf_counter() - t0))

left_follower.disconnect()