from lerobot.robots.so101_follower import SO101FollowerConfig, SO101Follower  
  
left_follower_config = SO101FollowerConfig(  
    port="/dev/ttyACM0",  
    id="den_follower_arm"  
)  

left_follower = SO101Follower(robot_config)  
left_follower.connect()  
  
# 目標関節角度を指定（例: ホームポジション）  
watching = {  
    "shoulder_pan.pos": -70,  
    "shoulder_lift.pos": -50,  
    "elbow_flex.pos": 0,  
    "wrist_flex.pos": 30.0,  
    "wrist_roll.pos": 0.0,  
    "gripper.pos": 20  
}  

working = {  
    "shoulder_pan.pos": 0,  
    "shoulder_lift.pos": -70,  
    "elbow_flex.pos": 60,  
    "wrist_flex.pos": 30.0,  
    "wrist_roll.pos": 0.0,  
    "gripper.pos": 20  
}  
  
# アクションを送信  
left_follower.send_action(working)