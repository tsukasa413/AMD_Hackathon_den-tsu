from lerobot.robots.so101_follower import SO101FollowerConfig, SO101Follower  


def return_watching_home():
    """left_follower を watching ホームポジションに戻す"""
    left_follower_config = SO101FollowerConfig(  
        port="/dev/ttyACM2",  
        id="den_follower_arm"  
    )  

    left_follower = SO101Follower(left_follower_config)  
    left_follower.connect()  
    
    watching = {  
        "shoulder_pan.pos": -70,  
        "shoulder_lift.pos": -50,  
        "elbow_flex.pos": 0,  
        "wrist_flex.pos": 30.0,  
        "wrist_roll.pos": 0.0,  
        "gripper.pos": 20  
    }  
    
    left_follower.send_action(watching)
    left_follower.disconnect()


def return_working_home():
    """left_follower を working ホームポジションに戻す"""
    left_follower_config = SO101FollowerConfig(  
        port="/dev/ttyACM2",  
        id="den_follower_arm"  
    )  

    left_follower = SO101Follower(left_follower_config)  
    left_follower.connect()  
    
    working = {  
        "shoulder_pan.pos": 0,  
        "shoulder_lift.pos": -70,  
        "elbow_flex.pos": 60,  
        "wrist_flex.pos": 30.0,  
        "wrist_roll.pos": 0.0,  
        "gripper.pos": 20  
    }  
    
    left_follower.send_action(working)
    left_follower.disconnect()
