# Dream and play

Hackaton for AI on Edge devices.


Concept is pretty simple: use modern advances in computer vision (pose estimation, gesture recognition) to use your webcam as a gaming controller.

[Mediapipe](https://github.com/google-ai-edge/mediapipe/blob/master/docs/solutions/pose.md) provides open source solution to run pose estimation on various edge devices including laptops and smartphones. 

In addition to a custom lightweight neural network used to process extracted poses, we're able to infer a few actions (jump, crouch, raise arms). This task is definitely not trivial, especially detecting jumps is very challenging. We provide a script to [capture a dataset](/scripts/capture_gesture_data.py) for supervised learning - simply perform the same motion in front of your webcam during a minute (feel free to experiment with mixed actions during capture to add more transitions).


### 💡 Team

- Balthazar Neveu
- Achraff Adjileye 
- Victor Guichard
- Louis Martinez