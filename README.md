# Computer Vision Breakout -- AI-Controlled Gameplay

Traditional gaming often relies on hardware-heavy inputs, such as from 
keyboards or controllers. This project explores a software based approach
to augmented reality, using bounding box estimation to create a low-latence,
webcam-based, motion control system. The goal is to provide a more inclusive
and intuitive way to interact with classic 'Breakout' gameplay through high-speed
visual perception.

The system processes an incoming webcam feed at 30 FPS, detecting relevant visual
signals such as hand position or keypoints for hand tracking, which is used to 
control the paddle in real time with ease.

Low-latency and high-speed visual tracking were both essential in making the motion 
controls feel fluid and enjoyable, so I tested and compared various models and 
detection solutions for mapping the paddle to the on-screen hand.

<p align="center">
  <img src="breakoutDemo.gif" alt="Live YOLO Hand Sign Detection Demo" width="600">
</p>


------------------------------------------------------------------------

## Project Objectives

-   Explore real-time computer vision interaction systems
-   Evaluate different AI-based object and pose detection models
-   Implement a perception → decision → action loop
-   Understand the trade-offs between accuracy, latency, and robustness
    in real-time AI applications

------------------------------------------------------------------------

## System Architecture

The system follows a typical AI perception pipeline:

1.  **Video Capture**
    -   Webcam frames @ 30 FPS are captured in real time using OpenCV.
2.  **Visual Detection**
    -   Multiple computer vision models are used to identify hand motion
        or pose landmarks.
3.  **Feature Extraction**
    -   Detected landmarks or objects are converted into coordinates
        representing player motion.
4.  **Decision Logic**
    -   Paddle position is calculated based on detected hand movement. The paddle slides 
        to the average center of the hand's bounding box instead of jumping to prevent "jittery" motion.
5.  **Game Control**
    -   The system sends automated keyboard commands to control the
        game.

------------------------------------------------------------------------

## Models and Frameworks Evaluated

To determine the most effective detection approach for real-time
gameplay control, multiple computer vision frameworks and model
configurations were tested.

### YOLO Object Detection

Generic YOLO object detection models were evaluated for detecting
relevant visual elements within the gameplay environment.

To compare performance and latency tradeoffs, multiple YOLO model sizes
were tested:

-   **YOLO26 Nano**
-   **YOLO26 Small**
-   **YOLO26 Large**

Smaller models provided faster inference suitable for real-time
processing, while larger models provided improved detection accuracy but
required more computational resources.

### YOLO Pose Estimation

YOLO Pose models were also evaluated to detect body keypoints and
estimate hand motion. Pose-based detection proved to be highly reliable
for tracking hand movement used to control the paddle.

### MediaPipe Hand Tracking

MediaPipe provides pretrained models for real-time hand tracking and
pose estimation. Because these models are already pretrained and
optimized for landmark detection, traditional object detection metrics
such as mAP were not calculated in this experiment. Instead, MediaPipe
performance was evaluated qualitatively based on tracking stability and
responsiveness.

------------------------------------------------------------------------
## Model Performance Comparison

| Model                        | mAP@50 | mAP@50–95 | Notes |
|------------------------------|-------:|----------:|------|
| YOLO (Generic Object Detection) | 0.70  | 0.40 | Tested using Nano, Small, and Large model variants |
| YOLO Pose                    | 0.90  | 0.92 | Provided strong pose landmark detection for motion tracking |
| MediaPipe                    | N/A   | N/A  | Pretrained hand tracking model evaluated qualitatively |

### Metric Definitions

- **mAP@50** – Mean Average Precision at an Intersection over Union (IoU) threshold of 0.5  
- **mAP@50–95** – Mean Average Precision averaged across IoU thresholds from 0.5 to 0.95



------------------------------------------------------------------------

## Key Features

-   Real-time webcam-based interaction
-   AI-driven paddle movement using visual input
-   Multiple computer vision models tested for detection reliability
-   Automated game control using computer vision outputs
-   Low-latency perception--action loop

------------------------------------------------------------------------

## Technologies Used

**Programming Language** - Python

**Computer Vision** - OpenCV - MediaPipe - YOLO - YOLO Pose

**Development Tools** - Git - VS Code

------------------------------------------------------------------------

## Key Learning Outcomes

Through this project, I explored:

-   Real-time computer vision pipelines
-   Pose estimation and object detection techniques
-   Model experimentation and evaluation
-   Trade-offs between accuracy, latency, and robustness
-   Building AI-driven interactive systems

------------------------------------------------------------------------

## Future Improvements

Possible extensions include:

-   Training a custom model for improved hand detection
-   Implementing trajectory prediction for smoother paddle control
-   GPU acceleration for faster inference
-   Adding reinforcement learning for autonomous gameplay

------------------------------------------------------------------------

## Demo

<p align="center">
  <img src="Screen-Recording.gif" alt="Live YOLO Hand Sign Detection Demo" width="600">
</p>