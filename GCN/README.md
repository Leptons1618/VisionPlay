# Pose-Controlled Game

This project uses Graph Convolutional Networks (GCN) for pose detection to control a simple computer game through computer vision. It detects your body movements via webcam and translates them into game controls.

## Features

- Real-time pose detection using MediaPipe's pose estimation model
- Simple game where you collect targets by moving your body
- Webcam preview with pose landmarks visualization
- Score tracking

## How to Play

Control the green circle (player) by moving your body:
- **Tilt shoulders left/right**: Move the player left/right
- **Raise left hand above shoulder**: Move the player up
- **Lower left hand below shoulder**: Move the player down

The goal is to collect as many red targets as possible!

## Requirements

- Python 3.6+
- OpenCV
- MediaPipe
- PyGame
- NumPy

## Installation

1. Clone this repository or download the code
2. Install the required packages:

```bash
pip install opencv-python mediapipe pygame numpy
```

3. Run the game:

```bash
python pose_game_controller.py
```

## System Requirements

- Webcam
- Sufficient lighting for pose detection
- Enough space to move around

## Troubleshooting

- **No webcam detected**: Ensure your webcam is connected and not being used by another application
- **Poor pose detection**: Try improving lighting or moving to a location with better contrast between you and the background
- **Game runs slowly**: Try closing other applications to free up system resources

## Extending the Project

Some ideas to enhance this project:
- Add more complex game mechanics
- Implement different gestures for special actions
- Create multiple game modes
- Add options to calibrate pose detection for different users
- Implement multiplayer functionality

## How It Works

The system uses MediaPipe's pose detection model, which implements Graph Convolutional Networks (GCNs) for human pose estimation. The pose landmarks are processed to detect specific body positions, which are then mapped to game controls.

The main components are:
1. **Pose Detection**: Uses MediaPipe to detect and track body landmarks
2. **Control Mapping**: Translates body positions into game controls
3. **Game Logic**: A simple game controlled by the detected poses
4. **Visualization**: Displays both the webcam feed with pose overlay and the game screen