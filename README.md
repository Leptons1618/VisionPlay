# VisionPlay - Gesture-Based Game Controller

A Python-based gesture control system that converts body movements into keyboard inputs for gaming, using computer vision and pose detection.

## Features

- Real-time pose detection using MediaPipe
- Configurable controls via JSON
- Support for multiple gestures:
  - Right/Left punch
  - Right/Left kick
  - Special moves
  - Blocking stance
- Performance monitoring and logging
- Visual feedback with pose landmarks
- FPS counter

## Requirements

- Python 3.8+
- Webcam
- Required packages listed in `requirements.txt`

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/VisionPlay.git
cd VisionPlay
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Configuration

The `controller_config.json` file allows you to customize:
- Control mappings (keyboard keys)
- Detection thresholds
- Display settings

Default controls:
- `p` - Punch
- `k` - Kick
- `s` - Special move
- `b` - Block
- `o` - Left punch
- `l` - Left kick

## Usage

1. Start the controller:
```bash
python pose_controller.py
```

2. Position yourself in front of the camera:
- Ensure good lighting
- Keep your full body visible
- Maintain ~2 meters distance

3. Available
