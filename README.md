# VisionPlay - Pose-Based Game Controller

VisionPlay is a computer vision-based game controller that allows you to control games using body movements. It uses MediaPipe for pose detection and converts physical movements into keyboard inputs.

## Features

- Real-time pose detection
- Configurable controls via JSON
- Support for multiple gestures:
  - Right/Left punch
  - Right/Left kick
  - Special moves
  - Blocking stance
- Performance monitoring and logging
- Visual feedback with landmark visualization
- FPS counter display

## Requirements

- Python 3.8+
- Webcam
- Required packages (install via `pip install -r requirements.txt`):
  - mediapipe
  - opencv-python
  - numpy
  - pyautogui
  - and more (see requirements.txt)

## Installation

1.Clone the repository:

```bash
git clone https://github.com/yourusername/VisionPlay.git
cd VisionPlay
```

2.Install dependencies:

```bash
pip install -r requirements.txt
```

3.Configure your controls in `controller_config.json` (optional)

## Usage

Run the pose controller:

```bash
python pose_controller.py
```

1. Stand in front of your webcam (ensure good lighting)
2. Use the following gestures:
   - Extend right arm for right punch
   - Extend left arm for left punch
   - Raise right arm for right kick
   - Raise left arm for left kick
   - Extend both arms for special move
   - Raise both arms for blocking
3. Press ESC to exit

## Configuration

You can customize controls and thresholds in `controller_config.json`:

```json
{
    "controls": {
        "right_punch": "p",
        "left_punch": "o",
        "right_kick": "k",
        "left_kick": "l",
        "special_move": "s",
        "block": "b"
    }
}
```

## Logging

The application logs events to `pose_controller.log`, including:

- Initialization status
- Error messages
- Performance metrics
- Detected gestures
- Key press/release events

## Troubleshooting

1. Camera not detected:
   - Ensure webcam is connected
   - Try changing camera index in code (default is 0)

2. Poor performance:
   - Ensure good lighting
   - Reduce screen resolution in config
   - Check CPU/GPU usage

3. Gestures not detecting:
   - Adjust visibility_threshold in config
   - Ensure you're in camera frame
   - Check pose landmark visibility

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- MediaPipe for pose detection
- OpenCV for image processing
- PyAutoGUI for keyboard control
