import mediapipe as mp
import numpy as np
import cv2
import pyautogui
import json
import logging
import time
import signal
import sys
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pose_controller.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class GestureConfig:
    """Configuration for gesture detection thresholds"""
    visibility_threshold: float = 0.7
    punch_threshold: float = 110
    screen_width: int = 640
    screen_height: int = 480

class PoseBasedController:
    def __init__(self, config_file: str = "controller_config.json"):
        """Initialize the controller with optional config file"""
        logger.info("Initializing PoseBasedController")
        try:
            # Setup signal handler
            signal.signal(signal.SIGINT, self.signal_handler)
            self.is_running = True

            self.pose = mp.solutions.pose
            self.drawing = mp.solutions.drawing_utils
            self.obj = self.pose.Pose(
                min_detection_confidence=0.6,
                min_tracking_confidence=0.6
            )
            
            # Load configuration
            self.config = GestureConfig()
            self.key_states = {}
            self.load_config(config_file)
            
            # Initialize camera
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                raise RuntimeError("Failed to open camera")
                
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.screen_width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.screen_height)
            logger.info("Camera initialized successfully")
            
            # Performance monitoring
            self.frame_count = 0
            self.start_time = time.time()

            # Add overlay text settings
            self.overlay_font = cv2.FONT_HERSHEY_SIMPLEX
            self.text_color = (0, 255, 0)  # Green
            self.text_color_warning = (0, 165, 255)  # Orange
            self.line_spacing = 30
            
        except Exception as e:
            logger.error(f"Initialization failed: {str(e)}")
            raise

    def signal_handler(self, signum, frame):
        """Handle interrupt signals"""
        logger.info("Interrupt received, shutting down...")
        self.is_running = False

    def load_config(self, config_file: str) -> None:
        """Load control configuration from JSON file"""
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
                self.control_map = config.get('controls', {
                    'right_punch': 'p',
                    'left_punch': 'o',
                    'right_kick': 'k',
                    'left_kick': 'l',
                    'special_move': 's',
                    'block': 'b'
                })
                self.key_states = {key: False for key in self.control_map.values()}
                logger.info(f"Configuration loaded from {config_file}")
        except FileNotFoundError:
            logger.warning(f"Config file {config_file} not found, using defaults")
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in config file: {str(e)}")
            raise

    def handle_key_state(self, action: str, should_press: bool) -> None:
        """Handle keyboard press/release with state tracking"""
        key = self.control_map.get(action)
        if not key:
            return

        try:
            if should_press and not self.key_states.get(key, False):
                pyautogui.keyDown(key)
                self.key_states[key] = True
                logger.debug(f"Key pressed: {key} for action {action}")
            elif not should_press and self.key_states.get(key, False):
                pyautogui.keyUp(key)
                self.key_states[key] = False
                logger.debug(f"Key released: {key} for action {action}")
        except Exception as e:
            logger.error(f"Error handling key state for {action}: {str(e)}")

    def detect_gestures(self, landmarks) -> Dict[str, bool]:
        """Detect various gestures and return their states"""
        try:
            gestures = {}
            
            # Right side movements
            if landmarks.landmark[16].visibility > self.config.visibility_threshold:
                right_punch_distance = abs(
                    landmarks.landmark[16].x * self.config.screen_width -
                    landmarks.landmark[12].x * self.config.screen_width
                )
                gestures['right_punch'] = right_punch_distance > self.config.punch_threshold
                
                right_kick = (landmarks.landmark[16].y * self.config.screen_height <
                             landmarks.landmark[10].y * self.config.screen_height)
                gestures['right_kick'] = right_kick

            # Left side movements
            if landmarks.landmark[15].visibility > self.config.visibility_threshold:
                left_punch_distance = abs(
                    landmarks.landmark[15].x * self.config.screen_width -
                    landmarks.landmark[11].x * self.config.screen_width
                )
                gestures['left_punch'] = left_punch_distance > self.config.punch_threshold
                
                left_kick = (landmarks.landmark[15].y * self.config.screen_height <
                            landmarks.landmark[9].y * self.config.screen_height)
                gestures['left_kick'] = left_kick

            # Special moves and blocks
            if (landmarks.landmark[15].visibility > self.config.visibility_threshold and
                landmarks.landmark[16].visibility > self.config.visibility_threshold):
                
                # Detect blocking stance
                blocking = (abs(landmarks.landmark[15].x - landmarks.landmark[16].x) < 0.1 and
                           landmarks.landmark[15].y * self.config.screen_height < landmarks.landmark[9].y * self.config.screen_height)
                gestures['block'] = blocking
                
                # Detect special move
                special_move = (landmarks.landmark[15].x < landmarks.landmark[11].x and
                              landmarks.landmark[16].x > landmarks.landmark[12].x)
                gestures['special_move'] = special_move

            # Log detected gestures
            active_gestures = [g for g, state in gestures.items() if state]
            if active_gestures:
                logger.debug(f"Active gestures: {', '.join(active_gestures)}")
                
            return gestures
        except Exception as e:
            logger.error(f"Error detecting gestures: {str(e)}")
            return {}

    def add_overlay_text(self, frame: np.ndarray, gestures: Dict[str, bool]) -> np.ndarray:
        """Add overlay text to frame"""
        # Add title and FPS
        cv2.putText(frame, "VisionPlay Controller", (10, 30),
                    self.overlay_font, 0.7, self.text_color, 2)
        
        # Calculate and display FPS
        elapsed_time = time.time() - self.start_time
        fps = self.frame_count / elapsed_time if elapsed_time > 0 else 0
        cv2.putText(frame, f"FPS: {int(fps)}", (10, 60),
                    self.overlay_font, 0.6, self.text_color, 1)

        # Display controls info
        controls_text = "Controls: ESC - Exit | CTRL+C - Quit"
        cv2.putText(frame, controls_text, (10, frame.shape[0] - 20),
                    self.overlay_font, 0.5, self.text_color, 1)

        # Display active gestures
        y_pos = 90
        if any(gestures.values()):
            cv2.putText(frame, "Active Gestures:", (10, y_pos),
                        self.overlay_font, 0.6, self.text_color, 1)
            y_pos += self.line_spacing
            
            for action, is_active in gestures.items():
                if is_active:
                    cv2.putText(frame, f"- {action.replace('_', ' ').title()}",
                              (20, y_pos), self.overlay_font, 0.6, self.text_color, 1)
                    y_pos += self.line_spacing
        
        # Add visibility warning if needed
        if not any(gestures.values()):
            warning_text = "No poses detected - Please stand in frame"
            cv2.putText(frame, warning_text, (frame.shape[1]//4, frame.shape[0]//2),
                        self.overlay_font, 0.7, self.text_color_warning, 2)
        
        return frame

    def run(self) -> None:
        """Main loop for the controller"""
        logger.info("Starting pose controller")
        try:
            while self.is_running:
                self.frame_count += 1
                
                ret, frame = self.cap.read()
                if not ret:
                    logger.error("Failed to capture frame")
                    break

                # Process frame
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.obj.process(rgb_frame)
                
                gestures = {}
                if results.pose_landmarks:
                    self.drawing.draw_landmarks(frame, results.pose_landmarks, 
                                             self.pose.POSE_CONNECTIONS)
                    gestures = self.detect_gestures(results.pose_landmarks)
                    for action, is_active in gestures.items():
                        self.handle_key_state(action, is_active)

                # Add overlay text and flip frame
                frame = self.add_overlay_text(frame, gestures)
                frame = cv2.flip(frame, 1)
                cv2.imshow("VisionPlay Controller", frame)

                if cv2.waitKey(1) == 27:  # ESC key
                    logger.info("ESC pressed, stopping controller")
                    break

        except Exception as e:
            logger.error(f"Runtime error: {str(e)}")
        finally:
            self.cleanup()

    def cleanup(self) -> None:
        """Release resources and reset key states"""
        logger.info("Cleaning up resources")
        try:
            # Release all pressed keys
            for key, is_pressed in self.key_states.items():
                if is_pressed:
                    pyautogui.keyUp(key)
                    logger.debug(f"Released key: {key}")
            
            self.cap.release()
            cv2.destroyAllWindows()
            logger.info("Cleanup completed successfully")
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")

if __name__ == "__main__":
    try:
        controller = PoseBasedController()
        controller.run()
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received, shutting down...")
    except Exception as e:
        logger.critical(f"Fatal error: {str(e)}")
