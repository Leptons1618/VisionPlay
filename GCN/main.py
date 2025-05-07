import cv2
import mediapipe as mp
import pygame
import sys
import math
import numpy as np

class PoseGameController:
    def __init__(self):
        # Initialize MediaPipe Pose
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5)
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Initialize webcam
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("Error: Could not open webcam.")
            sys.exit()
        
        # Get webcam dimensions
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Initialize pygame for the game
        pygame.init()
        self.screen_width = 800
        self.screen_height = 600
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption("Pose-Controlled Game")
        self.clock = pygame.time.Clock()
        
        # Game variables
        self.player_x = self.screen_width // 2
        self.player_y = self.screen_height // 2
        self.player_size = 50
        self.player_speed = 10
        self.game_score = 0
        
        # Target variables
        self.target_size = 30
        self.spawn_new_target()
        
        # Font for displaying score
        self.font = pygame.font.Font(None, 36)
        
        # Control flags
        self.move_left = False
        self.move_right = False
        self.move_up = False
        self.move_down = False
    
    def spawn_new_target(self):
        """Create a new target at a random position"""
        margin = 50  # Keep away from edges
        self.target_x = np.random.randint(margin, self.screen_width - margin)
        self.target_y = np.random.randint(margin, self.screen_height - margin)
    
    def process_pose(self, landmarks):
        """Process pose landmarks to control the game"""
        if not landmarks:
            return
        
        # Get positions of key points
        # Left and right shoulders (for left/right movement)
        left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        
        # Shoulders tilt for left/right movement
        shoulder_diff = left_shoulder.y - right_shoulder.y
        tilt_threshold = 0.1
        
        self.move_left = shoulder_diff < -tilt_threshold
        self.move_right = shoulder_diff > tilt_threshold
        
        # Use hands up/down for vertical movement
        left_wrist = landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value]
        left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        
        # If left hand is above shoulder, move up
        self.move_up = left_wrist.y < left_shoulder.y - 0.1
        
        # If left hand is below shoulder, move down
        self.move_down = left_wrist.y > left_shoulder.y + 0.2
    
    def update_game_state(self):
        """Update the game state based on pose control flags"""
        if self.move_left and self.player_x > self.player_size:
            self.player_x -= self.player_speed
        if self.move_right and self.player_x < self.screen_width - self.player_size:
            self.player_x += self.player_speed
        if self.move_up and self.player_y > self.player_size:
            self.player_y -= self.player_speed
        if self.move_down and self.player_y < self.screen_height - self.player_size:
            self.player_y += self.player_speed
        
        # Check for collision with target
        distance = math.sqrt((self.player_x - self.target_x)**2 + (self.player_y - self.target_y)**2)
        if distance < (self.player_size + self.target_size) / 2:
            self.game_score += 1
            self.spawn_new_target()
    
    def draw_game(self):
        """Draw the game elements on screen"""
        # Clear screen
        self.screen.fill((0, 0, 0))
        
        # Draw player
        pygame.draw.circle(self.screen, (0, 255, 0), (self.player_x, self.player_y), self.player_size)
        
        # Draw target
        pygame.draw.circle(self.screen, (255, 0, 0), (self.target_x, self.target_y), self.target_size)
        
        # Display score
        score_text = self.font.render(f"Score: {self.game_score}", True, (255, 255, 255))
        self.screen.blit(score_text, (20, 20))
        
        # Display controls
        controls_text = self.font.render("Tilt shoulders: left/right, Raise hand: up, Lower hand: down", True, (200, 200, 200))
        self.screen.blit(controls_text, (20, self.screen_height - 30))
        
        # Update the display
        pygame.display.flip()
    
    def display_webcam_preview(self, image):
        """Display webcam preview with pose landmarks"""
        # Resize image for display (keeping aspect ratio)
        preview_width = 320
        preview_height = int(preview_width * (self.height / self.width))
        preview = cv2.resize(image, (preview_width, preview_height))
        
        # Display the image in a separate window
        cv2.imshow('Pose Detection', preview)
    
    def run(self):
        """Main game loop"""
        running = True
        while running:
            # Process pygame events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
            
            # Read from webcam
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to capture image")
                break
            
            # Flip the image horizontally for a mirror effect
            frame = cv2.flip(frame, 1)
            
            # Convert the BGR image to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process the image and get pose landmarks
            results = self.pose.process(rgb_frame)
            
            # Draw pose landmarks on the image
            if results.pose_landmarks:
                self.mp_drawing.draw_landmarks(
                    frame, 
                    results.pose_landmarks,
                    self.mp_pose.POSE_CONNECTIONS)
                
                # Process pose for game control
                self.process_pose(results.pose_landmarks.landmark)
            
            # Display webcam feed with landmarks
            self.display_webcam_preview(frame)
            
            # Update game state based on pose controls
            self.update_game_state()
            
            # Draw game
            self.draw_game()
            
            # Cap the frame rate
            self.clock.tick(60)
            
            # Check for quit via OpenCV window
            if cv2.waitKey(1) & 0xFF == ord('q'):
                running = False
        
        # Clean up
        self.cap.release()
        cv2.destroyAllWindows()
        pygame.quit()

if __name__ == "__main__":
    game = PoseGameController()
    game.run()