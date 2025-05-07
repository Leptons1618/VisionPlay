import cv2
import pygame
import numpy as np
from mmpose.apis import MMPoseInferencer
import asyncio
import platform

# Initialize camera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise Exception("Cannot open camera")

# Initialize MMPose inferencer
inferencer = MMPoseInferencer(
    pose2d='models/vitpose/vitpose-b-coco.yml',
    pose2d_weights='models/vitpose/vitpose-b-coco.pth'
)

# Initialize Pygame
pygame.init()
screen = pygame.display.set_mode((800, 600))
pygame.display.set_caption("Pose-Controlled Game")
clock = pygame.time.Clock()

# Character class
class Character:
    def __init__(self):
        self.x = 400
        self.y = 500
        self.width = 50
        self.height = 50
        self.velocity_y = 0
        self.jumping = False

    def move_left(self):
        self.x -= 5
        if self.x < 0:
            self.x = 0

    def move_right(self):
        self.x += 5
        if self.x > 800 - self.width:
            self.x = 800 - self.width

    def jump(self):
        if not self.jumping:
            self.velocity_y = -15
            self.jumping = True

    def update(self):
        if self.jumping:
            self.y += self.velocity_y
            self.velocity_y += 1
            if self.y >= 500:
                self.y = 500
                self.jumping = False
                self.velocity_y = 0

    def draw(self, screen):
        pygame.draw.rect(screen, (255, 0, 0), (self.x, self.y, self.width, self.height))

# Pose detection functions
def is_raising_arms(keypoints):
    if len(keypoints) < 17:
        return False
    left_wrist = keypoints[9]
    right_wrist = keypoints[10]
    left_shoulder = keypoints[5]
    right_shoulder = keypoints[6]
    threshold = 50  # Pixels
    if (left_wrist[2] > 0.5 and right_wrist[2] > 0.5 and
        left_shoulder[2] > 0.5 and right_shoulder[2] > 0.5):
        if (left_wrist[1] < left_shoulder[1] - threshold and
            right_wrist[1] < right_shoulder[1] - threshold):
            return True
    return False

def is_leaning_left(keypoints):
    if len(keypoints) < 17:
        return False
    nose = keypoints[0]
    left_hip = keypoints[11]
    right_hip = keypoints[12]
    threshold = 20  # Pixels
    if nose[2] > 0.5 and left_hip[2] > 0.5 and right_hip[2] > 0.5:
        hip_center_x = (left_hip[0] + right_hip[0]) / 2
        if nose[0] < hip_center_x - threshold:
            return True
    return False

def is_leaning_right(keypoints):
    if len(keypoints) < 17:
        return False
    nose = keypoints[0]
    left_hip = keypoints[11]
    right_hip = keypoints[12]
    threshold = 20  # Pixels
    if nose[2] > 0.5 and left_hip[2] > 0.5 and right_hip[2] > 0.5:
        hip_center_x = (left_hip[0] + right_hip[0]) / 2
        if nose[0] > hip_center_x + threshold:
            return True
    return False

# Initialize character
character = Character()

async def main():
    FPS = 30
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Read frame
        ret, frame = cap.read()
        if not ret:
            break

        # Run pose estimation
        results = inferencer(frame, return_vis=False)
        keypoints = []
        if results.get('predictions') and len(results['predictions']) > 0:
            keypoints = results['predictions'][0][0]['keypoints']  # First person

        # Detect poses and control character
        if is_raising_arms(keypoints):
            character.jump()
        if is_leaning_left(keypoints):
            character.move_left()
        if is_leaning_right(keypoints):
            character.move_right()

        # Update character
        character.update()

        # Draw
        screen.fill((0, 0, 0))
        character.draw(screen)
        pygame.display.flip()

        # Control frame rate
        await asyncio.sleep(1.0 / FPS)

if platform.system() == "Emscripten":
    asyncio.ensure_future(main())
else:
    if __name__ == "__main__":
        asyncio.run(main())

# Cleanup
cap.release()
pygame.quit()
cv2.destroyAllWindows()