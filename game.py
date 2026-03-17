import cv2
import numpy as np
from ultralytics import YOLO
import math
import random

MODEL_PATH   = "C:\\Users\\Nyle\\Documents\\ResumeProjects\\ComputerVisionBreakout\\runs\\pose\\train15\\weights\\best.pt"  
#MODEL_PATH = "C:\\Users\\Nyle\\Documents\\ResumeProjects\\ComputerVisionBreakout\\runs\\detect\\train\\weights\\best.pt"
CAMERA_INDEX = 0
WIN_W, WIN_H = 1280, 720

PADDLE_W, PADDLE_H = 160, 14
BALL_RADIUS = 12
BRICK_ROWS, BRICK_COLS = 4, 10
BRICK_H = 28
BRICK_MARGIN = 6
BRICK_TOP_OFFSET = 80          
BALL_SPEED_INIT = 8
BALL_SPEED_MAX  = 18
SPEED_INCREMENT = 0.3          

FONT = cv2.FONT_HERSHEY_SIMPLEX

C_PADDLE = (255, 200,  50)
C_BALL   = (255, 255, 255)
C_TEXT   = (255, 255, 255)
C_LIVES  = (50,  200, 255)
BRICK_COLOURS = [
    (50,  100, 255),
    (50,  200, 100),
    (255, 180,  50),
    (200,  50, 255),
]


def make_bricks():
    bricks = []
    total_w = WIN_W - 2 * BRICK_MARGIN
    bw = (total_w - (BRICK_COLS - 1) * BRICK_MARGIN) // BRICK_COLS
    for r in range(BRICK_ROWS):
        for c in range(BRICK_COLS):
            x1 = BRICK_MARGIN + c * (bw + BRICK_MARGIN)
            y1 = BRICK_TOP_OFFSET + r * (BRICK_H + BRICK_MARGIN)
            bricks.append([x1, y1, x1 + bw, y1 + BRICK_H, True])
    return bricks


def draw_bricks(frame, bricks):
    for i, (x1, y1, x2, y2, alive) in enumerate(bricks):
        if not alive:
            continue
        colour = BRICK_COLOURS[i // BRICK_COLS % len(BRICK_COLOURS)]
        cv2.rectangle(frame, (x1, y1), (x2, y2), colour, -1)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 1)


def draw_paddle(frame, px, py):
    cv2.rectangle(frame,
                  (px, py),
                  (px + PADDLE_W, py + PADDLE_H),
                  C_PADDLE, -1)
    cv2.rectangle(frame,
                  (px, py),
                  (px + PADDLE_W, py + PADDLE_H),
                  (255, 255, 255), 2)


def draw_ball(frame, bx, by):
    cv2.circle(frame, (int(bx), int(by)), BALL_RADIUS, C_BALL, -1)


def draw_hud(frame, score, lives, level):
    cv2.putText(frame, f"Score: {score}", (10, 30),
                FONT, 0.8, C_TEXT, 2)
    cv2.putText(frame, f"Level: {level}", (WIN_W // 2 - 50, 30),
                FONT, 0.8, C_TEXT, 2)
    for i in range(lives):
        cx = WIN_W - 30 - i * 30
        cv2.circle(frame, (cx, 22), 10, C_LIVES, -1)


def check_brick_collision(bx, by, vx, vy, bricks):
    points = 0
    for brick in bricks:
        x1, y1, x2, y2, alive = brick
        if not alive:
            continue
        if x1 - BALL_RADIUS < bx < x2 + BALL_RADIUS and \
           y1 - BALL_RADIUS < by < y2 + BALL_RADIUS:
            brick[4] = False
            points += 10
            overlap_left  = bx - x1
            overlap_right = x2 - bx
            overlap_top   = by - y1
            overlap_bot   = y2 - by
            min_ov = min(overlap_left, overlap_right, overlap_top, overlap_bot)
            if min_ov in (overlap_top, overlap_bot):
                vy = -vy
            else:
                vx = -vx
            break  
    return vx, vy, points


def overlay_dim(frame, alpha=0.55):
    """Darken the frame for menu/game-over screens."""
    overlay = np.zeros_like(frame)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)


def center_text(frame, text, y, scale=1.4, colour=C_TEXT, thickness=3):
    sz, _ = cv2.getTextSize(text, FONT, scale, thickness)
    x = (WIN_W - sz[0]) // 2
    cv2.putText(frame, text, (x, y), FONT, scale, colour, thickness)



class Game:
    def __init__(self):
        self.reset_full()

    def reset_full(self):
        self.score  = 0
        self.lives  = 3
        self.level  = 1
        self.state  = "start"   # start | playing | dead | win | gameover
        self._reset_ball_and_bricks()

    def _reset_ball_and_bricks(self):
        self.bricks  = make_bricks()
        speed = min(BALL_SPEED_INIT + (self.level - 1) * 1.5, BALL_SPEED_MAX)
        angle = math.radians(random.uniform(40, 140))
        self.ball_x  = float(WIN_W // 2)
        self.ball_y  = float(WIN_H - 120)
        self.ball_vx = speed * math.cos(angle)
        self.ball_vy = -abs(speed * math.sin(angle))
        self.paddle_x = WIN_W // 2 - PADDLE_W // 2
        self.paddle_y = WIN_H - 50
        self.speed    = speed

    def update(self, paddle_x_target):
        if self.state != "playing":
            return

        self.paddle_x += (paddle_x_target - self.paddle_x) * 0.25
        self.paddle_x = int(np.clip(self.paddle_x, 0, WIN_W - PADDLE_W))

        self.ball_x += self.ball_vx
        self.ball_y += self.ball_vy

        if self.ball_x - BALL_RADIUS <= 0:
            self.ball_x  = BALL_RADIUS
            self.ball_vx = abs(self.ball_vx)
        if self.ball_x + BALL_RADIUS >= WIN_W:
            self.ball_x  = WIN_W - BALL_RADIUS
            self.ball_vx = -abs(self.ball_vx)
        if self.ball_y - BALL_RADIUS <= 0:
            self.ball_y  = BALL_RADIUS
            self.ball_vy = abs(self.ball_vy)

        px, py = int(self.paddle_x), self.paddle_y
        if (py - BALL_RADIUS <= self.ball_y <= py + PADDLE_H and
                px <= self.ball_x <= px + PADDLE_W and
                self.ball_vy > 0):
            rel = (self.ball_x - px) / PADDLE_W  
            angle = math.radians(-150 + rel * 120) 
            self.ball_vy = self.speed * math.sin(angle)
            self.ball_vx = self.speed * math.cos(angle)

        self.ball_vx, self.ball_vy, pts = check_brick_collision(
            self.ball_x, self.ball_y, self.ball_vx, self.ball_vy, self.bricks)
        if pts:
            self.score += pts
            self.speed = min(self.speed + SPEED_INCREMENT, BALL_SPEED_MAX)

        if self.ball_y - BALL_RADIUS > WIN_H:
            self.lives -= 1
            if self.lives <= 0:
                self.state = "gameover"
            else:
                self.state = "dead"

        if all(not b[4] for b in self.bricks):
            self.level += 1
            self.state = "win"

    def draw(self, frame):
        draw_bricks(frame, self.bricks)
        draw_paddle(frame, int(self.paddle_x), self.paddle_y)
        draw_ball(frame, self.ball_x, self.ball_y)
        draw_hud(frame, self.score, self.lives, self.level)

        if self.state == "start":
            overlay_dim(frame)
            center_text(frame, "BREAKOUT", WIN_H // 2 - 60, scale=2.5)
            center_text(frame, "Move your hand to control the paddle",
                        WIN_H // 2 + 10, scale=0.8)
            center_text(frame, "Press SPACE to start", WIN_H // 2 + 60, scale=0.9)

        elif self.state == "dead":
            overlay_dim(frame)
            center_text(frame, f"Life lost!  {self.lives} remaining",
                        WIN_H // 2, scale=1.0, colour=(50, 50, 255))
            center_text(frame, "Press SPACE to continue", WIN_H // 2 + 50, scale=0.8)
            self.ball_x  = float(WIN_W // 2)
            self.ball_y  = float(WIN_H - 120)

        elif self.state == "win":
            overlay_dim(frame)
            center_text(frame, f"Level {self.level - 1} Clear!", WIN_H // 2 - 40, scale=1.8)
            center_text(frame, "Press SPACE for next level", WIN_H // 2 + 30, scale=0.9)

        elif self.state == "gameover":
            overlay_dim(frame)
            center_text(frame, "GAME OVER", WIN_H // 2 - 60, scale=2.5,
                        colour=(50, 50, 255))
            center_text(frame, f"Final Score: {self.score}", WIN_H // 2 + 20, scale=1.1)
            center_text(frame, "Press R to restart  |  Q to quit",
                        WIN_H // 2 + 80, scale=0.8)



def get_hand_x(results, frame_w):
    for r in results:
        if r.boxes is not None and len(r.boxes):
            box = r.boxes[0].xyxy[0].cpu().numpy()  # x1,y1,x2,y2
            return int((box[0] + box[2]) / 2)
    return None


def main():
    print("Loading model…")
    model = YOLO(MODEL_PATH)

    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  WIN_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, WIN_H)

    game = Game()
    paddle_target = WIN_W // 2 - PADDLE_W // 2

    print("Window ready. Press SPACE to start, Q to quit.")
    cv2.namedWindow("Breakout", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Breakout", WIN_W, WIN_H)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (WIN_W, WIN_H))
        frame = cv2.flip(frame, 1)   

        results = model.predict(frame, verbose=False, conf=0.45)
        hand_x  = get_hand_x(results, WIN_W)

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 100), 2)

        if hand_x is not None:
            paddle_target = hand_x - PADDLE_W // 2

        game.update(paddle_target)
        game.draw(frame)

        cv2.imshow("Breakout", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):
            if game.state in ("start", "dead", "win"):
                if game.state == "win":
                    game._reset_ball_and_bricks()
                game.state = "playing"
        elif key == ord('r') and game.state == "gameover":
            game.reset_full()

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
