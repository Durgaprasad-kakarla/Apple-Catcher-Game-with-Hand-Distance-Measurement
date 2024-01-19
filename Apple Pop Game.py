import cv2
import cvzone
import mediapipe as mp
import time
import HandtrackingModule as htm
import math
import numpy as np
import random

cap = cv2.VideoCapture(0)
detector = htm.HandDetector()
score = 0
x = [300, 245, 200, 170, 145, 130, 112, 103, 93, 87, 80, 75, 70, 67, 62, 59, 57]
y = [20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]
coff = np.polyfit(x, y, 2)
cx, cy = 250, 250
color = (172, 145, 255)
counter = 0
totTime = 20
timeStart = time.time()

apple = cv2.imread('apple1.png', cv2.IMREAD_UNCHANGED)
original_apple = apple.copy()

# Corrected balloon size to match roi size
apple= cv2.resize(apple, (150, 120))
new_apple=apple.copy()



while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    img = cv2.resize(img, (1100, 720))

    if time.time() - timeStart < totTime:
        hands, img = detector.findHands(img, draw=False)
        if hands:
            lmList = hands[0]['lmList']
            x, y, w, h = hands[0]['bbox']
            x1, y1 = lmList[5][:2]
            x2, y2 = lmList[17][:2]
            dist = int(math.sqrt((y1 - y2) ** 2 + (x2 - x1) ** 2))
            A, B, C = coff
            distCM = A * dist ** 2 + B * dist + C
            if distCM < 40:
                if x < cx < x + w and y < cy < y + h:
                    counter = 1
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 255), 3)
            cvzone.putTextRect(img, 'Dist:' + str(int(distCM)) + 'cm', (x + 40, y), 2)

        if counter:
            counter += 1
            apple[:, :, :3] = [0, 255, 0]
            if counter == 3:
                cx = random.randint(100, 900)
                cy = random.randint(150, 600)
                apple[:, :, :3] = new_apple[:, :, :3]  # Reset color to original
                counter = 0
                score += 1
        print(cx, cy)

        # Get the region of interest for placing the balloon
        roi = img[cy:cy + apple.shape[0], cx:cx + apple.shape[1]]

        # Resize the balloon to match the roi size (explicit rounding to integers)
        apple_resized = cv2.resize(apple, (int(roi.shape[1]), int(roi.shape[0])))

        # Create a mask from the alpha channel of the resized balloon
        mask = apple_resized[:, :, 3]

        # Invert the mask to use it as an alpha channel
        mask = ~mask

        # Resize the inverted mask to match the balloon size
        mask = cv2.resize(mask.astype(np.uint8), (apple.shape[1], apple.shape[0]))

        alpha = 1.0 - mask.astype(float) / 255.0

        # Split the balloon and alpha channel
        apple_rgb = apple_resized[:, :, :3]

        # Merge the RGB channels with the inverted alpha channel
        result = cv2.merge([roi[:, :, 0] * (1.0 - alpha) + apple_rgb[:, :, 0] * alpha,
                            roi[:, :, 1] * (1.0 - alpha) + apple_rgb[:, :, 1] * alpha,
                            roi[:, :, 2] * (1.0 - alpha) + apple_rgb[:, :, 2] * alpha])

        # Place the result back into the original image
        img[cy:cy + apple_resized.shape[0], cx:cx + apple_resized.shape[1]] = result

        cvzone.putTextRect(img, 'Time:' + str(int(totTime - (time.time() - timeStart))), (950, 70), scale=2, offset=20,colorT=(0,0,0),colorR=(172, 145, 255))
        cvzone.putTextRect(img, 'Score:' + str(score).zfill(2), (20, 70), scale=2, offset=20,colorR=(172, 145, 255),colorT=(0,0,0))
    else:
        cvzone.putTextRect(img, 'Game Over', (350, 400), scale=5, offset=30, thickness=7,colorT=(0,0,0),colorR=(0,255,255))
        cvzone.putTextRect(img, "Your Score: " + str(score), (400, 500), scale=3, offset=20,colorT=(0,0,0),colorR=(0,255,255))
        cvzone.putTextRect(img, "Press P to restart", (420, 575), scale=2, offset=10,colorT=(0,0,0),colorR=(0,255,255))
    cv2.imshow("Game and Hand Measurement", img)
    key = cv2.waitKey(1)
    if key == ord('p'):
        timeStart = time.time()
        score = 0
