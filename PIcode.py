import cv2
import serial
import time

# ─── Setup Serial Communication with Arduino ──────────────────────────────────
# /dev/ttyUSB0 is the usual port when Arduino connects via USB
# Check yours with: ls /dev/tty* in the terminal
arduino = serial.Serial('/dev/ttyUSB0', 9600, timeout=1)
time.sleep(2)   # Wait for Arduino to reset after serial connection

# ─── Load Haar Cascades ───────────────────────────────────────────────────────
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# ─── Pi Camera via VideoCapture ───────────────────────────────────────────────
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

FRAME_W  = 640
FRAME_CX = FRAME_W // 2      # 320 — center of frame
DEADZONE = 80                 # pixels: face must be this far off-center to trigger turn

last_command = ''             # Avoid sending same command repeatedly

def send_command(cmd):
    global last_command
    if cmd != last_command:
        arduino.write(cmd.encode())   # Send single character: F, S, L, R
        print(f"Sent: {cmd}")
        last_command = cmd

print("Robot Dog Vision Active | q to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray  = cv2.equalizeHist(gray)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(80, 80))

    if len(faces) == 0:
        # No face detected → stop and wait
        send_command('S')

    else:
        # Use the largest detected face (most likely the main person)
        (x, y, w, h) = max(faces, key=lambda f: f[2] * f[3])
        face_cx = x + w // 2    # Center x of the detected face

        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # ── Decision Logic ────────────────────────────────────────────────────
        offset = face_cx - FRAME_CX   # Negative = face is left, Positive = face is right

        if abs(offset) < DEADZONE:
            # Face is roughly centered → walk forward
            send_command('F')
            cv2.putText(frame, 'FORWARD', (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        elif offset < 0:
            # Face is on left side → turn head/body left
            send_command('L')
            cv2.putText(frame, 'LOOK LEFT', (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)
        else:
            # Face is on right side → turn head/body right
            send_command('R')
            cv2.putText(frame, 'LOOK RIGHT', (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)

        # Show offset value on screen for debugging
        cv2.putText(frame, f'Offset: {offset}px', (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    cv2.imshow('Robot Dog Vision', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        send_command('S')   # Stop before quitting
        break

cap.release()
arduino.close()
cv2.destroyAllWindows()