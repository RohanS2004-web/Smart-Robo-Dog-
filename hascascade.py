import cv2

#
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade  = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml') 

# Check if classifiers loaded correctly
if face_cascade.empty():
    print("Error: Face cascade not loaded!")
    exit()
if eye_cascade.empty():
    print("Error: Eye cascade not loaded!")
    exit()

# ─── Open Webcam ─────────────────────────────────────────────────────────────
cap = cv2.VideoCapture(0)   # 0 = default webcam, change to 1 for external camera

if not cap.isOpened():
    print("Error: Cannot open webcam!")
    exit()

print("Press 'q' to quit | Press 's' to save screenshot")

# ─── Main Loop ────────────────────────────────────────────────────────────────
while True:
    ret, frame = cap.read()   # Read one frame
    if not ret:
        print("Error: Failed to read frame!")
        break

    # Convert to grayscale (cascades work on grayscale only)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Optional: equalize histogram to improve detection in low light
    gray = cv2.equalizeHist(gray)

    # ── Step 1: Detect Faces ──────────────────────────────────────────────────
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor  = 1.1,    # Scale image by 10% each pass
        minNeighbors = 5,       # Higher = fewer false positives
        minSize      = (60, 60) # Ignore tiny detections
    )

    face_count = len(faces)   # Count detected faces
    eye_count  = 0

    for (x, y, w, h) in faces:

        # Draw blue rectangle around face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # Label above the face rectangle
        cv2.putText(frame, 'Face', (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        # ── Step 2: Detect Eyes INSIDE the face region only ──────────────────
        roi_gray  = gray[y:y+h, x:x+w]     # Grayscale face region
        roi_color = frame[y:y+h, x:x+w]    # Color face region (for drawing)

        eyes = eye_cascade.detectMultiScale(
            roi_gray,
            scaleFactor  = 1.1,
            minNeighbors = 10,    # Higher value avoids false eye detections
            minSize      = (20, 20)
        )

        eye_count += len(eyes)

        for (ex, ey, ew, eh) in eyes:
            # Draw green rectangle around each eye
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
            cv2.putText(roi_color, 'Eye', (ex, ey - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # ── Step 3: Display Stats on Frame ───────────────────────────────────────
    cv2.putText(frame, f'Faces: {face_count}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)
    cv2.putText(frame, f'Eyes : {eye_count}', (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)

    # ── Step 4: Show the Frame ───────────────────────────────────────────────
    cv2.imshow('Real-Time Face & Eye Detection  |  q = quit', frame)

    # ── Key Controls ─────────────────────────────────────────────────────────
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):        # Q → quit
        print("Quitting...")
        break
    elif key == ord('s'):      # S → save screenshot
        cv2.imwrite('screenshot.png', frame)
        print("Screenshot saved!")

# ─── Cleanup ──────────────────────────────────────────────────────────────────
cap.release()
cv2.destroyAllWindows()