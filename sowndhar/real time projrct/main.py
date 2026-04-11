import cv2
import numpy as np

cap = cv2.VideoCapture(0)  # 0 for default webcam

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Threshold for glare detection (tune the value 240 if needed)
    _, mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)

    # Invert mask and apply Gaussian blur to smooth glare
    glare_removed = cv2.inpaint(frame, mask, 3, cv2.INPAINT_TELEA)

    # Show output
    cv2.imshow("Original", frame)
    cv2.imshow("Glare Removed", glare_removed)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
import cv2
import numpy as np

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize (optional, for speed)
    frame = cv2.resize(frame, (640, 480))

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Dynamically calculate threshold using mean brightness
    mean_brightness = np.mean(gray)
    glare_threshold = int(mean_brightness + 40)  # Adjust this value to improve

    # Create glare mask
    _, mask = cv2.threshold(gray, glare_threshold, 255, cv2.THRESH_BINARY)

    # Optional: Remove small noise using morphological operations
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # Inpaint glare area
    glare_removed = cv2.inpaint(frame, mask, 3, cv2.INPAINT_TELEA)

    # Show outputs
    cv2.imshow("Original", frame)
    cv2.imshow("Glare Removed", glare_removed)
    cv2.imshow("Glare Mask", mask)  # (For testing — can hide later)

    # Exit with 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
cv2.namedWindow("Glare Removed", cv2.WINDOW_NORMAL)
cv2.imshow("Glare Removed", glare_removed)
frame = cv2.resize(frame, (640, 480))
