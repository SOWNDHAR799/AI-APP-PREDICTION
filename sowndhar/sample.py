import cv2
import numpy as np

def detect_glare_mask(image):
    """Detect glare regions and return a binary mask."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Bright white glare (low saturation, high value)
    lower = np.array([0, 0, 220], dtype=np.uint8)
    upper = np.array([180, 60, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)

    # Extra: detect bright spots in grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, bright = cv2.threshold(gray, 230, 255, cv2.THRESH_BINARY)
    mask = cv2.bitwise_or(mask, bright)

    # Clean up mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=1)

    return mask

def remove_glare(input_path, output_path):
    """Remove glare from an image file and save the result."""
    image = cv2.imread(input_path)
    if image is None:
        raise FileNotFoundError(f"Could not read {input_path}")

    mask = detect_glare_mask(image)

    # Inpainting
    result = cv2.inpaint(image, mask, 7, cv2.INPAINT_TELEA)

    # Save
    cv2.imwrite(output_path, result)
    print(f"[INFO] Saved glare-removed image as {output_path}")

# --------- Example usage ---------
if __name__ == "__main__":
    input_file = "remover.png"   # ✅ rename your file without spaces
    output_file = "output.png"
    
cv2.imshow("Glare Removed", result)
cv2.waitKey(0)  
cv2.destroyAllWindows()
def remove_glare(input_path, output_path):
    """Remove glare from an image file and save the result."""
    image = cv2.imread(input_path)
    if image is None:
        raise FileNotFoundError(f"Could not read {input_path}")

    mask = detect_glare_mask(image)

    # Inpainting
    result = cv2.inpaint(image, mask, 7, cv2.INPAINT_TELEA)

    # Save
    cv2.imwrite(output_path, result)
    print(f"[INFO] Saved glare-removed image as {output_path}")

    # ---------- Show images ----------
    cv2.imshow("Original", image)
    cv2.imshow("Glare Removed", result)
    cv2.waitKey(0)   # எந்த ஒரு key அழுத்தும் வரை காத்திருக்கும்
    cv2.destroyAllWindows()


