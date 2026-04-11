"""
Professional-Grade Real-Time Glare Remover
Objective: CCTV surveillance, vehicle cameras, and mobile cameras.
Algorithm: Intelligent HSV-based detection + LAB-space auto-gain control.
"""

import cv2
import numpy as np

def detect_glare_mask(hsv_frame, v_threshold=235, s_threshold=50, dilation=2):
    """
    Intelligently detects glare by looking for High Brightness (V) 
    and Low Color Saturation (S). Glare is typically white.
    """
    s = hsv_frame[:, :, 1]
    v = hsv_frame[:, :, 2]
    
    # Mask: Bright (V > thresh) AND Colorless (S < thresh)
    mask = cv2.bitwise_and(
        cv2.threshold(v, v_threshold, 255, cv2.THRESH_BINARY)[1],
        cv2.threshold(s, s_threshold, 255, cv2.THRESH_BINARY_INV)[1]
    )
    
    # Refine mask with morphology
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    # Expand mask to cover the "bloom" (glow) around the glare
    if dilation > 0:
        mask = cv2.dilate(mask, kernel, iterations=dilation)
    return mask

def apply_auto_gain(lab_frame, target_l=128, mask=None):
    """
    Simulates a CCTV's Auto-Gain Control (AGC) by normalizing scene brightness.
    If 'mask' is provided, it calculates brightness statistics ignoring glare areas.
    """
    l, a, b = cv2.split(lab_frame)
    
    # Calculate current median brightness (ignoring glare if mask exists)
    if mask is not None and np.any(mask > 0):
        # We only consider pixels where mask is 0 (no glare)
        valid_l = l[mask == 0]
        if valid_l.size > 0:
            curr_median = np.median(valid_l)
        else:
            curr_median = np.median(l)
    else:
        curr_median = np.median(l)
    
    # Calculate brightness shift to reach target
    shift = target_l - curr_median
    
    # Apply shift safely (clipping)
    l = cv2.add(l, int(shift))
    
    # Apply local contrast enhancement (CLAHE)
    # clipLimit 3.0 gives better definition than 2.5
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    
    return cv2.merge((l, a, b))

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Could not open camera.")
        return

    # Attempt to set HD resolution for better clarity
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    # Check if HD was applied, if not standard 720p fallback
    actual_w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    print(f"INFO: Camera initialized at {int(actual_w)}p resolution.")

    window_name = "Professional Glare Remover (CCTV Mode)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    # Professional controls
    def nothing(x): pass
    cv2.createTrackbar("Detection Sensitivity", window_name, 230, 255, nothing)
    cv2.createTrackbar("Mask Expansion (Bloom)", window_name, 3, 15, nothing)
    cv2.createTrackbar("Inpaint Intensity", window_name, 5, 20, nothing)
    cv2.createTrackbar("Detail Sharpening", window_name, 3, 10, nothing)
    cv2.createTrackbar("Auto-Gain Target", window_name, 125, 255, nothing)

    while True:
        ret, frame = cap.read()
        if not ret: break

        # 1. Image Formats
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)

        # 2. Settings from UI
        v_thresh = cv2.getTrackbarPos("Detection Sensitivity", window_name)
        bloom_val = cv2.getTrackbarPos("Mask Expansion (Bloom)", window_name)
        inpaint_rad = cv2.getTrackbarPos("Inpaint Intensity", window_name)
        sharp_val = cv2.getTrackbarPos("Detail Sharpening", window_name)
        gain_target = cv2.getTrackbarPos("Auto-Gain Target", window_name)

        # 3. Intelligent Glare Detection (including bloom control)
        mask = detect_glare_mask(hsv, v_threshold=v_thresh, dilation=bloom_val)

        # 4. Auto-Gain Control (now ignores glare pixels for statistics)
        normalized_lab = apply_auto_gain(lab, target_l=gain_target, mask=mask)
        result = cv2.cvtColor(normalized_lab, cv2.COLOR_LAB2BGR)

        # 5. Intelligent Inpainting (Reconstruction)
        # Instead of just blurring, we reconstruction the glare area using surrounding pixels
        if inpaint_rad > 0:
            # We inpaint on the color-corrected frame for best blending
            result = cv2.inpaint(result, mask, inpaint_rad, cv2.INPAINT_TELEA)

        # 6. Detail Sharpening (Unsharp Mask)
        if sharp_val > 0:
            sigma = 1.0
            blurred = cv2.GaussianBlur(result, (0, 0), sigma)
            # result = original + (original - blurred) * amount
            # Using addWeighted for clarity and speed
            result = cv2.addWeighted(result, 1.0 + (sharp_val/10.0), blurred, -(sharp_val/10.0), 0)

        # 7. Final Polish: Gamma for highlights
        gamma = 1.0 # Set to neutral by default for clarity
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        result = cv2.LUT(result, table)

        # Output View (Original vs Corrected)
        combined = cv2.hconcat([frame, result])
        
        # Display performance info
        cv2.putText(combined, "MODE: PROFESSIONAL SURVEILLANCE", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow(window_name, combined)

        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()