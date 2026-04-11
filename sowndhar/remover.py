# ai_glare_remover.py
import cv2
import numpy as np
import os
import time

# Optional: TensorFlow model support
USE_TF_MODEL = True            # set False to skip AI model
TF_MODEL_PATH = "glare_model.h5"      # path to a Keras .h5 or SavedModel directory
TFLITE_MODEL_PATH = "glare_model.tflite"  # alternative: TFLite file

# ---------- Helper: load TF/TFLite model if available ----------
tf_model = None
tflite_interpreter = None
use_tflite = False

if USE_TF_MODEL:
    try:
        import tensorflow as tf
        # Prefer tflite if exists
        if os.path.exists(TFLITE_MODEL_PATH):
            use_tflite = True
            from tensorflow.lite.python.interpreter import Interpreter
            tflite_interpreter = Interpreter(model_path=TFLITE_MODEL_PATH)
            tflite_interpreter.allocate_tensors()
            print("[INFO] Using TFLite model:", TFLITE_MODEL_PATH)
        elif os.path.exists(TF_MODEL_PATH):
            tf_model = tf.keras.models.load_model(TF_MODEL_PATH, compile=False)
            print("[INFO] Loaded TF model:", TF_MODEL_PATH)
        else:
            print("[INFO] No AI model found — running traditional pipeline.")
    except Exception as e:
        print("[WARN] TensorFlow import failed or model load error:", e)
        tf_model = None
        tflite_interpreter = None
        use_tflite = False

# ---------- Utility functions ----------
def detect_glare_mask(frame):
    """Return a binary mask of glare areas (255 where glare is detected)."""
    # Resize for stable performance detection (but keep original for inpaint)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Detect very bright/white regions: low saturation, high value
    # Tune these ranges if needed
    lower = np.array([0, 0, 220], dtype=np.uint8)
    upper = np.array([180, 60, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)

    # Optional: also detect very high intensity in BGR
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, bright = cv2.threshold(gray, 230, 255, cv2.THRESH_BINARY)
    mask = cv2.bitwise_or(mask, bright)

    # Morphological ops to clean mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    # Dilate slight to cover halos/rays
    mask = cv2.dilate(mask, kernel, iterations=1)
    return mask

def post_process(img):
    """Improve visual clarity: CLAHE on L-channel, edge-preserving, sharpen."""
    # Edge-preserving filter (fast)
    img_ep = cv2.edgePreservingFilter(img, flags=1, sigma_s=60, sigma_r=0.4)

    # CLAHE on L channel
    lab = cv2.cvtColor(img_ep, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    merged = cv2.merge((cl, a, b))
    enhanced = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

    # Light sharpening
    kernel_sharp = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]], dtype=np.float32)
    sharp = cv2.filter2D(enhanced, -1, kernel_sharp)
    return sharp

def run_tflite_on_frame(frame, interpreter, input_size=(256,256)):
    # Preprocess: resize and normalize
    inp = cv2.resize(frame, input_size)
    inp = inp.astype(np.float32) / 255.0
    inp = np.expand_dims(inp, axis=0)  # [1,H,W,3]

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    # Match dtype
    if input_details[0]['dtype'] == np.uint8:
        inp_t = (inp * 255).astype(np.uint8)
    else:
        inp_t = inp.astype(np.float32)

    interpreter.set_tensor(input_details[0]['index'], inp_t)
    interpreter.invoke()
    out = interpreter.get_tensor(output_details[0]['index'])
    # Postprocess
    out = np.squeeze(out)
    if out.dtype != np.uint8:
        out = np.clip(out * 255.0, 0, 255).astype(np.uint8)
    out = cv2.resize(out, (frame.shape[1], frame.shape[0]))
    return out

def run_tf_on_frame(frame, model, input_size=(256,256)):
    inp = cv2.resize(frame, input_size)
    inp = inp.astype(np.float32) / 255.0
    inp = np.expand_dims(inp, axis=0)
    out = model.predict(inp)  # shape [1,H,W,3] expected
    out = np.squeeze(out)
    out = np.clip(out * 255.0, 0, 255).astype(np.uint8)
    out = cv2.resize(out, (frame.shape[1], frame.shape[0]))
    return out

# ---------- Main loop ----------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Cannot open webcam (check camera index)")

# Optional: set smaller resolution for higher FPS if needed
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

fps_time = time.time()
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    # 1) Detect glare mask
    mask = detect_glare_mask(frame)

    # 2) Inpaint using mask (works better for small-to-medium glare)
    # Use Telea or NS depending on look; Telea is faster
    inpainted = cv2.inpaint(frame, mask, 7, cv2.INPAINT_TELEA)

    # 3) If AI model loaded, run it to refine result
    ai_result = None
    if tflite_interpreter is not None:
        try:
            ai_result = run_tflite_on_frame(inpainted, tflite_interpreter, input_size=(256,256))
        except Exception as e:
            print("[WARN] TFLite inference failed:", e)
            tflite_interpreter = None
    elif tf_model is not None:
        try:
            ai_result = run_tf_on_frame(inpainted, tf_model, input_size=(256,256))
        except Exception as e:
            print("[WARN] TF inference failed:", e)
            tf_model = None

    # 4) Post-process: if AI result available use it, else use inpainted
    if ai_result is not None:
        final = post_process(ai_result)
    else:
        final = post_process(inpainted)

    # 5) Display: show original and final side-by-side
    try:
        combined = np.hstack((cv2.resize(frame, (640,480)), cv2.resize(final, (640,480))))
    except:
        combined = np.hstack((frame, final))
    cv2.imshow("Original | GlareRemoved (press q to quit, s to save)", combined)

    # show FPS occasionally
    if frame_count % 30 == 0:
        now = time.time()
        fps = 30.0 / max(1e-6, now - fps_time)
        fps_time = now
        # print or overlay FPS if desired
        # print(f"[INFO] approx fps: {fps:.1f}")

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    if key == ord('s'):
        # save sample
        cv2.imwrite("glare remover image.jp", frame)
        cv2.imwrite("sample_processed.jpg", final)
        print("[INFO] Saved sample_original.jpg and sample_processed.jpg")

cap.release()
cv2.destroyAllWindows()
