import cv2, wave, os, random, threading, time, pyautogui, keyboard, serial, simpleaudio as sa, sounddevice as sd, numpy as np
from sklearn.cluster import DBSCAN

# -----------------------------
# CONFIG / TUNABLE PARAMETERS
# -----------------------------
DEBUG = False
SERIAL_PORT = 'COM5'
SERIAL_BAUD = 9600

# Vision / attention params
CENTER_FOCUS_STRENGTH = 0.7
CURIOSITY_CHANCE = 0.02
MOVEMENT_THRESHOLD = 10_000
COOLDOWN_FRAMES = 50

# DBSCAN params (tune for your camera / scene)
DBSCAN_EPS = 12
DBSCAN_MIN_SAMPLES = 40

# smoothing / persistence
SMOOTH_ALPHA_FOCUSED = 0.40
SMOOTH_ALPHA_CURIOUS = 0.20
PERSIST_DIST_THRESHOLD = 30.0
PERSIST_SCORE_RATIO = 1.10

# mapping to servo ranges (assumes camera resolution 640x480)
CAM_W = 640
CAM_H = 480
SERVO_MIN = 45
SERVO_MAX = 135

# audio / mouth params (used in audio thread / callback)
SMOOTHED_RMS = 0.0

# Standby Position
mouth_default = 50
eyes_default = 90

# starting flag
fstart=0


# -----------------------------
# helpers: mask and select_target
# -----------------------------
def visu_get_mask(frame1, frame2, threshold_value=30, kernel_size=3):
    diff = cv2.absdiff(frame1, frame2)
    if len(diff.shape) == 3:
        diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(diff, threshold_value, 255, cv2.THRESH_BINARY)
    mask = cv2.medianBlur(mask, 3)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask

def get_attention_mode_from_scene(num_clusters, avg_aspect):
    """Auto mode suggestion from cluster stats."""
    if num_clusters >= 20 or avg_aspect > 1.5:
        return "curious"
    elif num_clusters <= 8 and avg_aspect < 1.0:
        return "focused"
    else:
        return "neutral"

def select_target(points, last_pos, frame_shape,
                  external_mode="neutral",
                  eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES):
    """
    points: Nx2 array of (y,x) coordinates of motion pixels
    last_pos: previous target position (y,x) or None
    frame_shape: (h, w)
    external_mode: "focused" / "curious" / "neutral" (from main loop)
    returns: best_centroid (y,x) or None, labels (for visualization)
    """
    if points is None or len(points) == 0:
        return None, None

    # DBSCAN expects float features
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(points)
    labels = clustering.labels_
    unique_labels = [l for l in np.unique(labels) if l != -1]
    if len(unique_labels) == 0:
        return None, labels

    h, w = frame_shape
    center = np.array([h / 2.0, w / 2.0])

    cluster_data = []
    aspect_ratios = []
    for label in unique_labels:
        cluster_pts = np.array(points)[labels == label]
        centroid = cluster_pts.mean(axis=0)
        size = len(cluster_pts)

        y_coords, x_coords = cluster_pts[:, 0], cluster_pts[:, 1]
        width = (x_coords.max() - x_coords.min()) or 1.0
        height = (y_coords.max() - y_coords.min()) or 1.0
        aspect = width / height
        aspect_ratios.append(aspect)

        cluster_data.append((label, centroid, size))

    avg_aspect = np.mean(aspect_ratios) if len(aspect_ratios) > 0 else 1.0
    num_clusters = len(unique_labels)

    # combine external mode with automatic suggestion (prefer external_mode if given)
    auto_mode = get_attention_mode_from_scene(num_clusters, avg_aspect)
    mode = external_mode if external_mode in ("focused", "curious", "neutral") else auto_mode

    # Behavior tuning (weights)
    if mode == "focused":
        w_d, w_s, w_c = 0.25, 0.1, 0.65
    elif mode == "curious":
        w_d, w_s, w_c = 0.4, 0.4, 0.2
    else:
        w_d, w_s, w_c = 0.3, 0.3, 0.4

    scores = []
    for label, centroid, size in cluster_data:
        size_score = size
        dist_score = 0.0
        center_score = 0.0
        if last_pos is not None:
            dist = np.linalg.norm(centroid - last_pos)
            dist_score = 1.0 / (1.0 + dist)
        dist_center = np.linalg.norm(centroid - center)
        center_score = 1.0 / (1.0 + dist_center)
        score = w_d * dist_score + w_s * size_score + w_c * center_score
        scores.append((score, centroid))

    best_score, best_centroid = max(scores, key=lambda x: x[0])

    # Stability: only switch if best is sufficiently better OR distance small
    if last_pos is not None:
        dist = np.linalg.norm(best_centroid - last_pos)
        if dist > PERSIST_DIST_THRESHOLD:
            if len(scores) > 1:
                second_best = sorted(scores, key=lambda x: x[0], reverse=True)[1][0]
            else:
                second_best = best_score
            if best_score < second_best * PERSIST_SCORE_RATIO:
                # keep previous target
                return last_pos, labels

    return best_centroid, labels

# -----------------------------
# Serial / Arduino init
# -----------------------------
arduino = None
try:
    arduino = serial.Serial(port=SERIAL_PORT, baudrate=SERIAL_BAUD, timeout=1)
    time.sleep(2.0)  # allow serial to initialize
    if DEBUG:
        print(f"[DEBUG] Serial connected to {SERIAL_PORT}")
except Exception as e:
    print(f"[WARN] Could not open serial port {SERIAL_PORT}: {e}")
    arduino = None

# -----------------------------
# Audio / mouth control (thread + callback)
# -----------------------------
mouth_position = mouth_default  # default
smoothed_rms = 0.0

def audio_callback(indata, frames, time_info, status):
    global smoothed_rms, mouth_position
    # parameters
    threshold = 0.02
    attack = 0.6
    decay = 0.2
    base_angle = 50
    max_angle = 105
    gamma = 0.5
    gain = 3.0

    chunk = indata[:, 0].astype(np.float32)
    if len(chunk) == 0:
        return
    mean_sq = np.mean(chunk**2)
    rms = np.sqrt(mean_sq) if mean_sq > 0 else 0
    if rms < threshold:
        rms = 0.0

    if rms > smoothed_rms:
        smoothed_rms += (rms - smoothed_rms) * attack
    else:
        smoothed_rms += (rms - smoothed_rms) * decay

    normalized = min(1.0, (smoothed_rms * gain) ** gamma)
    mouth_position = int(base_angle + (max_angle - base_angle) * normalized)


def audio_thread():
    global mouth_position
    global fstart
    while fstart==0:
        pass
    del fstart

    while True:
        try:
            op = int(input("Which mode would u like to hear?\n\t1 - File\n\t2 - Microphone\n"))
        except:
            continue
        if op == 1:

            wav_file = input("Which file do you want? ")
            while not os.path.isfile(wav_file):
                wav_file = input("File not found. Try again: ")
            obj = wave.open(wav_file, "rb")
            framerate = obj.getframerate()
            n_channels = obj.getnchannels()
            n_frames = obj.getnframes()
            frames = obj.readframes(n_frames)
            obj.close()

            data = np.frombuffer(frames, dtype=np.int16)
            if n_channels > 1:
                data = data.reshape(-1, n_channels)
                data = data[:, 0]
            max_val = np.percentile(np.abs(data), 98)
            if max_val == 0:
                max_val = 1
            frame_size = int(framerate * 0.04)
            attack = 0.6
            decay = 0.4
            base_angle = 50
            max_angle = 105

            sm = 0.0
            sa.stop_all()
            wave_obj = sa.WaveObject(frames, n_channels, 2, framerate)
            play_obj = wave_obj.play()
            startTime = time.time()
            wavTime = n_frames / framerate
            while time.time() - startTime < wavTime:
                diftime = time.time() - startTime
                startChunk = int(framerate * diftime)
                endChunk = startChunk + frame_size
                chunk = data[startChunk:endChunk].astype(np.float32)
                if len(chunk) == 0:
                    rms = 0.0
                else:
                    mean_sq = np.mean(chunk ** 2)
                    rms = np.sqrt(mean_sq) / max_val if mean_sq > 0 else 0.0
                    if rms < 0.02:
                        rms = 0.0
                if rms > sm:
                    sm += (rms - sm) * attack
                else:
                    sm += (rms - sm) * decay
                if sm > 0:
                    sm = sm ** 0.75
                mouth_position = int(base_angle + (max_angle - base_angle) * sm)
                time.sleep(frame_size / framerate)
            
            play_obj.wait_done()
            sa.stop_all()
            mouth_position = mouth_default

        elif op == 2:
            if True:
                print("Currently under work!")
            else:    
                with sd.InputStream(callback=audio_callback, channels=1, samplerate=44100,
                                    blocksize=int(0.02 * 44100)):
                    print("Listening... press 'S' to stop microphone mode.")
                    while True:
                        time.sleep(0.1)
                        if keyboard.is_pressed('S'):
                            mouth_position = mouth_default
                            break

# start audio thread
t = threading.Thread(target=audio_thread, daemon=True)
t.start()

# -----------------------------
# Video capture & main loop
# -----------------------------
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
ret, frame = cap.read()
if not ret:
    raise RuntimeError("Cannot read from camera.")

# initialization
flag = 0
lastframe = None
lx = ly = None  # last target coords (x,y in image coords)
frame_count = 0
last_curiosity_frame = -COOLDOWN_FRAMES
mode = "focused"
rand_blink_timer = time.time()
rand_blink_next = (random.random() * 10) + 5
doblink = False
print("[INFO] Starting main loop. Press ESC to exit.")
fstart = 1

try:
    while not keyboard.is_pressed('esc'):
        ret, frame = cap.read()
        if not ret:
            print("[WARN] frame read failed; breaking.")
            break

        # Preprocess
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)

        if flag == 0:
            lastframe = blur.copy()
            flag = 1

        diff_frame = visu_get_mask(lastframe, blur)
        thresh_frame = cv2.threshold(diff_frame, 30, 255, cv2.THRESH_BINARY)[1]
        thresh_frame = cv2.dilate(thresh_frame, None, iterations=2)

        points = np.column_stack(np.where(thresh_frame == 255))
        motion_level = len(points)

        # curiosity trigger
        if (motion_level > MOVEMENT_THRESHOLD or random.random() < CURIOSITY_CHANCE) and \
                (frame_count - last_curiosity_frame > COOLDOWN_FRAMES):
            mode = "curious"
            last_curiosity_frame = frame_count
        else:
            mode = "focused"

        colored = np.zeros((thresh_frame.shape[0], thresh_frame.shape[1], 3), dtype=np.uint8)

        target = None
        labels = None

        if points.size > 0:
            # downsample points for performance
            N = 6_000
            if len(points) > N:
                idx = np.random.choice(len(points), N, replace=False)
                points_sampled = points[idx]
            else:
                points_sampled = points

            # select cluster target
            last_pos_for_select = np.array([ly, lx]) if (lx is not None and ly is not None) else None
            sel_target, labels = select_target(points_sampled, last_pos_for_select, thresh_frame.shape, mode)
            if sel_target is not None:
                # sel_target is (y, x)
                # smoothing
                if lx is not None and ly is not None:
                    alpha = SMOOTH_ALPHA_FOCUSED if mode == "focused" else SMOOTH_ALPHA_CURIOUS
                    ly = int(alpha * ly + (1 - alpha) * sel_target[0])
                    lx = int(alpha * lx + (1 - alpha) * sel_target[1])
                else:
                    ly, lx = int(sel_target[0]), int(sel_target[1])
                target = (lx, ly)
            else:
                # no cluster target found -> fall back to raw centroid (moments)
                M = cv2.moments(thresh_frame)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    if lx is None or ly is None:
                        lx, ly = cx, cy
                    else:
                        alpha = SMOOTH_ALPHA_FOCUSED if mode == "focused" else SMOOTH_ALPHA_CURIOUS
                        lx = int(alpha * lx + (1 - alpha) * cx)
                        ly = int(alpha * ly + (1 - alpha) * cy)
                    target = (lx, ly)

            # labels visualization
            if labels is not None and len(labels) > 0:
                n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                colors = np.random.randint(0, 255, size=(n_clusters if n_clusters>0 else 1, 3))
                # label colors map: dbscan labels -> index [0..n_clusters-1]
                label_to_idx = {}
                idx_counter = 0
                for l in sorted(set(labels)):
                    if l == -1:
                        continue
                    label_to_idx[l] = idx_counter
                    idx_counter += 1
                for pi, lab in enumerate(labels):
                    y, x = points_sampled[pi]
                    if lab != -1:
                        colored[y, x] = colors[label_to_idx[lab] % len(colors)]

        # draw tracking markers
        if lx is not None and ly is not None:
            cv2.circle(frame, (lx, ly), 8, (0, 255, 0), -1)  # stabilized target
        # also show raw center of motion
        M = cv2.moments(thresh_frame)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)

        # Show debug windows
        cv2.imshow('Camera', cv2.flip(frame, 1))
        cv2.imshow('Motion', cv2.flip(thresh_frame, 1))
        cv2.imshow('Clusters', cv2.flip(colored, 1))

        # Map to servo angles and send to Arduino
        if lx is not None and ly is not None:
            # Map x (0..CAM_W) to servo X (SERVO_MIN..SERVO_MAX)
            servo_x = int(((CAM_W - lx) / CAM_W) * (SERVO_MAX - SERVO_MIN) + SERVO_MIN)
            servo_y = int(((CAM_H - ly) / CAM_H) * (SERVO_MAX - SERVO_MIN) + SERVO_MIN)
            if arduino:
                try:
                    arduino.write(bytes(f"<X{servo_x:03}Y{servo_y:03}M{mouth_position:03}>", "utf-8"))
                    # read ack (non-blocking)
                    if DEBUG:
                        print(arduino.readline())
                    else:
                        arduino.readline()
                except Exception as e:
                    if DEBUG:
                        print("[DEBUG] Serial write failed:", e)
                    # drop serial if broken
                    arduino = None

        # blinking (randomized)
        if time.time() - rand_blink_timer > rand_blink_next:
            # perform blink
            if arduino:
                try:
                    arduino.write(bytes(f"<X900Y900M{mouth_position:03}>", "utf-8"))
                    if DEBUG:
                        print("[DEBUG] Blink sent")
                    else:
                        arduino.readline()
                except:
                    pass
            rand_blink_timer = time.time()
            rand_blink_next = (random.random() * 10) + 5

        # keyboard / exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # prepare next iteration
        frame_count += 1
        lastframe = blur.copy()

    # end main loop

finally:
    # send home position
    try:
        if arduino:
            arduino.write(bytes(f"<X{eyes_default:03}Y{eyes_default:03}M{mouth_default:03}>", "utf-8"))
            time.sleep(0.1)
            arduino.readline()
    except:
        pass
    cap.release()
    cv2.destroyAllWindows()
    if arduino:
        try:
            arduino.close()
        except:
            pass

print("[INFO] Exited cleanly.")