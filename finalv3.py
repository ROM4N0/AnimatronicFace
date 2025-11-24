import threading, serial,time,pyautogui,keyboard,random,cv2,numpy as np, wave, simpleaudio as sa, os.path,sounddevice as sd



smoothed_rms = 0.0

def audio_callback(indata, frames, time_info, status):

                global mouth_position
                mouth_position = -2
                # Parameters
                samplerate = 44100
                frame_size = int(0.02 * samplerate)  # 20ms chunks
                threshold = 0.02
                attack = 0.6
                decay = 0.2
                base_angle = 50
                max_angle = 105
                gamma = 0.5
                gain = 3.0
                global smoothed_rms

                # Convert to mono float
                chunk = indata[:, 0].astype(np.float32)
                if len(chunk) == 0:
                    return

                mean_sq = np.mean(chunk**2)
                rms = np.sqrt(mean_sq) if mean_sq > 0 else 0
                if rms < threshold:
                    rms = 0

                # Smooth attack/decay
                if rms > smoothed_rms:
                    smoothed_rms += (rms - smoothed_rms) * attack
                else:
                    smoothed_rms += (rms - smoothed_rms) * decay

                # Scale + nonlinear curve
                normalized = min(1.0, (smoothed_rms * gain) ** gamma)
                mouth_position = int(base_angle + (max_angle - base_angle) * normalized)

                # Send to Arduino here
                # arduino.write(bytes(f"<M{mouth_position:03}>", "utf-8"))

                # print(f"Mouth: {mouth_position}° (RMS {smoothed_rms:.3f})")

            

def get_mask(frame1,frame2,kernel=np.array((9,9),np.uint8)):

    framedif=cv2.subtract(frame2,frame1)
    framedif=cv2.medianBlur(framedif,5)


    mask = cv2.adaptiveThreshold(framedif, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 11, 5)

    mask = cv2.medianBlur(mask, 5)

    # morphological operations
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    return mask



arduino = serial.Serial(port='COM5',baudrate=9600,timeout=1)
cap = cv2.VideoCapture(0)
mouth_position=45
debug = False
ti=time.time()
rand=(random.random()*10)+5 ## To blink in a random ammount of seconds between [5;15]
flag=0
double = False # always False
lx,ly=None,None
cx,cy=0,0
blink=True

def audio_thread():
    global mouth_position
    while(True):
        op=int(input("Which mode would u like to hear?\n\t1 - File\n\t2 - Microphone\n"))
        if op==1: 
            wav_file = input("Which file do you want?")
            #wav_file = "audio/harvard.wav"
            while not os.path.isfile(wav_file):
                wav_file = input("Error 404: File not found\nWhich file do you want?")
            mouth_position=-2
            obj = wave.open(wav_file, "rb")
            framerate = obj.getframerate()
            n_channels = obj.getnchannels()
            n_frames = obj.getnframes()
            frames = obj.readframes(n_frames)
            data = np.frombuffer(frames, dtype=np.int16)

            
            obj.close()

            if n_channels > 1:
                data = data.reshape(-1, n_channels)
                data = data[:, 0]

            #max_val = np.max(np.abs(data))
            max_val = np.percentile(np.abs(data), 98)
            frame_size = int(framerate * 0.04)  # 40ms chunks
            threshold = 0.05
            if max_val < 10_000:
                max_val += max_val**.7
            # Parameters
            threshold = 0.02        # smaller threshold so even quiet sounds move
            attack = 0.6
            decay = 0.4
            base_angle  =   50         # closed
            max_angle   =  105         # wide open

            smoothed_rms = 0.0

            # play audio
            wave_obj = sa.WaveObject(frames, n_channels, 2, framerate)
            play_obj = wave_obj.play()
            startTime = time.time()
            if max_val==0:
                max_val=1
            
            wavTime=n_frames/framerate
            diftime=time.time()-startTime

            while diftime < wavTime:
                startChunk = int(framerate * diftime)
                endChunk = startChunk + frame_size

                if startChunk >= len(data):
                    break

                chunk = data[startChunk:endChunk].astype(np.float32)
                if len(chunk) == 0:
                    rms = 0
                else:
                    mean_sq = np.mean(chunk ** 2)
                    rms = np.sqrt(mean_sq) / max_val if mean_sq > 0 else 0
                    if rms < threshold:
                        rms = 0

                # Smooth (attack/decay)
                if rms > smoothed_rms:
                    smoothed_rms += (rms - smoothed_rms) * attack
                else:
                    smoothed_rms += (rms - smoothed_rms) * decay
                if(smoothed_rms>0):
                    smoothed_rms = smoothed_rms**.75
                '''if random.random() > .5:
                    noise = (smoothed_rms + (random.random() * .10))
                else:
                    noise = (smoothed_rms - (random.random() * .10))
                if noise < 0:
                    noise=0 
                elif noise>1:
                    noise=1
                '''
                mouth_position = int(base_angle + (max_angle - base_angle) * smoothed_rms)

                print(f"Mouth: {mouth_position}°  (RMS {smoothed_rms:.3f}) Seconds passed: {diftime:.3f}")
                # Sleep per chunk
                #time.sleep(frame_size / framerate)
                diftime = time.time() - startTime


            play_obj.wait_done()
            mouth_position=-1

        elif 2: # Microphone

            with sd.InputStream(callback=audio_callback,
                                channels=1,
                                samplerate=44100,
                                blocksize=int(0.02 * 44100)):
                print("Listening... press S to stop")
                while True:
                    time.sleep(0.1)
                    if keyboard.is_pressed('S'):
                        print("Microphone Stopped")
                        mouth_position = -1
                        break

            
t = threading.Thread(target=audio_thread, daemon=True)
t.start()

while not keyboard.is_pressed('esc'):

    _,m = pyautogui.position()
    m=int((1079-m)*(120/1079))+30

    if blink and time.time()-ti>rand:
        if rand<5.05:
            double=True
        arduino.write(bytes(f"<X900Y900M{mouth_position:03}>","utf-8"))
        rand=(random.random()*10)+5
        ti=time.time()
        
        if debug:
            print(arduino.readline())
        else:
            arduino.readline()

    else:
        ## x,y = pyautogui.position()
        ret,frame = cap.read()
        
        if not ret:
            break

        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian blur
        blur = cv2.GaussianBlur(gray, (5, 5), 0)

        if flag==0:
            lastframe=blur
            flag=1
        #diff_frame = cv2.absdiff(lastframe,gray)
        diff_frame=get_mask(lastframe,gray)

        thresh_frame = cv2.threshold(diff_frame,30,255,cv2.THRESH_BINARY)[1]

        thresh_frame = cv2.dilate(thresh_frame,None,iterations=2)
        
        M = cv2.moments(thresh_frame)
        
        if M["m00"] != 0:  # avoid division by zero
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        
        if cx is not None and cy is not None:
            if lx is None or ly is None:
                # First frame, just accept it
                lx, ly = cx, cy
            else:
                vec=[cx-lx,cy-ly]
                ## 
                vecx=vec[0]**2
                vecy=vec[1]**2
                
                if vecx + vecy <= 10000: # 100**2 = 100000
                    lx,ly=cx,cy
                else:
                    mag = np.sqrt(vecx + vecy)
                    vec=(vec/mag)*100
                    lx,ly=round(vec[0]+lx),round(vec[1]+ly)

        # Draw the stabilized dot
        cv2.circle(frame, (lx, ly), 5, (0, 255, 0), -1)
        cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
        
        cv2.imshow('idk1', cv2.flip(frame, 1))
        cv2.imshow('thresh', cv2.flip(thresh_frame, 1))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        lastframe=blur

        ## Formula to convert given position to eyes :: x = x * (90/MAX_x) + 45
        ## Video Resolution = 640x480
        if lx is not None and ly is not None:
            x=int((640-lx)*(90/640))+45
            y=int((480-ly)*(90/480))+45
            
            arduino.write(bytes(f"<X{x:03}Y{y:03}M{mouth_position:03}>","utf-8"))
            if debug:
                print(arduino.readline())
            else:
                arduino.readline()


        if double:
                arduino.write(bytes(f"<X900Y900M{mouth_position:03}>","utf-8"))
                double=False
                if debug:
                    print(arduino.readline())
                else:
                    arduino.readline()

arduino.write(bytes(f"<X090Y090M{mouth_position:03}>","utf-8"))
if debug:
    print(arduino.read())
else:
    arduino.readline()

cap.release()
cv2.destroyAllWindows()