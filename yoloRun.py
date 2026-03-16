import cv2
from ultralytics import YOLO
from PIL import Image
import time

def run_live_detection(model_path, webcam_index=0):
    # 1. Load your custom-trained YOLO11m model
    print(f"Loading model from {model_path}...")
    model = YOLO(model_path)
    
    # 2. Open the webcam
    cap = cv2.VideoCapture(webcam_index)
    
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Webcam started. Press 'q' to quit.")

    #GIF variables
    recording = False
    record_start_time = 0
    gif_frames = []
    frame_counter = 0

    # 3. Process the video stream
    # 3. Process the video stream
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame. Exiting...")
            break

        # Flip the frame horizontally for a "mirror" effect
        frame = cv2.flip(frame, 1)

        # 4. Run inference
        results = model(frame, device=0, stream=True, conf=0.5) 

        annotated_frame = frame
        # 5. Draw the bounding boxes on the frame
        for r in results:
            annotated_frame = r.plot()
            # ❌ REMOVED the extra cv2.imshow from here!

        # --- RECORDING LOGIC ---
        if recording:
            current_duration = time.time() - record_start_time
            
            # Draw the red circle and text
            cv2.circle(annotated_frame, (30, 30), 10, (0, 0, 255), -1)
            cv2.putText(annotated_frame, f"REC {current_duration:.1f}s", (50, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            frame_counter += 1
            if frame_counter % 3 == 0:
                rgb_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                small_frame = cv2.resize(rgb_frame, (460, 360))
                gif_frames.append(Image.fromarray(small_frame))

            if current_duration > 8:
                print("Time reached! Saving GIF...")
                if gif_frames:
                    gif_frames[0].save('demo.gif', save_all=True, append_images=gif_frames[1:], optimize=True, duration=100, loop=0)
                    print("✅ saved 'demo.gif' successfully!")
                recording = False
                gif_frames = []

        # ✅ ONLY ONE imshow HERE! (It will now show the boxes AND the red circle)
        cv2.imshow("YOLO Live Hand Detection", annotated_frame)
        
        # 6. Listen for keys
        key = cv2.waitKey(1) & 0xFF

        if key == ord('r') and not recording:
            print("Recording started!")
            recording = True
            record_start_time = time.time()
            gif_frames = []
            frame_counter = 0

        elif key == ord('q'):
            if gif_frames:
                print("💾 Saving partial recording before quitting...")
                gif_frames[0].save('demo.gif', save_all=True, append_images=gif_frames[1:], optimize=True, duration=100, loop=0)
            break
       

    # 7. Clean up
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # UPDATE THIS PATH to your new runs folder!
    CUSTOM_WEIGHTS_PATH = "C:\\Users\\Nyle\\Documents\\ResumeProjects\\ComputerVisionBreakout\\runs\\pose\\train9\\weights\\best.pt" 
    
    run_live_detection(CUSTOM_WEIGHTS_PATH)