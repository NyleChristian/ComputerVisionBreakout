import cv2
from ultralytics import YOLO

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

    # 3. Process the video stream
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame. Exiting...")
            break

        # Flip the frame horizontally for a "mirror" effect (feels more natural for gaming)
        frame = cv2.flip(frame, 1)

        # 4. Run inference (device=0 ensures it uses your RTX 4070)
        # stream=True is highly recommended for live video to save memory
        results = model(frame, device=0, stream=True, conf=0.5) 

        # 5. Draw the bounding boxes on the frame
        for r in results:
            # r.plot() automatically draws the boxes and labels based on the results
            annotated_frame = r.plot()
            
            # Display the resulting frame
            cv2.imshow("YOLO11m Live Hand Detection", annotated_frame)

        # 6. Listen for the 'q' key to stop the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 7. Clean up
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # UPDATE THIS PATH to your new runs folder!
    CUSTOM_WEIGHTS_PATH = "C:\\Users\\Nyle\\Documents\\ResumeProjects\\ComputerVisionBreakout\\runs\\detect\\train\\weights\\best.pt" 
    
    run_live_detection(CUSTOM_WEIGHTS_PATH)