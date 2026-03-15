from ultralytics import YOLO

def main():

    model = YOLO("yolo26n-pose.pt")

    results = model.train(
        data = "hand-keypoints.yaml",
        epochs=50,
        imgsz=1000, 
        batch=24,
        patience=20,
        device=0,

        workers=0,          # FIX 2: Forces data loading on the main thread (Fixes Windows IPC crashes)
        
    )


if __name__ == '__main__':
    main()
