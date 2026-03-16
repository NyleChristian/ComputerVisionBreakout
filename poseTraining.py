from ultralytics import YOLO

def main():

    model = YOLO("yolo26m-pose.pt")

    results = model.train(
        data = "hand-keypoints.yaml",
        epochs=100,
        imgsz=1000, 
        batch=6,
        patience=20,
        device=0,

        workers=4,          # FIX 2: Forces data loading on the main thread (Fixes Windows IPC crashes)
        
    )


if __name__ == '__main__':
    main()
