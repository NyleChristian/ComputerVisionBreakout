from ultralytics import YOLO

def main():

    model = YOLO("yolo11n.pt")

    results = model.train(
        data = "data.yaml",
        epochs=100,
        imgsz=800, 
        batch=-1,
        patience=20,
        device=0,
        
    )

if __name__ == '__main__':
    main()