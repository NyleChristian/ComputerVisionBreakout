from ultralytics import YOLO

def main():

    model = YOLO("yolo26x.pt")

    results = model.train(
        data = "data.yaml",
        epochs=100,
        imgsz=1000, 
        batch=6,
        patience=20,
        device=0,

        workers=8, 
        
    )

if __name__ == '__main__':
    main()