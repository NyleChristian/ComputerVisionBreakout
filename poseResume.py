from ultralytics import YOLO

def main():

    model = YOLO("C:/Users/Nyle/Documents/ResumeProjects/ComputerVisionBreakout/runs/pose/train15/weights/last.pt")

    model.train(resume = True, batch=6, workers=8)

if __name__ == '__main__':
    main()