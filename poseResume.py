from ultralytics import YOLO

def main():

    model = YOLO("C:/Users/Nyle/Documents/ResumeProjects/ComputerVisionBreakout/runs/pose/train9/weights/last.pt")

    model.train(resume = True)

if __name__ == '__main__':
    main()