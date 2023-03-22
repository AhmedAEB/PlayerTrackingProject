from ultralytics import YOLO

model = YOLO("./weights/SecondSeg.pt")

results = model("./data/moving2.mp4", save=True, save_txt=True, show=True, max_det=1, conf=0.8)