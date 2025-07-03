from FaceDetector import FaceDetector
from FaceRecognizer import FaceRecognizer
from resnet.resnet import ResNet 
import random

# detector = FaceDetector(
#     train_dir="./dataset/detect/images/train",
#     train_label="./dataset/detect/labels/train",
#     val_dir="./dataset/detect/images/val",
#     val_label="./dataset/detect/labels/val"
# )

# detector.train(100)

# for idx in random.sample(range(0, 1400), 100):
#     detector.predict_one(idx=idx)

# detector = FaceDetector()

resnet = ResNet(n=2)
recognizer = FaceRecognizer(
    csv_path="RecLabels.csv",
    resnet_model=resnet,
    batch_size=32,
    lr=1e-3
)

res = recognizer.predict("images/Vijay Deverakonda_103.jpg")
print(res)

# recognizer.train(epochs=10000)
