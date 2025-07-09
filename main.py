from FaceDetector import FaceDetector
#from FaceRecognizer import FaceRecognizer
import random

detector = FaceDetector(
    train_dir="./dataset/detect/images/train",
    train_label="./dataset/detect/labels/train",
    val_dir="./dataset/detect/images/val",
    val_label="./dataset/detect/labels/val"
)

detector.train(100)

# for idx in random.sample(range(0, 1400), 100):
#     detector.predict_one(idx=idx)

# detector = FaceDetector()

# recognizer = FaceRecognizer(
#     csv_path="dataset/rec/Dataset.csv",
#     img_dir="dataset/rec/Faces",
#     batch_size=32,
#     lr=1e-3,
#     face_detector=detector
# )

# recognizer.train(epochs=100)
