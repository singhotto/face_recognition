from FaceDetector import FaceDetector
from FaceRecognizer import FaceRecognizer
import random
# import os
# from PIL import Image

# input_dir = "otto_images"
# output_dir = "dataset/rec/Faces"

# detector = FaceDetector(
#     train_dir="./dataset/detect/images/train",
#     train_label="./dataset/detect/labels/train",
#     val_dir="./dataset/detect/images/val",
#     val_label="./dataset/detect/labels/val"
# )

# detector.train(100)

# for idx in random.sample(range(0, 1400), 100):
#     detector.predict_one(idx=idx)

detector = FaceDetector()

# Create output directory if it doesn't exist
# os.makedirs(output_dir, exist_ok=True)

# Process each image
# for filename in os.listdir(input_dir):
#     if filename.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
#         input_path = os.path.join(input_dir, filename)
#         try:
#             img = Image.open(input_path).convert("RGB")
#             face = detector.get_face(img)
#             if face:
#                 output_path = os.path.join(output_dir, f"{filename}")
#                 face.save(output_path)
#                 print(f"Saved: {output_path}")
#             else:
#                 print(f"No face detected in {filename}")
#         except Exception as e:
#             print(f"Error processing {filename}: {e}")

recognizer = FaceRecognizer(
    img_dir="dataset/rec/Faces"
    # face_detector=detector
)

# recognizer.train(epochs=1000)
recognizer.build_reference_embeddings()

# val = recognizer.predict('otto_images/otto_1.jpeg')
# print(val)

# recognizer.detect_from_camera()
