# face_detector.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as T

from FDDataset import FaceDetectionDataset
from FBCNN import FaceBoxCNN
from bbox_visualizer import BBoxVisualizer
import numpy as np
from PIL import Image

class FaceDetector:
    def __init__(
        self,
        train_dir=None,
        train_label=None,
        val_dir=None,
        val_label=None,
        checkpoint_path="checkpoints/data.pth",
        batch_size=32,
        lr=0.001
    ):
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.transform = T.Compose([T.Resize((64, 64)), T.ToTensor()])

        # Load datasets only if both image and label dirs are provided
        self.train_dataset = None
        self.train_loader = None
        if train_dir and train_label:
            self.train_dataset = FaceDetectionDataset(img_dir=train_dir, label_dir=train_label, transform=self.transform)
            self.train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True)

        self.val_dataset = None
        self.val_loader = None
        if val_dir and val_label:
            self.val_dataset = FaceDetectionDataset(img_dir=val_dir, label_dir=val_label, transform=self.transform)
            self.val_loader = DataLoader(self.val_dataset, batch_size=batch_size, shuffle=False)

        self.model = FaceBoxCNN()
        if torch.cuda.device_count() > 1:
            self.model = nn.DataParallel(self.model)
        self.model = self.model.to(self.device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = self.yolo_style_loss
        self.checkpoint_path = checkpoint_path
        os.makedirs(os.path.dirname(self.checkpoint_path), exist_ok=True)
        self.visualizer = BBoxVisualizer(save_dir="results")

        self.load_checkpoint()

    def yolo_style_loss(self, pred, target):
        class_loss = F.binary_cross_entropy(pred[:, 0], target[:, 0])
        bbox_loss = F.mse_loss(pred[:, 1:], target[:, 1:])
        return class_loss + bbox_loss

    def load_checkpoint(self):
        if os.path.exists(self.checkpoint_path):
            checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
            state_dict = checkpoint['model_state_dict']

            # Check if model is wrapped with DataParallel
            model_is_parallel = isinstance(self.model, nn.DataParallel)
            ckpt_is_parallel = list(state_dict.keys())[0].startswith('module.')

            # Fix mismatch between model and checkpoint
            if model_is_parallel and not ckpt_is_parallel:
                # Wrap checkpoint keys with 'module.'
                from collections import OrderedDict
                new_state_dict = OrderedDict()
                for k, v in state_dict.items():
                    new_state_dict[f"module.{k}"] = v
                state_dict = new_state_dict

            elif not model_is_parallel and ckpt_is_parallel:
                # Strip 'module.' from checkpoint keys
                from collections import OrderedDict
                new_state_dict = OrderedDict()
                for k, v in state_dict.items():
                    new_state_dict[k.replace("module.", "")] = v
                state_dict = new_state_dict

            self.model.load_state_dict(state_dict)
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print(f"✅ Loaded checkpoint from {self.checkpoint_path}")

    def save_checkpoint(self, epoch, loss):
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
        }, self.checkpoint_path)
        print(f"Checkpoint saved to {self.checkpoint_path}")

    def train(self, num_epochs=10):
        if self.train_loader is None:
            print("Training data not available.")
            return
        for epoch in range(num_epochs):
            self.model.train()
            running_loss = 0.0
            for images, targets in self.train_loader:
                images, targets = images.to(self.device), targets.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()

            avg_loss = running_loss / len(self.train_loader)
            print(f"[Epoch {epoch + 1}] Training Loss: {avg_loss:.4f}")
            self.save_checkpoint(epoch + 1, avg_loss)

    def evaluate(self):
        self.model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for images, targets in self.val_loader:
                images, targets = images.to(self.device), targets.to(self.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, targets)
                test_loss += loss.item()
        avg_loss = test_loss / len(self.val_loader)
        print(f"[Eval] Avg Loss (BCE + MSE): {avg_loss:.4f}")
        return avg_loss

    def save_marked_face(self, img_path):
        result = self.predict(img_path)
        if result is None:
            return

        x1, y1, x2, y2, conf, orig_image, orig_w, orig_h = result

        print(f"Original image size: {orig_w}x{orig_h}")
        print(f"Predicted box: ({x1:.2f}, {y1:.2f}) to ({x2:.2f}, {y2:.2f})")

        orig_np = np.array(orig_image)
        box_array = np.array([[x1, y1, x2, y2, conf]])

        filename = os.path.basename(img_path)
        self.visualizer.draw_boxes(orig_np, box_array, filename, normalized=False, yolo_format=False)
        print(f"Saved prediction as {filename} in 'results/'")

        row = [filename, round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2), round(conf, 4)]
        print("Logged:", row)


    def predict(self, img_path):
        self.model.eval()

        orig_image = Image.open(img_path).convert("RGB")
        image = self.transform(orig_image)
        image_tensor = image.unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(image_tensor).squeeze().cpu().numpy()
            print(f"output: {output}")

        if output[0] > 0.5:
            x_center, y_center, w, h, conf = *output[1:], output[0]
            orig_w, orig_h = orig_image.size

            # Convert normalized bbox to pixel coordinates
            x1 = (x_center - w / 2) * orig_w
            y1 = (y_center - h / 2) * orig_h
            x2 = (x_center + w / 2) * orig_w
            y2 = (y_center + h / 2) * orig_h

            return (x1, y1, x2, y2, conf, orig_image, orig_w, orig_h)
        else:
            print("Low confidence. No prediction visualized.")
            return None

    def resnet_tensor(self, img_path):
        result = self.predict(img_path)
        if result is None:
            return None

        x1, y1, x2, y2, _, orig_image, _, _ = result

        # Clamp coordinates to image bounds
        x1 = max(0, int(x1))
        y1 = max(0, int(y1))
        x2 = min(orig_image.width, int(x2))
        y2 = min(orig_image.height, int(y2))

        cropped_face = orig_image.crop((x1, y1, x2, y2))

        face_tensor = self.resnet_transform(cropped_face).unsqueeze(0).to(self.device)
        return face_tensor

    def detect_from_camera(self, confidence_threshold=0.5):
        self.model.eval()
        cap = cv2.VideoCapture(0)  # 0 for default webcam

        if not cap.isOpened():
            print("Error: Could not access the camera.")
            return

        print("Starting real-time detection. Press 'q' to quit.")
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame.")
                break

            orig_h, orig_w, _ = frame.shape
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image_pil = Image.fromarray(image)
            image_tensor = self.transform(image_pil).unsqueeze(0).to(self.device)

            with torch.no_grad():
                output = self.model(image_tensor).squeeze().cpu().numpy()

            if output[0] > confidence_threshold:
                x_center, y_center, w, h = output[1:]
                x1 = int((x_center - w / 2) * orig_w)
                y1 = int((y_center - h / 2) * orig_h)
                x2 = int((x_center + w / 2) * orig_w)
                y2 = int((y_center + h / 2) * orig_h)

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"Conf: {output[0]:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            cv2.imshow("Face Detection (Press 'q' to exit)", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
