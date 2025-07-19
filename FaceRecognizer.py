import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn.functional as F
from PIL import Image
import pickle
from collections import OrderedDict
from resnet.resnet import ResNet
import cv2

class FaceRecognizer:
    def __init__(self, img_dir=None, embedding_store_path="embeddings/embeddings.pkl", 
                 batch_size=32, lr=1e-3, face_detector=None):
        
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.img_dir = img_dir
        self.face_detector = face_detector
        self.embedding_store_path = embedding_store_path

        self.transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3)
        ])

        self.model = ResNet(n=2).to(self.device)
        if torch.cuda.device_count() > 1:
            self.model = nn.DataParallel(self.model)

        self.loss_fn = nn.TripletMarginLoss(margin=1.0, p=2)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

        if img_dir:
            from TripletFRDataset import TripletFRDataset  # dynamic import
            self.dataset = TripletFRDataset(img_dir, transform=self.transform)
            self.dataloader = DataLoader(self.dataset, batch_size=batch_size, shuffle=True)

        self.load_checkpoint()

        # Load stored reference embeddings if available
        self._load_embeddings()

    def train(self, epochs=10):
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            for anchor, positive, negative in self.dataloader:
                anchor = anchor.to(self.device)
                positive = positive.to(self.device)
                negative = negative.to(self.device)

                anchor_emb = self.model(anchor)
                positive_emb = self.model(positive)
                negative_emb = self.model(negative)

                loss = self.loss_fn(anchor_emb, positive_emb, negative_emb)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

            print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss/len(self.dataloader):.4f}")
            self.save_checkpoint(epoch+1, total_loss)

        self.build_reference_embeddings()
        
    def build_reference_embeddings(self):
        """Save mean embeddings per label for inference."""
        from FRDataset import FRDataset
        dataset = FRDataset(self.img_dir, transform=self.transform)
        loader = DataLoader(dataset, batch_size=1, shuffle=False)

        label_embeddings = {}

        self.model.eval()
        with torch.no_grad():
            for image, label in loader:
                image = image.to(self.device)
                emb = self.model(image).cpu()

                label = label.item()
                if label not in label_embeddings:
                    label_embeddings[label] = []
                label_embeddings[label].append(emb)

        # Average embeddings per label
        for label in label_embeddings:
            label_embeddings[label] = torch.mean(torch.stack(label_embeddings[label]), dim=0)

        # Save both embeddings and label_map
        to_save = {
            "embeddings": label_embeddings,
            "label_map": {v: k for k, v in dataset.label_map.items()}  # reverse map
        }

        with open(self.embedding_store_path, 'wb') as f:
            pickle.dump(to_save, f)
        
        print(f"✅ Stored {len(label_embeddings)} label embeddings.")

        self.reference_embeddings = label_embeddings
        self.label_to_name = to_save["label_map"]

    def predict_rgb(self, img, threshold=0.6):
        img = self.transform(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            emb = self.model(img).cpu()
        if not self.reference_embeddings:
            raise ValueError("❌ No stored embeddings. Please train or load reference embeddings.")

        # Flatten emb once
        emb = emb.view(-1)  # Now shape: [64]

        similarities = {
            label: F.cosine_similarity(emb, ref_emb.view(-1), dim=0).item()
            for label, ref_emb in self.reference_embeddings.items()
        }
        
        # Get best match
        best_label, best_score = max(similarities.items(), key=lambda x: x[1])

        if best_score > threshold:
            name = self.label_to_name.get(best_label, f"Label {best_label}")
            return name, best_score
        else:
            return "Unknown", best_score

    def predict(self, img_path, threshold=0.6):
        self.model.eval()
        img = Image.open(img_path).convert("RGB")
        if self.face_detector:
            img = self.face_detector.get_face(img)  # assume it crops the face

        return self.predict_rgb(img)


    def save_checkpoint(self, epoch, loss, path='checkpoints/face_recognizer.pth'):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
        }, path)
        print(f"💾 Checkpoint saved to {path}")

    def load_checkpoint(self, path='checkpoints/face_recognizer.pth'):
        if os.path.exists(path):
            checkpoint = torch.load(path, map_location=self.device)
            state_dict = checkpoint['model_state_dict']

            # Fix possible DataParallel mismatch
            model_is_parallel = isinstance(self.model, nn.DataParallel)
            ckpt_is_parallel = list(state_dict.keys())[0].startswith('module.')
            if model_is_parallel != ckpt_is_parallel:
                new_state_dict = OrderedDict()
                for k, v in state_dict.items():
                    key = f"module.{k}" if model_is_parallel else k.replace("module.", "")
                    new_state_dict[key] = v
                state_dict = new_state_dict

            self.model.load_state_dict(state_dict)
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print(f"✅ Loaded checkpoint from {path}")
        else:
            print(f"⚠️ Checkpoint not found at {path}")

    def _load_embeddings(self):
        """Load precomputed embeddings and label names."""
        with open(self.embedding_store_path, 'rb') as f:
            data = pickle.load(f)
            self.reference_embeddings = data["embeddings"]
            self.label_to_name = data["label_map"]  # maps int → name
        print(f"✅ Loaded {len(self.reference_embeddings)} reference embeddings.")

    def detect_from_camera(self):
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
            image = Image.fromarray(image)

            result = self.face_detector.predict_rgb(image)

            if result is not None:
                x1, y1, x2, y2, conf, _, _, _ = result
                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

                label, conf = self.predict_rgb(image)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"Conf: {conf:.2f} Label: {label}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Display frame
            cv2.imshow("Face Recognizer", frame)

            # ❗️ Allow pressing 'q' to quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Quitting...")
                break

        cap.release()
        cv2.destroyAllWindows()