import os
import cv2
import numpy as np
import torch
from typing import Union, List, Tuple

class BBoxVisualizer:
    def __init__(self, save_dir: str = "output_images"):
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

    def draw_boxes(
        self,
        image_input: Union[torch.Tensor, np.ndarray],
        bboxes: List[List[float]],
        filename: str,
        color: Tuple[int, int, int] = (0, 255, 0),
        thickness: int = 2,
        normalized: bool = False,
        yolo_format: bool = False,
    ):
        """
        Draws bounding boxes on the image and saves it.

        Args:
            image_input: torch.Tensor (CHW) or np.ndarray (HWC)
            bboxes: List of boxes [x, y, w, h, conf] or [x1, y1, x2, y2, conf]
            normalized: Whether coordinates are normalized (0-1)
            yolo_format: Whether boxes are in YOLO format (x_center, y_center, w, h)
        """
        if isinstance(image_input, torch.Tensor):
            if image_input.dim() == 4:
                image_input = image_input.squeeze(0)
            image = image_input.permute(1, 2, 0).cpu().numpy()
            image = (image * 255).clip(0, 255).astype(np.uint8)
        else:
            image = image_input.copy()

        image = np.ascontiguousarray(image)
        h, w = image.shape[:2]

        for box in bboxes:
            if yolo_format:
                x_center, y_center, bw, bh, conf = box
                if normalized:
                    x_center *= w
                    y_center *= h
                    bw *= w
                    bh *= h
                x1 = int(x_center - bw / 2)
                y1 = int(y_center - bh / 2)
                x2 = int(x_center + bw / 2)
                y2 = int(y_center + bh / 2)
            else:
                x1, y1, x2, y2, conf = box
                if normalized:
                    x1 *= w
                    y1 *= h
                    x2 *= w
                    y2 *= h
                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

            cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)

        save_path = os.path.join(self.save_dir, filename)
        cv2.imwrite(save_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        print(f"[Saved] {save_path}")
