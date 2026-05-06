import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader

# Takes Grayscale Image and returns the Local Binary Pattern Image
def get_lbp(gray):
   h, w  = gray.shape

   # Initialising lbp image
   lbp = np.zeros((h, w), dtype=np.uint8)

   for i in range(1, h-1):
       for j in range(1, w-1):
           center = gray[i, j]
           binary = 0
           # Calculating lbp code for pixel at [i, j]
           binary |= (gray[i-1, j-1] >= center) << 7
           binary |= (gray[i-1, j  ] >= center) << 6
           binary |= (gray[i-1, j+1] >= center) << 5
           binary |= (gray[i  , j+1] >= center) << 4
           binary |= (gray[i+1, j+1] >= center) << 3
           binary |= (gray[i+1, j  ] >= center) << 2
           binary |= (gray[i+1, j-1] >= center) << 1
           binary |= (gray[i  , j-1] >= center) << 0

           lbp[i, j] = binary

   return lbp


# Takes Grayscale Image and returns its Canny Edge Map
def get_canny(gray):
    canny = cv2.Canny(gray, 100, 200)
    return canny


# Takes RGB Image and returns the 6 color features
def extract_color_features(img):

    # Convert RGB to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)

    # Mean Hue (Feature 1)
    mean_h = np.mean(h)

    # Standard Deviation of Hue (Feature 2)
    std_h = np.std(h)

    # Mean Saturation (Feature 3)
    mean_s = np.mean(s)

    # Standard Deviation of Saturation (Feature 4)
    std_s = np.std(s)

    # Mean Value (Feature 5)
    mean_v = np.mean(v)

    # Standard Deviation of Value (Feature 6)
    std_v = np.std(v)

    return np.array([mean_h, std_h, mean_s, std_s, mean_v, std_v])


# Takes Grayscale Image and returns the 6 shape features
def extract_shape_features(gray):

    # Since the fruits have white background, we will use binary mask for extracting shape features
    _, mask = cv2.threshold(gray, 245, 255, cv2.THRESH_BINARY_INV)  # Creating binary mask

    # Contour Detection in the binary mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # If no contour is detected, return all features as zero
    if not contours:
        return np.zeros(6)

    # Contour with largest Area
    cnt = max(contours, key=cv2.contourArea)

    # Contour Area
    area = cv2.contourArea(cnt)

    # Perimeter of Contour
    perimeter = cv2.arcLength(cnt, True)

    # Image Area
    img_area = gray.shape[0] * gray.shape[1]

    # Area Ratio (Feature 1)
    area_ratio = area / img_area

    # Bounding Box
    x, y, w, h = cv2.boundingRect(cnt)

    # Aspect Ratio (Feature 2)
    aspect_ratio = w / h if h > 0 else 0

    # Computes Convex Hull
    hull = cv2.convexHull(cnt)

    # Convex Hull Area
    hull_area = cv2.contourArea(hull)

    # Solidity (Feature 3)
    solidity = area / hull_area if hull_area > 0 else 0

    # Circularity (Feature 4)
    circularity = 4 * np.pi * area / (perimeter ** 2 + 1e-10)

    # Hu Moments
    moments = cv2.moments(cnt)
    hu = cv2.HuMoments(moments).flatten()

    # Log-scale for numerical stability
    # Hu_1 (Feature 5)
    hu_1 = -np.sign(hu[0]) * np.log10(np.abs(hu[0]) + 1e-10)

    # Hu_2 (Feature 6)
    hu_2 = -np.sign(hu[1]) * np.log10(np.abs(hu[1]) + 1e-10)
    # Used offset value of 10^(-10) in order to avoid exact zero inside the logarithm,otherwise it might blow up

    return np.array([area_ratio, aspect_ratio, solidity, circularity, hu_1, hu_2])


# Dataset Loader
class Fruit360_DataLoader(Dataset):
    def __init__(self, path):

        # To store (img_path,label) tuples
        self.samples = []

        # Mapping class names to integer label
        self.class_to_idx = {}

        # Get sorted list of class folders
        classes = sorted(os.listdir(path))

        # Assign index to each class

        for idx, cls in enumerate(classes):
            self.class_to_idx[cls] = idx
            cls_path = os.path.join(path, cls)

            # Iterating over all image in the class folder
            for img in os.listdir(cls_path):
                img_path = os.path.join(cls_path, img)
                self.samples.append((img_path, idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        # Get image path and corresponding label
        img_path, label = self.samples[idx]

        # Read image and conversion to RGB format
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Resize for consistent input dimensions (images are already 100x100)
        img = cv2.resize(img, (100, 100))

        # Conversion to Grayscale Image
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        # Get LBP features
        lbp = get_lbp(gray)

        # Get Canny Edge Map
        canny = get_canny(gray)

        # Extract 6 color features
        color_features = extract_color_features(img)

        # Extract 6 shape features
        shape_features = extract_shape_features(gray)

        # PyTorch expects image tensors of shape (C, H, W)
        img = torch.tensor(img).permute(2, 0, 1).float() / 255.0
        lbp = torch.tensor(lbp).unsqueeze(0).float() / 255.0
        canny = torch.tensor(canny).unsqueeze(0).float() / 255.0

        # Conversion to tensor
        color_features = torch.tensor(color_features).float()
        shape_features = torch.tensor(shape_features).float()
        label = torch.tensor(label, dtype=torch.long)

        # Return dictionary containing all modalities and label
        return {
            "rgb": img,                       # RGB image tensor (3 x H x W)
            "lbp": lbp,                       # LBP texture image (1 x H x W)
            "canny": canny,                   # Canny edge image (1 x H x W)
            "color_features": color_features, # Extracted color features (6,)
            "shape_features": shape_features, # Extracted shape features (6,)
            "label": label                    # Class label
        }
    

def data_loader(training, test):
    train_dataset = Fruit360_DataLoader(training)
    test_dataset  = Fruit360_DataLoader(test)

    train_loader = DataLoader(
        train_dataset,
        batch_size=16,      # ← small, only 8GB RAM shared with OS
        shuffle=True,
        num_workers=2,      # ← Ryzen 5 5600H has 6 cores, use 2 safely
        pin_memory=False,   # ← False, no GPU
        persistent_workers=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=16,
        shuffle=False,
        num_workers=2,
        pin_memory=False,
        persistent_workers=True
    )

    return train_loader, test_loader