import os
import cv2
import numpy as np
from glob import glob
from skimage.measure import regionprops, label
from skimage.feature import canny
from scipy.fft import fft
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_distances
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_distances

import tkinter as tk
from tkinter import filedialog, Frame
from PIL import Image, ImageTk

# Định nghĩa hàm segment_image
def segment_image(img, bg_threshold=240):
    mask = np.min(img, axis=2) < bg_threshold
    return mask.astype(np.uint8)

# Định nghĩa hàm extract_color_histogram
def extract_color_histogram(img, mask, bins=(8,8,8)):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0,1,2], mask, bins, [0,180,0,256,0,256])
    hist = cv2.normalize(hist, hist).flatten()
    return hist

# Định nghĩa hàm extract_shape_features
def extract_shape_features(mask):
    labels_img = label(mask)
    props = regionprops(labels_img)
    if not props:
        return [0] + [0]*7
    prop = max(props, key=lambda x: x.area)
    ecc = prop.minor_axis_length / (prop.major_axis_length + 1e-5)
    moments = cv2.moments(mask.astype(np.uint8))
    hu = cv2.HuMoments(moments).flatten()
    hu = -np.sign(hu) * np.log10(np.abs(hu) + 1e-10)
    return [ecc] + hu.tolist()

def extract_fourier_descriptor(mask, num_coeffs=10):
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return [0]*num_coeffs
    cnt = max(contours, key=cv2.contourArea)
    pts = cnt.squeeze()
    complex_pts = pts[:,0] + 1j*pts[:,1]
    coeffs = fft(complex_pts)
    coeffs /= np.abs(coeffs[1])
    desc = np.abs(coeffs[:num_coeffs])
    return desc.tolist()

# Định nghĩa hàm extract_grid_code
def extract_grid_code(mask, grid_size=(16,16), threshold=0.15):
    h,w = mask.shape
    gh,gw = grid_size
    cell_h,cell_w = h//gh, w//gw
    code = []
    for i in range(gh):
        for j in range(gw):
            cell = mask[i*cell_h:(i+1)*cell_h, j*cell_w:(j+1)*cell_w]
            frac = cell.sum()/(cell_h*cell_w)
            code.append(int(frac > threshold))
    return code

# Định nghĩa hàm extract_edge_histogram
def extract_edge_histogram(img, mask, grid_size=(4,4)):
    edges = canny(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), sigma=2)
    edges = edges & mask.astype(bool)
    h,w = edges.shape
    gh,gw = grid_size
    cell_h,cell_w = h//gh, w//gw
    hist = []
    for i in range(gh):
        for j in range(gw):
            cell = edges[i*cell_h:(i+1)*cell_h, j*cell_w:(j+1)*cell_w]
            hist.append(cell.sum())
    hist = np.array(hist, dtype=float)
    if hist.max()>0:
        hist /= hist.max()
    return hist.tolist()

def extract_features(img_path):
    img = cv2.imread(img_path)
    mask = segment_image(img)
    color = extract_color_histogram(img, mask)
    shape = extract_shape_features(mask)
    fourier = extract_fourier_descriptor(mask)
    grid = extract_grid_code(mask)
    edge = extract_edge_histogram(img, mask)
    feat = np.hstack([color, shape, fourier, grid, edge])
    return feat

# Hàm tìm kiếm
def find_similar(query_path, k=3):
    qf = extract_features(query_path)
    qf_norm = scaler.transform([qf])
    dists = cosine_distances(qf_norm, features_norm).flatten()
    idx = np.argsort(dists)[:k]
    return [(image_paths[i], dists[i]) for i in idx]

def load_image(path, size=(200, 200)):
    img = Image.open(path).convert("RGB")
    img = img.resize(size)
    return ImageTk.PhotoImage(img)


def show_results(query_path):
    # Hiển thị ảnh truy vấn
    img = load_image(query_path)
    query_label.config(image=img)
    query_label.image = img
    query_title.config(text=os.path.basename(query_path))

    # Hiển thị ảnh kết quả
    results = find_similar(query_path)
    for i in range(3):
        if i < len(results):
            img_path, dist = results[i]
            img = load_image(img_path)
            result_labels[i].config(image=img)
            result_labels[i].image = img
            result_titles[i].config(text=f"{os.path.basename(img_path)}\nCosine distance: {dist:.4f}")


def browse_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.png *.jpeg")])
    if file_path:
        show_results(file_path)            

dataset_dir = 'DataLoc2'
image_paths = glob(os.path.join(dataset_dir, '*', '*.jpg'))
features = []
labels = []

for p in image_paths[:500]:
    features.append(extract_features(p))
    labels.append(os.path.basename(os.path.dirname(p)))

features = np.vstack(features)
scaler = MinMaxScaler()
features_norm = scaler.fit_transform(features)


root = tk.Tk()
root.title("🔍 Image Similarity Search")
root.configure(bg="#f0f0f0")
root.geometry("900x700")

# Nút chọn ảnh truy vấn
browse_btn = tk.Button(root, text="📁 Chọn ảnh truy vấn", font=("Helvetica", 12), bg="#4CAF50", fg="white", padx=20, pady=5, command=browse_image)
browse_btn.pack(pady=10)

# Khung chứa ảnh truy vấn
query_frame = Frame(root, bg="#f0f0f0")
query_frame.pack(pady=20)

query_title = tk.Label(query_frame, text="Ảnh truy vấn", font=("Helvetica", 14, "bold"), bg="#f0f0f0")
query_title.pack()

query_label = tk.Label(query_frame, bg="#dddddd", width=200, height=200)
query_label.pack(pady=10)

# Khung chứa ảnh kết quả
results_title = tk.Label(root, text="🎯 Ảnh tương đồng nhất", font=("Helvetica", 16, "bold"), bg="#f0f0f0", fg="#444")
results_title.pack(pady=10)

results_frame = Frame(root, bg="#f0f0f0")
results_frame.pack(pady=10)

result_labels = []
result_titles = []

for i in range(3):
    sub_frame = Frame(results_frame, bg="#f0f0f0", padx=15)
    sub_frame.grid(row=0, column=i)

    lbl_title = tk.Label(sub_frame, text=f"Similar #{i+1}", font=("Helvetica", 12, "bold"), bg="#f0f0f0")
    lbl_title.pack()

    lbl = tk.Label(sub_frame, bg="#dddddd", width=200, height=200)
    lbl.pack(pady=10)

    result_titles.append(lbl_title)
    result_labels.append(lbl)

root.mainloop()