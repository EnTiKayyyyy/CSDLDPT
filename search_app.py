# search_app.py

import os
import cv2
import numpy as np
from sklearn.metrics.pairwise import cosine_distances
from scipy.spatial.distance import minkowski
import pickle
import tkinter as tk
from tkinter import filedialog, Frame, Label, Button, Radiobutton, StringVar
from PIL import Image, ImageTk

# --- Dán lại 6 hàm extract_* và hàm extract_all_features vào đây ---
# (Lấy từ các câu trả lời trước)
from skimage.measure import regionprops, label
from skimage.feature import canny
from scipy.fft import fft

def segment_image(img, bg_threshold=240):
    mask = np.min(img, axis=2) < bg_threshold
    return mask.astype(np.uint8)

def extract_color_histogram(img, mask, bins=(8,8,8)):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0,1,2], mask, bins, [0,180,0,256,0,256])
    hist = cv2.normalize(hist, hist).flatten()
    return hist

def extract_shape_features(mask):
    labels_img = label(mask)
    props = regionprops(labels_img)
    if not props: return [0] * 8
    prop = max(props, key=lambda x: x.area)
    ecc = prop.minor_axis_length / (prop.major_axis_length + 1e-5)
    moments = cv2.moments(mask.astype(np.uint8))
    hu = cv2.HuMoments(moments).flatten()
    hu = -np.sign(hu) * np.log10(np.abs(hu) + 1e-10)
    return [ecc] + hu.tolist()

def extract_fourier_descriptor(mask, num_coeffs=10):
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours: return [0]*num_coeffs
    cnt = max(contours, key=cv2.contourArea)
    pts = cnt.squeeze()
    if pts.ndim < 2: return [0] * num_coeffs
    complex_pts = pts[:,0] + 1j*pts[:,1]
    coeffs = fft(complex_pts)
    if np.abs(coeffs[1]) < 1e-5: return [0] * num_coeffs
    coeffs /= np.abs(coeffs[1])
    desc = np.abs(coeffs[:num_coeffs])
    return desc.tolist()

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
    if hist.max()>0: hist /= hist.max()
    return hist.tolist()
    
def extract_all_features(img_path):
    img = cv2.imread(img_path)
    if img is None:
        return None
    mask = segment_image(img)
    
    features = {
        "color": extract_color_histogram(img, mask),
        "shape": extract_shape_features(mask),
        "fourier": extract_fourier_descriptor(mask),
        "grid": extract_grid_code(mask),
        "edge": extract_edge_histogram(img, mask)
    }
    return features
# --- KẾT THÚC CÁC HÀM TRÍCH XUẤT ---


# --- TẢI DATABASE ---
DB_LOADED = False
try:
    print("Đang tải database...")
    db = {
        'combined': np.load('db_combined.npy'),
        'color': np.load('db_color.npy'),
        'shape': np.load('db_shape.npy'),
        'fourier': np.load('db_fourier.npy'),
        'grid': np.load('db_grid.npy'),
        'edge': np.load('db_edge.npy'),
    }
    with open('paths_db.pkl', 'rb') as f:
        image_paths = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('indices.pkl', 'rb') as f:
        indices = pickle.load(f)
    
    print("Tải database thành công.")
    DB_LOADED = True
except FileNotFoundError:
    print("Lỗi: Không tìm thấy tệp database. Vui lòng chạy 'feature_extractor.py' trước.")

# --- CÁC HÀM TÌM KIẾM ---
def get_query_feature(query_path, search_type):
    """Trích xuất, chuẩn hóa và cắt đặc trưng của ảnh truy vấn."""
    features = extract_all_features(query_path)
    if features is None:
        return None
    
    query_combined = np.hstack(list(features.values()))
    query_norm_combined = scaler.transform([query_combined])

    if search_type == 'combined':
        return query_norm_combined
    else:
        feature_keys = ['color', 'shape', 'fourier', 'grid', 'edge']
        key_index = feature_keys.index(search_type)
        start_idx = sum(indices[:key_index])
        end_idx = start_idx + indices[key_index]
        return query_norm_combined[:, start_idx:end_idx]

def find_similar_cosine(query_feature, db_features, k=3):
    """Tìm kiếm bằng Cosine Distance."""
    dists = cosine_distances(query_feature, db_features).flatten()
    idx = np.argsort(dists)[:k]
    return [(image_paths[i], dists[i]) for i in idx]

def find_similar_l3(query_feature, db_features, k=3):
    """Tìm kiếm bằng L3 Distance (Minkowski p=3)."""
    dists = np.array([minkowski(query_feature[0], feat, p=3) for feat in db_features])
    idx = np.argsort(dists)[:k]
    return [(image_paths[i], dists[i]) for i in idx]

# --- PHẦN GIAO DIỆN NGƯỜI DÙNG (GUI) ---
# Biến toàn cục để lưu đường dẫn ảnh đang truy vấn
current_query_path = None

def center_window(win, width, height):
    """Hàm để canh giữa cửa sổ trên màn hình."""
    screen_width = win.winfo_screenwidth()
    screen_height = win.winfo_screenheight()
    x = int((screen_width/2) - (width/2))
    y = int((screen_height/2) - (height/2))
    win.geometry(f'{width}x{height}+{x}+{y}')

def load_image(path, size=(200, 200)):
    img = Image.open(path).convert("RGB")
    img = img.resize(size)
    return ImageTk.PhotoImage(img)

def show_results(query_path):
    global current_query_path
    current_query_path = query_path # Lưu lại đường dẫn

    search_type = search_type_var.get()
    
    # Cập nhật tiêu đề
    title_text = f"Đặc trưng: {search_type.capitalize()}"
    cosine_row_title.config(text=f"📐 Kết quả theo Cosine - {title_text}")
    l3_row_title.config(text=f"📏 Kết quả theo L3 - {title_text}")

    # Hiển thị ảnh truy vấn
    img = load_image(query_path)
    query_label.config(image=img); query_label.image = img
    query_title.config(text=os.path.basename(query_path))

    # Lấy đặc trưng ảnh truy vấn
    query_feature = get_query_feature(query_path, search_type)
    if query_feature is None:
        print("Không thể xử lý ảnh truy vấn.")
        return

    # Lấy CSDL đặc trưng tương ứng
    db_features = db[search_type]

    # Tìm và hiển thị kết quả cho cả hai phương pháp
    cosine_results = find_similar_cosine(query_feature, db_features, k=3)
    l3_results = find_similar_l3(query_feature, db_features, k=3)

    # Hiển thị kết quả Cosine
    for i in range(3):
        if i < len(cosine_results):
            img_path, dist = cosine_results[i]
            img = load_image(img_path)
            cosine_labels[i].config(image=img); cosine_labels[i].image = img
            cosine_titles[i].config(text=f"{os.path.basename(img_path)}\nCosine: {dist:.4f}")
        else:
            cosine_labels[i].config(image=None); cosine_titles[i].config(text="")

    # Hiển thị kết quả L3
    for i in range(3):
        if i < len(l3_results):
            img_path, dist = l3_results[i]
            img = load_image(img_path)
            l3_labels[i].config(image=img); l3_labels[i].image = img
            l3_titles[i].config(text=f"{os.path.basename(img_path)}\nL3: {dist:.4f}")
        else:
            l3_labels[i].config(image=None); l3_titles[i].config(text="")

def browse_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.png *.jpeg")])
    if file_path:
        show_results(file_path)

def on_feature_select():
    """Hàm được gọi khi người dùng chọn Radiobutton, tự động reload kết quả."""
    if current_query_path:
        print(f"Reloading results for {os.path.basename(current_query_path)} with feature '{search_type_var.get()}'")
        show_results(current_query_path)

# --- CỬA SỔ CHÍNH ---
root = tk.Tk()
root.title("🔍 Image Similarity Search")
root.configure(bg="#f0f0f0")
APP_WIDTH = 800
APP_HEIGHT = 1050 # Giảm chiều cao cho hợp lý
center_window(root, APP_WIDTH, APP_HEIGHT)


if DB_LOADED:
    browse_btn = Button(root, text="📁 Chọn ảnh truy vấn", font=("Helvetica", 12), bg="#4CAF50", fg="white", padx=20, pady=5, command=browse_image)
    browse_btn.pack(pady=10)

    # --- Khung chọn loại đặc trưng ---
    options_frame = Frame(root, bg="#f0f0f0", pady=10)
    options_frame.pack()
    Label(options_frame, text="Chọn loại đặc trưng:", font=("Helvetica", 12, "bold"), bg="#f0f0f0").pack(side=tk.LEFT, padx=10)
    
    search_type_var = StringVar(value="combined")
    
    search_options = [
        ("Tổng hợp", "combined"), ("Màu sắc", "color"), ("Hình dạng", "shape"),
        ("Fourier", "fourier"), ("Lưới", "grid"), ("Cạnh", "edge")
    ]
    for text, value in search_options:
        # Thêm command='on_feature_select' vào Radiobutton
        Radiobutton(
            options_frame, text=text, variable=search_type_var, value=value, 
            font=("Helvetica", 11), bg="#f0f0f0", command=on_feature_select
        ).pack(side=tk.LEFT)

    
    query_frame = Frame(root, bg="#f0f0f0")
    query_frame.pack(pady=10)
    query_title = Label(query_frame, text="Ảnh truy vấn", font=("Helvetica", 14, "bold"), bg="#f0f0f0")
    query_title.pack()
    query_label = Label(query_frame, bg="#dddddd", width=200, height=200)
    query_label.pack(pady=10)

    # --- Khung kết quả ---
    # Hàng kết quả Cosine
    cosine_row_title = Label(root, text="📐 Kết quả theo Cosine", font=("Helvetica", 14, "bold"), bg="#f0f0f0", fg="#333")
    cosine_row_title.pack()
    cosine_frame = Frame(root, bg="#f0f0f0")
    cosine_frame.pack()
    cosine_labels, cosine_titles = [], []
    for i in range(3):
        sub_frame = Frame(cosine_frame, bg="#f0f0f0", padx=15)
        sub_frame.grid(row=0, column=i)
        lbl_title = Label(sub_frame, text="", font=("Helvetica", 11), bg="#f0f0f0")
        lbl_title.pack()
        lbl = Label(sub_frame, bg="#dddddd", width=200, height=200)
        lbl.pack(pady=5)
        cosine_titles.append(lbl_title)
        cosine_labels.append(lbl)

    # Hàng kết quả L3
    l3_row_title = Label(root, text="📏 Kết quả theo L3", font=("Helvetica", 14, "bold"), bg="#f0f0f0", fg="#333")
    l3_row_title.pack(pady=(20, 0))
    l3_frame = Frame(root, bg="#f0f0f0")
    l3_frame.pack()
    l3_labels, l3_titles = [], []
    for i in range(3):
        sub_frame = Frame(l3_frame, bg="#f0f0f0", padx=15)
        sub_frame.grid(row=0, column=i)
        lbl_title = Label(sub_frame, text="", font=("Helvetica", 11), bg="#f0f0f0")
        lbl_title.pack()
        lbl = Label(sub_frame, bg="#dddddd", width=200, height=200)
        lbl.pack(pady=5)
        l3_titles.append(lbl_title)
        l3_labels.append(lbl)
else:
    error_label = Label(root, text="Lỗi: Không tìm thấy tệp database.\nVui lòng chạy 'feature_extractor.py' trước.",
                        font=("Helvetica", 14), fg="red", bg="#f0f0f0")
    error_label.pack(pady=100)

root.mainloop()