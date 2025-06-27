import os
import cv2
import numpy as np
from glob import glob
from skimage.measure import regionprops, label
from skimage.feature import canny
from scipy.fft import fft
from sklearn.preprocessing import MinMaxScaler
import pickle

print("Bắt đầu quá trình trích xuất đặc trưng...")
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
# --- KẾT THÚC CÁC HÀM TRÍCH XUẤT ---

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

# --- QUÁ TRÌNH TRÍCH XUẤT VÀ LƯU DATABASE ---

dataset_dir = 'DataLoc2'
image_paths = glob(os.path.join(dataset_dir, '*', '*.jpg'))

if not image_paths:
    print(f"Lỗi: Không tìm thấy tệp .jpg nào trong '{dataset_dir}'.")
else:
    print(f"Đang xử lý {len(image_paths)} hình ảnh...")
    
    # Khởi tạo các danh sách để lưu từng loại đặc trưng
    all_features = {
        "color": [], "shape": [], "fourier": [], "grid": [], "edge": []
    }
    valid_paths = []
    
    for path in image_paths:
        features = extract_all_features(path)
        if features:
            for key in all_features:
                all_features[key].append(features[key])
            valid_paths.append(path)

    # Chuyển đổi thành các mảng NumPy
    for key in all_features:
        all_features[key] = np.array(all_features[key])
        
    # Tạo đặc trưng tổng hợp
    combined_features = np.hstack([
        all_features["color"], all_features["shape"], all_features["fourier"],
        all_features["grid"], all_features["edge"]
    ])
    
    # Chuẩn hóa trên toàn bộ đặc trưng tổng hợp
    scaler = MinMaxScaler()
    norm_combined_features = scaler.fit_transform(combined_features)
    
    # Lưu các tệp database
    print("Đang lưu database...")
    
    # 1. Lưu đặc trưng tổng hợp
    np.save('db_combined.npy', norm_combined_features)
    
    # 2. Lưu 5 đặc trưng riêng (đã được chuẩn hóa theo scaler chung)
    # Chúng ta cần cắt mảng tổng hợp đã chuẩn hóa ra từng phần
    indices = [
        all_features["color"].shape[1],
        all_features["shape"].shape[1],
        all_features["fourier"].shape[1],
        all_features["grid"].shape[1],
        all_features["edge"].shape[1]
    ]
    
    current_idx = 0
    # Cắt và lưu đặc trưng màu
    np.save('db_color.npy', norm_combined_features[:, current_idx : current_idx + indices[0]])
    current_idx += indices[0]
    # Cắt và lưu đặc trưng hình dạng
    np.save('db_shape.npy', norm_combined_features[:, current_idx : current_idx + indices[1]])
    current_idx += indices[1]
    # ... và tiếp tục cho các đặc trưng còn lại
    np.save('db_fourier.npy', norm_combined_features[:, current_idx : current_idx + indices[2]])
    current_idx += indices[2]
    np.save('db_grid.npy', norm_combined_features[:, current_idx : current_idx + indices[3]])
    current_idx += indices[3]
    np.save('db_edge.npy', norm_combined_features[:, current_idx : current_idx + indices[4]])

    # 3. Lưu các tệp phụ trợ
    with open('paths_db.pkl', 'wb') as f:
        pickle.dump(valid_paths, f)
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    with open('indices.pkl', 'wb') as f:
        pickle.dump(indices, f)

    print("Hoàn thành!")