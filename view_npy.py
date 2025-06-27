import numpy as np
file_path = 'features_db.npy'
feature_data = np.load(file_path)
print(feature_data[:10])