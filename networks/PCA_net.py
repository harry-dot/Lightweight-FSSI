import numpy as np
from sklearn.decomposition import PCA

file_path = '/data/ch/FSSV_feature/Lib/Lib7s/train/14/14-208-0000.npy'
file = np.load(file_path).reshape(-1,80)
print(file.shape)
X = np.array(
[[66, 64, 65, 65, 65],
 [65, 63, 63, 65, 64],
 [57, 58, 63, 59, 66],
 [67, 69, 65, 68, 64],
 [61, 61, 62, 62, 63],
 [64, 65, 63, 63, 63],
 [64, 63, 63, 63, 64],
 [63, 63, 63, 63, 63],
 [65, 64, 65, 66, 64],
 [67, 69, 69, 68, 67],
 [62, 63, 65, 64, 64],
 [68, 67, 65, 67, 65],
 [65, 65, 66, 65, 64],
 [62, 63, 64, 62, 66],
 [64, 66, 66, 65, 67]]
)
print(X.shape)
# n_components 指明了降到几维
pca = PCA(n_components = 20)

# 利用数据训练模型（即上述得出特征向量的过程）
pca.fit(file)

# 得出原始数据的降维后的结果；也可以以新的数据作为参数，得到降维结果。
print(pca.transform(file))
print(pca.transform(file).shape)
# 打印各主成分的方差占比
print(pca.explained_variance_ratio_)
