from __future__ import division
import imageio as io
import numpy as np
from math import floor
from os import mkdir
from os.path import join, isdir
from imageio import imread, imwrite
import tifffile as tiff
import astra

# Configuration.
num_of_projections = 72
angles = np.linspace(-66/180 * np.pi, 78/180 * np.pi, num=num_of_projections, endpoint=False)
print(angles*180/np.pi)

output_dir = 'reconstruction_3_all_8bits_only_SIRT'

# Load projections.

projection_data = '3_all_8bits.tif'
sino_full = io.volread(projection_data)


sino_angles = np.size(sino_full, 0)
sino_rows = np.size(sino_full, 1)
sino_cols = np.size(sino_full, 2)
# vol_geom = np.zeros((sino_rows, num_of_projections, sino_cols)) #创建投影的数据体积

sino_full = np.rot90(sino_full, 1, (0,1))
print("sino_full 的形状: ", sino_full.shape)

# 生成投影仪
proj_geom = astra.create_proj_geom('parallel3d', 1.0, 1.0, sino_rows, sino_cols, angles)
projections_id = astra.data3d.create('-sino', proj_geom, sino_full)

# 生成存储重构数据的几何体
rec_id = astra.creators.create_vol_geom(sino_cols, sino_cols,
                                          sino_rows)

reconstruction_id = astra.data3d.create('-vol', rec_id, data=0)

cfg = astra.astra_dict('SIRT3D_CUDA')
cfg['ProjectionDataId'] = projections_id
cfg['ReconstructionDataId'] = reconstruction_id
alg_id = astra.algorithm.create(cfg)
astra.algorithm.run(alg_id, 20)
reconstruction = astra.data3d.get(reconstruction_id)

# 旋转图像，from 俯视图 to 侧视图
reconstruction = np.rot90(reconstruction)

# Limit and scale reconstruction.
reconstruction[reconstruction < 0] = 0
reconstruction /= np.max(reconstruction)
reconstruction = np.round(reconstruction * 255).astype(np.uint8)

# Save reconstruction. 检查输出目录，如果输出目录不存在，则创建它
if not isdir(output_dir):
    mkdir(output_dir)

# Prepare the full path for the output TIFF file，设置输出文件的完整路径
output_file_path = join(output_dir, 'reconstruction_stack.tif')


# Create an empty list to store image data，创建一个图像stack
stack = []

# Iterate through each slice, flip it, and add to the list 处理和添加图像
for i in range(sino_cols):
    im = reconstruction[i, :, :]
    im_flipped = np.flipud(im)
    stack.append(im_flipped)

# Save all images in the stack to a single multi-page TIFF file 保存tiff文件
io.mimwrite(output_file_path, stack, format='TIFF')
# Cleanup.
astra.algorithm.delete(alg_id)
astra.data3d.delete(reconstruction_id)
astra.data3d.delete(projections_id)







