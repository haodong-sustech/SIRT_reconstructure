import astra
import numpy as np
import imageio as io
import time
import tifffile as tiff
from math import floor
from tqdm.notebook import trange, tqdm
import matplotlib.pyplot as plt
import matplotlib
from skimage import exposure, img_as_float
from matplotlib.gridspec import GridSpec


def norm16(image, Min, Max, z):
    section = np.copy(image)
    section += Min  # make recon positive (if using TEM)
    section /= Max  # normalize to 0-1
    #section = np.round(section * 65535).astype(np.uint16)
    section = np.round(section * 255).astype(np.uint8)
    return section


def angles_list(start, step, num, excluded=[]):
    end = start + (num + len(excluded)) * step
    angles_list = np.arange(start, end, step)

    for angle in excluded:
        angles_list = angles_list[angles_list != angle]

    return angles_list


def recon_SIRT_3d(sino, angles, it, extend=False):
    # Extend if required
    '''if extend == True:
        sino, ext = extend_3d(sino)'''

    # defines geometry of the projections
    sino_angles = np.size(sino, 0)
    sino_rows = np.size(sino, 1)
    sino_cols = np.size(sino, 2)

    # flip data to sinogram side as required by astra (from FIJI plain view, rotation axis vertical)
    sino = np.rot90(sino, 1, (0, 1))

    # create projector geometry and id
    proj_geom = astra.create_proj_geom('parallel3d', 1, 1, sino_rows, sino_cols, angles)
    projections_id = astra.data3d.create('-sino', proj_geom, sino)

    # define geometry of host recon volume
    vol_geom = astra.create_vol_geom(sino_cols, sino_cols, sino_rows)
    vol_id = astra.data3d.create('-vol', vol_geom, data=0)

    # configure the algorithm
    alg_cfg = astra.creators.astra_dict('SIRT3D_CUDA')
    alg_cfg['ProjectionDataId'] = projections_id
    alg_cfg['ReconstructionDataId'] = vol_id
    # alg_cfg['option'] = {'MinConstraint' : 0} #For HAADF positivity constraint, not for TEM

    alg_id = astra.algorithm.create(alg_cfg)

    # run the algorithm
    astra.algorithm.run(alg_id, it)
    section_recon = astra.data3d.get(vol_id)

    # Cleanup.
    astra.algorithm.delete(alg_id)
    astra.data3d.delete(vol_id)
    astra.data3d.delete(projections_id)

    # return
    '''if extend == True:
        section_recon = section_recon[:, ext:-ext, ext:-ext]'''
    return section_recon

vol_name = '3_all.tif'
recon_name = 'recon_3_all.tif'

angles_start = -66 #Input start angles in °
angles_step = 2 #Input step in °

sino_full = io.volread(vol_name)
print(sino_full.shape)
#sino_full = sino_full[:,:,:] #[z,y,x] modify y for subvolume reconstruction

angles = angles_list(angles_start, angles_step, np.size(sino_full,0), excluded=[])


if angles[0] == angles_start:
    angles = (angles * np.pi) / 180


section = 50

# Define iteration range to be tested

iterations = [10,30,60,90,120,150]

sino = sino_full[:, section, :]
sino_rows = np.size(sino, 1)


nb_bites = 20
iterations = 30

recon_cols = np.size(sino_full, 2)
sino_full = np.array_split(sino_full, nb_bites, axis=1)
reconstruction_full = np.zeros((1, recon_cols, recon_cols), dtype='float32')

for i in trange(nb_bites):
    sino = sino_full[i]
    #reconstruction_full = np.concatenate((recon_SIRT_3d(sino, angles, iterations, extend=True), reconstruction_full),
                                         #axis=0)
    reconstruction_full = np.concatenate((recon_SIRT_3d(sino, angles, iterations), reconstruction_full),
                                         axis=0)
'''
reconstruction_full = recon_SIRT_3d(sino_full, angles, 30)
reconstruction_full[reconstruction_full < 0] = 0
reconstruction_full /= np.max(reconstruction_full)
reconstruction_full = np.round(reconstruction_full * 255).astype(np.uint8)
'''

reconstruction_full = np.flip(reconstruction_full, 0)
reconstruction_full = np.delete(reconstruction_full, 0, 0)
reconstruction_full = np.rot90(reconstruction_full)


tiff.imwrite(recon_name, np.array(reconstruction_full), bigtiff=True)
