import nibabel as nib
import os
from os.path import join as pjoin
from nibabel import Nifti1Header
from nibabel import Nifti1Image
import numpy as np
import subprocess
from tqdm import tqdm

# save nifti
def save_ciftifile(data, filename, template='./supportfiles/template.dtseries.nii'):
    ex_cii = nib.load(template)
    if data.ndim == 1:
      data = data[None,:]
    ex_cii.header.get_index_map(0).number_of_series_points = data.shape[0]
    nib.save(nib.Cifti2Image(data,ex_cii.header), filename)

# define paths
dataset_path = 'PATHtoDataset'
cft_path = f'{dataset_path}/derivatives/ciftify/'
fmriprep_path = f'{dataset_path}/derivatives/fmriprep'

# flag labels
ses_flag  = ['imagenet']
file_flag = ['Atlas.dtseries']
sub_flag = sorted([i for i in os.listdir(cft_path) if i.startswith('sub')])
# # if only specific subjects are wanted:
# sub_flag = ['sub-02', 'sub-03']
# sub_flag = sorted([i for i in os.listdir(cft_path) if i.startswith('sub') and int(i.split('-')[-1])<=9])

# collect Atlas files
Atlas_files = []
sub_dirs = [pjoin(cft_path, _, 'results/') for _ in os.listdir(cft_path) if _ in sub_flag ]
for sub_dir in sub_dirs:
  ses_dirs = [ _ for _ in os.listdir(sub_dir) if any([__ in _ for __ in ses_flag])]
  for ses_dir in ses_dirs:
    file = [_ for _ in os.listdir(pjoin(sub_dir, ses_dir)) if any([__ in _ for __ in file_flag])]
    Atlas_files.extend([pjoin(sub_dir, ses_dir, _) for _ in file])
Atlas_files.sort()

Atlas_files = [_ for _ in Atlas_files]

# highpass
for file in tqdm(Atlas_files):
  print(file)
  if not os.path.exists(file.replace('Atlas', 'Atlas_hp128')):
    img = nib.load(file)
    data = img.get_fdata()[:]
    dataTdim, dataXdim = data.shape[0], data.shape[1]
    datanewZdim = np.ceil(dataXdim/100)
    # transpose and demean
    mean_data = data.mean(axis=0)
    data = np.r_[(data - mean_data).transpose(), np.zeros((int(100*datanewZdim - dataXdim), dataTdim))]
    data = data.reshape((10, 10, int(datanewZdim), dataTdim))

    # write a header
    header = Nifti1Header()
    header.set_data_shape((10, 10, int(datanewZdim), dataTdim))

    # save the data
    file_name = file.split('/')[-1]
    nii_file = file.replace(f'{file_name}', 'Atlas.nii.gz')
    nib.save(Nifti1Image(data, np.diag([1,1,1,1]), header),  nii_file)
    highpass = 128
    nii_file_path = nii_file.replace('.nii.gz', '')
    fsl_cmd = f'fslmaths {nii_file_path} -bptf {highpass/4} -1 {nii_file_path}'
    subprocess.check_call(fsl_cmd, shell=True)
    data = nib.load(pjoin(cft_path, sub_dir, nii_file)).get_fdata()[:]
    data = data.reshape((int(100*datanewZdim), dataTdim))[0:dataXdim,:].transpose() + mean_data
    # save cifti
    save_ciftifile(data.astype(np.float32), file.replace('Atlas_s0', 'Atlas_hp128'), file)
  
  # delete the temporary nii.gz
  file_name = file.split('/')[-1]
  nii_file = file.replace(f'{file_name}', 'Atlas.nii.gz')
  if os.path.exists(nii_file): 
    rm_cmd = f'rm -f {nii_file}'
    subprocess.check_call(rm_cmd, shell=True)
    print(rm_cmd)

