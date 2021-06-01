
#Author: Alvaro Valarezo

# Import packages that we are going to use for the preprocessing pipeline
import dicom2nifti
import ants
import antspynet
import intensity_normalization
import nibabel as nib
import numpy as np
from deepbrain import Extractor
import subprocess
#Declaration of variables

example_path='./niidata/6_sag_3d_fspgr_bravo_straight.nii.gz'



# Function used to convert DICOM files to NII format in order to pre-process them.
def dicom2nii():
    dicom_directory= './data/'

    output_folder = './niidata/'
    dicom2nifti.convert_directory(dicom_directory, output_folder)  

# Function used to get nii orientation
def get_orientation(path):
    z = ants.image_read(path)
    original_ori = z.get_orientation() # eg: The MRis are in a RPI orientation
    print(original_ori)
    return original_ori

# Function used to get image header of nii file
def get_header(path):
    image_header = ants.image_header_info(example_path)
    print(image_header)
    return image_header
# Function used to denoise image using ANTs
def denoise_image(path, denoise_path):
    img= ants.image_read(path) # Read nii image and convert it to an ANTs array
    denoise= ants.denoise_image(img, noise_model='gaussian') 
    ants.image_write(denoise,denoise_path) #Write image in a specific path 

# Function used to correct the bias field using ANTs
def bias_field_correction(path, bias_path):
    img= ants.image_read(path)
    bias_correction = ants.n4_bias_field_correction(img)
    ants.image_write(bias_correction, bias_path)

# normalization with ans.registration rigid
def image_registration(path, normalized_path):
    img = ants.image_read(path)
    img_fixed= ants.image_read('./image_fixed.nii')
    mytx = ants.registration(fixed=img_fixed, moving=img, type_of_transform= 'Rigid')
    print(mytx)
    image_normalized= ants.apply_transforms( fixed=img_fixed, moving = img, transformlist=mytx['fwdtransforms'])
    ants.image_write(image_normalized, normalized_path)


#Function used to perform brain extraction using U-net and ANTs-based training data
#Note: returns a probability matrix mask (ants)
def brain_extraction(path, brain_path):
    command = subprocess.call('deepbrain-extractor -i '+ path +' -o '+ brain_path, shell = True)
    print(command)
    if command == 0 :
        print("Good")
     
    

#Function used to normalize MR images using zscore distribution without a mask
def normalize_img(path, normalize_path):
    
    img = nib.load(path)
    normalized_image= intensity_normalization.normalize.zscore.zscore_normalize(img, mask=None)
    nib.save(normalized_image, normalize_path)



def calculate_values(path):
    w = ants.image_read(path)
    m = w.numpy()
    m_min = np.min(m) 
    m_max = np.max(m)
    m_avg = np.average(m)
    print("The min value is: %d , max value: %d and the average value: %d " % ( m_min, m_max, m_avg))


denoise_image(example_path,'./niidata/denoise.nii.gz')
bias_field_correction('./niidata/denoise.nii.gz','./niidata/bias.nii.gz')
image_registration('./niidata/bias.nii.gz','./niidata/normalized.nii.gz')
brain_extraction('./niidata/normalized.nii.gz','./niidata/')
normalize_img('./niidata/brain.nii', './niidata/preprocessed.nii.gz')