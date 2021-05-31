
#Author: Alvaro Valarezo

import dicom2nifti
import ants

#Declaration of variables

example_path='./niidata/6_sag_3d_fspgr_bravo_straight.nii.gz'

denoise_path='./denoisedata/image.nii.gz'

bias_correction_path='./biasdata/image.nii.gz'

# Function to convert DICOM files to NII format in order to pre-process them.
def dicom2nii():
    dicom_directory= './data/'

    output_folder = './niidata/'
    dicom2nifti.convert_directory(dicom_directory, output_folder)  

# Function to get nii orientation
def get_orientation(path):
    z = ants.image_read(path)
    original_ori = z.get_orientation() # eg: The MRis are in a RPI orientation
    print(original_ori)

# Function to get image header of nii file
def get_header(path):
    image_header = ants.image_header_info(example_path)
    print(image_header)

# Function to denoise image using ANTs
def denoise_image(path):
    img= ants.image_read(path) # Read nii image and convert it to an ANTs array
    denoise= ants.denoise_image(img, noise_model='gaussian') 
    ants.image_write(denoise,denoise_path) #Write image in a specific path 

def bias_field_correction(path):
    img= ants.image_read(path)
    bias_correction = ants.n4_bias_field_correction(img)
    ants.image_write(bias_correction, bias_correction_path)



