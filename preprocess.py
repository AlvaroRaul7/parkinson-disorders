
# Author: Alvaro Valarezo
# Import packages that we are going to use for the preprocessing pipeline
import dicom2nifti
import ants
import antspynet
import intensity_normalization
import nibabel as nib
import numpy as np
from deepbrain import Extractor
import subprocess

import os 
#Declaration of variables

example_path='./niidata/6_sag_3d_fspgr_bravo_straight.nii.gz'

# Function used to convert DICOM files to NII format in order to pre-process them.
def dicom2nii(input_dir,output_dir):
    dicom_directory= './data/'

    output_folder = './niidata/'
    dicom2nifti.convert_directory(input_dir, output_dir)  


dic_gamble =[3536,3565,3863,40703,41886,50983,58454]


def create_dir(dic):

    # Directory
    directory = "preprocess/"
    
    # Parent Directory path
    parent_dir = "/home/alvaroraul7/repositories/parkinson-disorders/"
    # Path

    for i in dic:
        path = os.path.join(parent_dir, directory+str(i))
        os.mkdir(path)
        print("Directory '% s' created" % directory)
    

    for i in dic:
        dicom2nii('./TMGAMBLE/'+str(i),'./preprocess/'+str(i))
    
# create_dir(dic_gamble)

# Function used to get nii orientation
def get_orientation(path):
    z = ants.image_read(path)
    original_ori = z.get_orientation() # eg: The MRis are in a RPI orientation
    print(original_ori)
    return original_ori

# Function used to get image header of nii file
def get_header(path):
    image_header = ants.image_header_info(example_path)
    print(image_header) #Print image header 
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
    img_fixed= ants.image_read('./image_fixed.nii')   # Image fixed of the registrattion step
    mytx = ants.registration(fixed=img_fixed, moving=img, type_of_transform= 'Affine')
    print(mytx)
    image_normalized= ants.apply_transforms( fixed=img_fixed, moving = img, transformlist=mytx['fwdtransforms'])
    ants.image_write(image_normalized, normalized_path)


#Function used to perform brain extraction using U-net and ANTs-based training data
#Note: returns a probability matrix mask (ants)
def brain_extraction(path, brain_path, name):
    command = subprocess.call('deepbrain-extractor -i '+ path +' -o '+ brain_path, shell = True)
    print(command)
    if command == 0 :
        print("Good")
     
    img = ants.image_read(brain_path+'brain.nii')
    ants.image_write(img, brain_path+name)
    

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


def convertgz(path):

    img = ants.image_read(path)
    

def multiply(image1,image2,path):
     w1 = ants.image_read(image1)
     m1 = w1.numpy()
     w2 = ants.image_read(image2)
     m2 = w2.numpy()
     print(m1.shape)
     print(m2.shape)

     m3 = m1*m2
     print(m3.shape)
     mult = ants.from_numpy(m3)
     ants.image_write(mult,path)
    

    #mult = ants.multiply_images('niidata/brain.nii', 'segmentation.nii.gz')
    #ants.image_write(mult,path)



# calculate_values('./niidata/brain.nii')
# denoise_image(example_path,'./niidata/denoise.nii.gz')
# bias_field_correction(example_path,'./niidata/bias.nii.gz')
# image_registration('./niidata/bias.nii.gz','./niidata/normalized.nii.gz')
# brain_extraction('./niidata/normalized.nii.gz','./niidata/')
# normalize_img('./niidata/brain.nii', './niidata/preprocessed.nii.gz')
# multiply('niidata/brain.nii', 'segmentation.nii.gz', 'niidata/segmented.nii.gz')


def preprocessing_pipeline(preprocessing_path, dic):

    #Denoising stage

    for i in dic:
        # print("Denoising"+str(i))
        # denoise_image(preprocessing_path+str(i)+'/'+str(i)+'.nii.gz','./preprocessnoise/'+ str(i)+'.nii.gz')
        # print("Field correction"+str(i))
        # bias_field_correction('./preprocessnoise/'+ str(i)+'.nii.gz', './preprocessfieldcorrection/'+ str(i)+'.nii.gz')
        # print("Image registration"+str(i))
        # image_registration('./preprocessfieldcorrection/'+ str(i)+'.nii.gz', './preprocessimage_registration/'+str(i)+'.nii.gz')
        print("Brain extraction"+str(i))
        brain_extraction('./preprocessimage_registration/'+str(i)+'.nii.gz', './preprocessbrain_extraction/', str(i)+'.nii.gz')
        # print("Normalize"+str(i))
        # normalize_img('./preprocessbrain_extraction/'+str(i)+'.nii.gz', './preprocessnormalization'+str(i)+'.nii.gz')
        print("Multiply"+str(i))
        multiply('./preprocessbrain_extraction/'+str(i)+'.nii.gz', 'segmentation.nii.gz','final_preprocessed/'+str(i)+'.nii.gz')
    
# denoise_image('./preprocess/'+str(3062)+'/'+str(3062)+'.nii.gz','./preprocessnoise/'+ str(3062)+'.nii.gz')


preprocessing_pipeline('./preprocess/', dic_gamble)