
#Author: Alvaro Valarezo 
import dicom2nifti


# Function to convert DICOM files to NII format in order to pre-process them.
def dicom2nii():
    dicom_directory= './data/'

    output_folder = './niidata/'
    dicom2nifti.convert_directory(dicom_directory, output_folder)  

