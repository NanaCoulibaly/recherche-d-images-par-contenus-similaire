import numpy as np
from os import listdir
from descriptors import bitdesc, haralick, glcm
from paths import kth_dir, kth_path
import cv2

def main():
    # List for all the signatures from inages with BiTdec, etc..
    listOflists = list()
    print('Extracting features ....')
    # Loop in path and grab subfolders
    counter = 0
    for kth_class in kth_dir:
        print(f'Current folder: {kth_class}')
        # Grab files from subfolders
        for filename in listdir(kth_path + kth_class+'/'):
            counter += 1
            img_name = f'{kth_path}{kth_class}/{filename}'
            #print(img_name)
            # Read/Load Image as gray
            img = cv2.imread(img_name, 0)
            features = bitdesc(img) + [kth_class] + [img_name]
            print(f' Image count: {counter}')
            # Add image features to listOflists
            listOflists.append(features)
    final_array = np.array(listOflists)
    np.save('cbir_signatures_v1.npy', final_array)
    print('Extraction concluded successfully!') 

    def signaturebitdesc():
        # List for all the signatures from inages with BiTdec, etc..
        listOflists = list()
        print('Extracting features with bitdesc descriptor ....')
        # Loop in path and grab subfolders
        counter = 0
        for kth_class in kth_dir:
            print(f'Current folder: {kth_class}')
            # Grab files from subfolders
            for filename in listdir(kth_path + kth_class+'/'):
                counter += 1
                img_name = f'{kth_path}{kth_class}/{filename}'
                #print(img_name)
                # Read/Load Image as gray
                img = cv2.imread(img_name, 0)
                features = bitdesc(img) + [kth_class] + [img_name]
                print(f' Image count: {counter}')
                # Add image features to listOflists
                listOflists.append(features)
        final_array = np.array(listOflists)
        np.save('cbir_bitdesc_signatures_v1.npy', final_array)
        print('Extraction concluded successfully!')

 

    def signatureglcm():
        # List for all the signatures from inages with BiTdec, etc..
        listOflists = list()
        print('Extracting features with glcm descriptor ....')
        # Loop in path and grab subfolders
        counter = 0
        for kth_class in kth_dir:
            print(f'Current folder: {kth_class}')
            # Grab files from subfolders
            for filename in listdir(kth_path + kth_class+'/'):
                counter += 1
                img_name = f'{kth_path}{kth_class}/{filename}'
                #print(img_name)
                # Read/Load Image as gray
                img = cv2.imread(img_name, 0)
                features = glcm(img) + [kth_class] + [img_name]
                print(f' Image count: {counter}')
                # Add image features to listOflists
                listOflists.append(features)
        final_array = np.array(listOflists)
        np.save('cbir_glcm_signatures_v1.npy', final_array)
        print('Extraction concluded successfully!')

 

    def signaturegharalick():
        # List for all the signatures from inages with BiTdec, etc..
        listOflists = list()
        print('Extracting features with haralick descriptor ....')
        # Loop in path and grab subfolders
        counter = 0
        for kth_class in kth_dir:
            print(f'Current folder: {kth_class}')
            # Grab files from subfolders
            for filename in listdir(kth_path + kth_class+'/'):
                counter += 1
                img_name = f'{kth_path}{kth_class}/{filename}'
                #print(img_name)
                # Read/Load Image as gray
                img = cv2.imread(img_name, 0)
                features = haralick(img).tolist() + [kth_class] + [img_name]
                print(f' Image count: {counter}')
                # Add image features to listOflists
                listOflists.append(features)
        final_array = np.array(listOflists)
        np.save('cbir_haralick_signatures_v1.npy', final_array)
        print('Extraction concluded successfully!')

 

    signaturebitdesc()
    signatureglcm()
    signaturegharalick()

if __name__ =="__main__":
    main()