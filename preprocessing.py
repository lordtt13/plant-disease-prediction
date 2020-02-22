# -*- coding: utf-8 -*-
"""
Created on Sat Feb 22 17:10:00 2020

@author: Tanmay Thakur
"""
import pickle

from os import listdir
from utils import convert_image_to_array


directory_root = "PlantVillage/"

image_list, label_list = [], []

print("[INFO] Loading images ...")
root_dir = listdir(directory_root)

for plant_folder in root_dir :
    plant_disease_folder_list = listdir(f"{directory_root}/{plant_folder}")

    for plant_disease_folder in plant_disease_folder_list:
        print(f"[INFO] Processing {plant_disease_folder} ...")
        plant_disease_image_list = listdir(f"{directory_root}/{plant_folder}/{plant_disease_folder}/")

        for image in plant_disease_image_list[:200]:
            image_directory = f"{directory_root}/{plant_folder}/{plant_disease_folder}/{image}"
            if image_directory.endswith(".jpg") == True or image_directory.endswith(".JPG") == True:
                image_list.append(convert_image_to_array(image_directory))
                label_list.append(plant_disease_folder)
                
print("[INFO] Image loading completed")  

data_dump = image_list, label_list

pickle_out = open("data.pickle","wb")
pickle.dump(data_dump, pickle_out)
pickle_out.close()