#########
# Imports
#########

import pandas as pd
import os

###########
# Functions
###########

def remove_data_split_rename(csv_path, data_split_to_remove, data_split_to_rename, name_of_renamed_data_split, csv_name, save_path):

    df = pd.read_csv(csv_path)
    df_filtered = df[df['dataset'] != data_split_to_remove]
    df_filtered['dataset'] = df_filtered['dataset'].replace(data_split_to_rename, name_of_renamed_data_split)
    df_filtered.reset_index(drop=True, inplace=True)
    df_filtered.to_csv(os.path.join(save_path, csv_name))

# Function to create a list of image paths
def create_time_point_pairs(patient_directory, path_to_save):

    # Initialize empty lists to store data
    all_image_paths = []

    # Iterate through the patient directory
    image_paths = []
    
    files = os.listdir(patient_directory)
    files.sort()  # Sort the files to ensure the correct order

    for file in files:
        image_path = os.path.join(patient_directory, file)
        image_paths.append(image_path)
    all_image_paths.extend(image_paths)

    # Create a DataFrame with image paths and their corresponding next image paths
    df = pd.DataFrame({'Image_Path': all_image_paths})

    # Extract the patient and time point from the filenames
    df[['Patient', 'Time_Point']] = df['Image_Path'].str.rsplit(' ', 1, expand=True)

    # Extract the numeric part of the time point and convert to integers
    df['Time_Point'] = df['Time_Point'].str.extract(r'(\d+)').astype(int)

    # Sort the DataFrame based on patient and time point
    df = df.sort_values(by=['Patient', 'Time_Point'])

    # Create a new column for the next time point
    df['Next_Image_Path'] = df.groupby('Patient')['Image_Path'].shift(-1)

    # Remove the last row for each patient as it has no next image path
    df = df.dropna()

    # Save the DataFrame to a CSV file
    df = df[['Image_Path', 'Next_Image_Path']]
    df.to_csv(path_to_save, index=False)



############
# Running it
############

if __name__ == "__main__":

    # csv_path = '/mnt/data/model_36_split_df_all_gt.csv'
    # remove_data_split_rename(csv_path, 'test', 'test2', 'test', 'model_36_split_df_test_2_gt.csv', '/mnt/data')
    create_time_point_pairs('/sddata/projects/Cervical_Cancer_Projects/data/IRIS_cambodia', '/sddata/projects/Cervical_Cancer_Projects/cervical_cancer/csvs/IRIS_cambodia_timepoint_pairs.csv')