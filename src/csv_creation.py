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

############
# Running it
############

if __name__ == "__main__":

    csv_path = '/mnt/data/model_36_split_df_all_gt.csv'
    remove_data_split_rename(csv_path, 'test', 'test2', 'test', 'model_36_split_df_test_2_gt.csv', '/mnt/data')