# coding:utf8

import os
import csv
import pandas as pd
from datetime import datetime
from argparse import ArgumentParser
from tqdm import tqdm

def time_stat(start_time, end_time):
    return ((end_time - start_time).seconds)

def get_filepaths(args):
    # paths of target 
    paths = []
    for root, dirs, files in os.walk(args.path_click_log): 
        for file in files:  
            if os.path.splitext(file)[1] == '.csv':  
                paths.append(os.path.join(root, file)) 
    print(paths)
    return paths

def get_city_poi(args, path_files):
    #[0:'poi_id', 
    # 1:'displayname', 
    # 2:'alias', 
    # 3:'province', 
    # 4:'city', 
    # 5:'district', 
    # 6:'address', 
    # 7:'category', 
    # 8:'category_code', 
    # 9:'business_district', 
    # 10:'click_score', 
    # 11:'lat', 
    # 12:'lng', 
    # 13:'geohash', 
    # 14:'parent_id', 
    # 15:'child_poi_list', 
    # 16:'split_address']

    #[0:'poi_id', 
    # 1:'displayname', 
    # 6:'address', 
    # 13:'geohash',

    need_cols = [0, 1, 6, 13]
    start_time = datetime.now()
    for pf in path_files:
        count_err, count_pf = 0, 0
        with open(args.sample_city_path, "w") as file_city:
            csv_write = csv.writer(file_city, delimiter='\01')
            with open(pf, "r") as file:
                csv_reader = csv.reader(file)
                for index, row in tqdm(enumerate(csv_reader)):
                    if index == 0:
                        row_value = row[0].split("\01")
                        print(row_value, len(row_value), len(row))
                        
                        need_row_value = [row_value[col] for col in need_cols]
                        csv_write.writerow(need_row_value)

                        print(need_row_value)
                        count_pf += 1
                    else:
                        row_value = "".join(row).split('\01')
                        if len(row_value) == 17:
                            if row_value[4] == args.sample_city:
                                need_row_value = [row_value[col] for col in need_cols]
                                csv_write.writerow(need_row_value)
                                count_pf += 1
                        else:
                            count_err += 1
                            print(len(row_value), "\n", row_value, "\n", row)

                        #assert len(row_value) == 17, ("\n", len(row_value), "\n", row_value, "\n", row)
                        
            file.close()
        file_city.close()
        print(count_err, count_pf)
                        

def sample_data(args):
    # get file in path_log
    path_files = get_filepaths(args)
    print(path_files)
    
    # low memory read data, and get sample index
    get_city_poi(args, path_files)


if __name__ == "__main__":
    # parameter
    parser = ArgumentParser()
    parser.add_argument('--sample_city', type=str, default="北京市")
    parser.add_argument('--sample_city_path', type=str, default="1_poi_need_attr.csv")
    parser.add_argument('--path_click_log', type=str, default="../poi/poi_data_bj_20220929/")
    parser.add_argument('--chunk_size', type=int, default=50000)
    args = parser.parse_args()


    # function to sample data
    sample_data(args)