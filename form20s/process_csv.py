import os
import re
import pandas as pd
import numpy as np

state = "kerala"
pdf_folder = os.path.join("data/pdf/ls/2024/", state)
csv_folder = os.path.join("data/csv/raw/", state) 
processed_csv_folder = os.path.join("data/csv/processed/", state) 

if not os.path.exists(csv_folder):
    os.makedirs(csv_folder)

if not os.path.exists(processed_csv_folder):
    os.makedirs(processed_csv_folder)

table_folders = [f for f in os.listdir(csv_folder)]

for folder in table_folders:
    table_csvs = [f for f in os.listdir(os.path.join(csv_folder, folder)) if f.endswith(".csv")]
    table_csvs = sorted(table_csvs, key=lambda x: [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', x)])
    df_joined = pd.DataFrame()

    for file in table_csvs:
        file_path = os.path.join(csv_folder, folder, file)
        df_temp = pd.read_csv(file_path, skiprows=1)

        if not "Unnamed: 1" in df_temp.columns:
            df_temp.insert(1, "Unnamed: 1", np.nan)
            
            df_temp.columns = [
                re.sub(r"^Unnamed: (\d+)$", "Unnamed: " + str(int(col.split(": ")[1]) + 1), col)
                if re.match(r"^Unnamed: \d+$", col) and int(col.split(": ")[1]) > 1 else col
                for col in df_temp.columns
            ]

        df_joined = pd.concat([df_joined, df_temp], ignore_index=True)
    
    search_text = "Total\nEVM\nVotes"
    last_row_index = df_joined[df_joined.iloc[:, 0] == search_text].index

    if not last_row_index.empty:
        df_joined = df_joined.iloc[:last_row_index[0]].reset_index(drop=True)

    df_joined = df_joined.iloc[:, 2:]
    df_joined.dropna(axis=1, how='all', inplace=True)

    unnamed_cols = df_joined.columns[df_joined.columns.str.contains("Unnamed")]
        
    if len(unnamed_cols) == 5:
        df_joined[unnamed_cols] = df_joined[unnamed_cols].astype("object")
        df_joined.rename(columns=dict(zip(unnamed_cols, ["Total of Valid Votes", "No. Of Rejected Votes",
                                                            "Votes for NOTA", "Total", "No. Of Tendered Votes"
                                                            ])), inplace=True)
        error = False
        
    elif len(unnamed_cols) > 5:
        columns_to_update = unnamed_cols[-5:]
        df_joined[columns_to_update] = df_joined[columns_to_update].astype("object")
        df_joined.rename(columns=dict(zip(columns_to_update, ["Total of Valid Votes", "No. Of Rejected Votes",
                                                            "Votes for NOTA", "Total", "No. Of Tendered Votes"
                                                            ])), inplace=True)
        
        # if state == "kerala" and "PRATHAPAN BRAHMAKUMAR" in df_joined.columns:
        #     df_joined.columns.values[5:7] = ["PRATHAPAN", "BRAHMAKUMAR"]
        #     error = False
        # else:
        error = True

    else:
        print(unnamed_cols)
        print('error outputting', folder)
        error = True
        
    df_joined = df_joined.dropna()

    print('outputting processed file', folder)
    
    if not error:
        df_joined.to_csv(os.path.join(processed_csv_folder, f"{folder}.csv"), index=False)
    else:
        df_joined.to_csv(os.path.join(processed_csv_folder, f"messy_{folder}.csv"), index=False)
