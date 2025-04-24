import os
import re
import pandas as pd

pdf_folder = os.path.join("data/pdf/ls/2024/", "kerala")
csv_folder = os.path.join("data/csv/", "raw/kerala") 
processed_csv_folder = os.path.join("data/csv/", "processed/kerala") 

if not os.path.exists(csv_folder):
    os.makedirs(csv_folder)

if not os.path.exists(processed_csv_folder):
    os.makedirs(processed_csv_folder)

table_folders = [f for f in os.listdir(csv_folder)]

for folder in table_folders:
    table_csvs = [f for f in os.listdir(os.path.join(csv_folder, folder)) if f.endswith(".csv")]
    table_csvs = sorted(table_csvs, key=lambda x: [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', x)])
    df_joined = pd.DataFrame()

    try:
        for file in table_csvs:
            file_path = os.path.join(csv_folder, folder, file)
            df_temp = pd.read_csv(file_path, skiprows=1)

            df_joined = pd.concat([df_joined, df_temp], ignore_index=True)
            
    except:
        print(f"Error processing {file}")
        continue
    
    search_text = "Total\nEVM\nVotes"
    last_row_index = df_joined[df_joined.iloc[:, 0] == search_text].index

    if not last_row_index.empty:
        df_joined = df_joined.iloc[:last_row_index[0]].reset_index(drop=True)

    df_joined = df_joined.iloc[:, 2:]

    unnamed_cols = df_joined.columns[df_joined.columns.str.contains("Unnamed")]

    extra_cols = unnamed_cols[6:]

    if df_joined[extra_cols].isna().all().all():
        df_joined.drop(columns=extra_cols, inplace=True)
        unnamed_cols = unnamed_cols[:5]
        
    if len(unnamed_cols) == 5:
        columns_to_update = unnamed_cols[:5]
        df_joined[columns_to_update] = df_joined[columns_to_update].astype("object")
        df_joined.rename(columns=dict(zip(columns_to_update, ["Total of Valid Votes", "No. Of Rejected Votes",
                                                            "Votes for NOTA", "Total", "No. Of Tendered Votes"
                                                            ])), inplace=True)
        
        unnamed_cols = df_joined.columns[df_joined.columns.str.contains("Unnamed")]
        if len(unnamed_cols) == 0:
            error = False

    else:
        print(unnamed_cols)
        print('error outputting', folder)
        error = True
        
    df_joined.dropna(axis=1, how='all', inplace=True)
    df_joined = df_joined.dropna(how='all')

    print('outputting processed file', folder)
    
    if not error:
        df_joined.to_csv(os.path.join(processed_csv_folder, f"{folder}.csv"), index=False)
    else:
        df_joined.to_csv(os.path.join(processed_csv_folder, f"messy_{folder}.csv"), index=False)

