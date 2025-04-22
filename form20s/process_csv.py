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
    table_csvs = sorted(table_csvs)
    df_joined = pd.DataFrame()

    try:
        for file in table_csvs:
            file_path = os.path.join(csv_folder, folder, file)
            df_temp = pd.read_csv(file_path, skiprows=1)

            df_joined = pd.concat([df_joined, df_temp], ignore_index=True)
    
    except:
        print(f"Error processing {file}")
        continue
    
    df_joined = df_joined.iloc[:, 2:]
    unnamed_cols = df_joined.columns[df_joined.columns.str.contains("Unnamed")]
    unnamed_cols = sorted(unnamed_cols, key=lambda x: int(x.split(':')[1]))

    extra_cols = unnamed_cols[6:]
    print(f"extra cols: {extra_cols}")
    if df_joined[extra_cols].isna().all().all():
        df_joined.drop(columns=extra_cols, inplace=True)
        unnamed_cols = unnamed_cols[:5]
        
    if len(unnamed_cols) == 5:
        columns_to_update = unnamed_cols[:5]
        df_joined[columns_to_update] = df_joined[columns_to_update].astype("object")
        df_joined.rename(columns=dict(zip(columns_to_update, ["Total of Valid Votes", "No. Of Rejected Votes",
                                                            "Votes for NOTA", "Total", "No. Of Tendered Votes"
                                                            ])), inplace=True)
        error = False

    else:
        print(unnamed_cols)
        print('error outputting', folder)
        error = True
        
    df_joined = df_joined.dropna(how='all')

    search_text = "(To be filled in the case of election from an assembly constituency.)"
    row_index = df_joined[df_joined.iloc[:, 0] == search_text].index

    if not row_index.empty:
        df_joined = df_joined.drop(index=[row_index[0] - 1, row_index[0], row_index[0] + 1]).reset_index(drop=True)

    print('outputting processed file', folder)
    
    if not error:
        df_joined.to_csv(os.path.join(processed_csv_folder, f"{folder}.csv"), index=False)
    else:
        df_joined.to_csv(os.path.join(processed_csv_folder, f"messy_{folder}.csv"), index=False)

