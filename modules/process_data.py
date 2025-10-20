# Clean files
import pandas as pd
from numpy.random import randint
from numpy import sort

def extract_to_file(row):
    """
    Takes in a row of student data, extracts the 'Student Response', and creates a .txt file with the 'Student Response'
    """
    
    temp_text = row['Student Response']
    temp_file_path = f"./student_response_txt_files/{row['Psuedos']}.txt"
    with open(temp_file_path, 'w', encoding='utf-8') as text_file:
        text_file.write(temp_text)

def excel_to_txt_files(data_df):
    """
    Takes in an .xlsx file prepped with student responses, chatGPT feedback, and other columns to create a psuedonym and a .txt file for each row
    """
    data_df = data_df.reset_index(drop=True)

    if 'Psuedos' in data_df.keys():
        pass
    else:
        list_of_pseudos = []
        while len(list_of_pseudos) < len(data_df):
            temp_rand = randint(100000, 999999)
            if temp_rand in list_of_pseudos:
                pass
            else:
                list_of_pseudos.append(temp_rand)

        data_df['Psuedos'] = pd.Series(list_of_pseudos)

    data_df.apply(extract_to_file, axis=1)