import pandas as pd
import os
from .modules import process_data, taaco_analysis, regression_analysis

end_process = False

action_dict = {
    'Close script': 0,
    'Convert xlsx file to txt files for TAACO/TAALED/TAALES/TAASSC analysis': 1,
    'Run TAACO on txt files': 2, 
    'Run TAALED on txt files': 3, 
    'Run TAALES on txt files': 4,
    'Run TAASSC on txt files': 5, 
    'Run brute force feature selection on all features': 6,
    'Run SelectKBest feature selection on all features': 7,
    'Run PCA feature projection on all features': 8
}

while end_process == False:
    print('What would you like to accomplish?')
    for action_key in action_dict.keys():
        print(f'{action_dict[action_key]} - {action_key}')
    try:
        user_input = int(input('Type the single digit action you wish to perform: '))

        if user_input == 1:
            # extract student responses from xlsx to txt
            xlsx_file_name = 'combined_responses_scores_added'
            xlsx_path = f'./data/{xlsx_file_name}.xlsx'
            while not os.path.isfile(xlsx_path):
                print(f'Check that {xlsx_file_name} is in the data folder, then press any key to continue.')
                input()
            data_df = pd.read_excel(xlsx_path, index_col=0)
            process_data.excel_to_txt_files(data_df)

        elif user_input == 2:
            # case want to run TAACO on files
            print('Before running any files, do the following:')
            print('1. install spaCy: https://spacy.io/usage')
            print('2. clone TAACO repo in same parent: https://github.com/LCR-ADS-Lab/TAACO')
            print('    For example, **GitHub**/COAS-GenAI-feedback and **GitHub**/TAACO share the same parent folder **GitHub**')
            taaco_analysis.taaco_on_folder_of_files('student_response_txt_files', 'taaco_results')

        elif user_input == 3 or user_input == 4 or user_input == 5:
            # case any analysis other than TAACO
            print('Natively running this analysis is not yet supported. Download the appropriate program app, then move resulting file to predictor_results folder.')
            feature_dict = {
                '3': 'taaled',
                '4': 'taales',
                '5': 'taassc'
            }
            print(f'Naming convention should be {feature_dict[str(user_input)]}_results.xlsx')

        elif user_input == 6:
            total_features = int(input('How many features would you like the largest model to contain? \nCaution: runtime increases with factorial growth. Choosing more than 4 features may be prohibitively long.'))
            regression_analysis.feature_selection_analysis(total_features)
        
        elif user_input == 7:
            regression_analysis.select_k_best_feature_selection_analysis()

        elif user_input == 8:
            regression_analysis.pca_feature_projection()
            
    except: 
        print('Invalid user input. Please try again.')