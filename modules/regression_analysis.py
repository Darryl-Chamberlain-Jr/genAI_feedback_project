import pandas as pd
import matplotlib.pyplot as plt

import numpy as np
from itertools import combinations

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge

from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score, train_test_split

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression

from sklearn.decomposition import PCA

import os

def reconcile_filename_column(df):
    if 'filename' in df.keys():
        df.rename(columns={'filename': 'Filename'}, inplace=True)
    return df

def drop_txt(txt):
    if type(txt) == type(2):
        return txt
    else:
        return(txt.replace('.txt', ''))

def combine_psuedos_with_scores(df, scores_df):
    df = reconcile_filename_column(df)
    df['Filename'] = df['Filename'].apply(drop_txt).astype('int')
    scores_df['Psuedos'] = scores_df['Psuedos'].astype('int')

    combined_df = df.merge(scores_df, how='right', left_on='Filename', right_on='Psuedos')
    combined_df = combined_df.dropna()
    return combined_df

def create_X_y(df, scores_df):
    combined_df = combine_psuedos_with_scores(df, scores_df)

    response_var = combined_df['ChatGPT Percent Score']
    combined_df.drop(['Filename', 'Psuedos', 'ChatGPT Percent Score', 'Response Word Count'], axis=1, inplace=True)
    predictor_vars = combined_df

    return {'X': predictor_vars, 
            'y': response_var
    }

def load_dfs(math=None):
    scores_df = pd.read_excel('combined_responses_scores_added.xlsx', index_col=0)
    scores_df = scores_df[['Psuedos', 'ChatGPT Percent Score', 'Response Word Count', 'Course']]
    if math == None:
        scores_df = scores_df[scores_df['Course'] != 'MATH 111']
    
    scores_df = scores_df.drop(['Course'], axis=1)

    list_of_df_dicts = []

    for pred_name in ['taaco', 'taaled', 'taales', 'taassc']:
        temp_df = pd.read_csv(f'./predictor_results/{pred_name}_results.csv')

        temp_X_y_dict = create_X_y(temp_df, scores_df)
        temp_df_dict = {
            'name': pred_name, 
            'df': temp_df, 
            'X': temp_X_y_dict['X'], 
            'y': temp_X_y_dict['y']
        }
        list_of_df_dicts.append(temp_df_dict)
    return [list_of_df_dicts, scores_df]

def create_all_feature_dicts(df, target, subset_n):
    features = df.keys()
    all_subsets = list(combinations(features, subset_n))

    list_of_feature_dicts = []
    name_index = 0
    for subset in all_subsets:
        temp_dict = {
            'name': f'model_{name_index}',
            'features': list(subset), 
            'reduced_df': df[list(subset)], 
            'target': target,
            'model': LinearRegression()
        }
        list_of_feature_dicts.append(temp_dict)
        name_index += 1

    return list_of_feature_dicts

def run_cross_val(dict, cv_n):
    model = dict['model']
    X = dict['reduced_df']
    y = dict['target']
    all_scores = cross_val_score(model, X, y, cv=cv_n)

    dict['cross_val_scores'] = all_scores
    dict['cross_val_avg'] = np.mean(all_scores)
    return dict

def return_selected_dict(list_of_dicts, name):
    selected_dict = list(filter(lambda model: model['name'] == name, list_of_dicts))[0]
    return selected_dict

def brute_force_feature_selection_analysis(total_features):
    base_dir=os.getcwd()
    list_of_df_dicts, scores_df = load_dfs()

    model_conversion_dict = {
        'linear': LinearRegression(),
        'ridge': Ridge(),
        'lasso': Lasso()
    }

    for df_dict in list_of_df_dicts:
        list_of_cv_dicts = []

        for model_type in ['linear', 'ridge', 'lasso']:
            for n_features in range(2, total_features):
                feature_names = df_dict['X'].keys()
                list_all_subsets = list(combinations(feature_names, n_features))
                print(f'Running models for {len(list_all_subsets)} combinations of {n_features} features.')
                
                for subset in list_all_subsets:
                    model_reg = model_conversion_dict[model_type]
                    full_X = df_dict['X']
                    restricted_X = full_X[list(subset)]
                    y = df_dict['y']

                    all_scores = cross_val_score(model_reg, restricted_X, y, cv=5)
                    cross_val_avg = np.mean(all_scores)

                    temp_dict = {
                        'model_type': model_type, 
                        'n_features': n_features, 
                        'feature_names': subset, 
                        'all_cv_scores': all_scores,
                        'cross_val_avg': cross_val_avg
                    }
                    list_of_cv_dicts.append(temp_dict)

        temp_df = pd.DataFrame(list_of_cv_dicts)
        temp_df_name = df_dict['name']
        cv_results_path = os.path.join(base_dir, 'dim_redux_results', f'{temp_df_name}.xlsx')
        temp_df.to_excel(cv_results_path)

def select_k_best_feature_selection_analysis():
    base_dir=os.getcwd()
    list_of_df_dicts, scores_df = load_dfs()

    model_conversion_dict = {
        'linear': LinearRegression(),
        'ridge': Ridge(),
        'lasso': Lasso()
    }

    list_of_cat_cv_dicts = []

    for df_dict in list_of_df_dicts:
        list_of_cv_dicts = []
        X = df_dict['X']
        y = df_dict['y']

        for model_type in ['linear', 'ridge', 'lasso']:
            model_reg = model_conversion_dict[model_type]

            for n_features in range(2, 9):
                fit_X = SelectKBest(f_regression, k=n_features).fit(X, y)
                X_new = fit_X.transform(X)
                feature_names = fit_X.get_feature_names_out()

                all_scores = cross_val_score(model_reg, X_new, y, cv=5)
                cross_val_avg = np.mean(all_scores)

                temp_dict = {
                    'model_type': model_type, 
                    'n_features': n_features, 
                    'feature_names': feature_names, 
                    'all_cv_scores': all_scores,
                    'cross_val_avg': cross_val_avg
                    }
                list_of_cv_dicts.append(temp_dict)

        temp_df = pd.DataFrame(list_of_cv_dicts)
        temp_df_name = df_dict['name']

        list_of_cat_cv_dicts.append({'df': temp_df, 'df_name': temp_df_name})

    cv_results_path = os.path.join(base_dir, 'dim_redux_results', f'kbest_all_categories.xlsx')

    with pd.ExcelWriter(cv_results_path, mode='a', if_sheet_exists='replace') as writer:
        for final_dict in list_of_cat_cv_dicts:
            final_dict['df'].to_excel(writer, sheet_name=f'{final_dict["df_name"]}')

def perform_pca(df_dict, n_components):
    pca = PCA(n_components=n_components, svd_solver='full')
    new_X = pca.fit(df_dict['X']).transform(df_dict['X'])
    df_dict['full_X'] = df_dict['X']
    df_dict['X'] = new_X
    return df_dict

def pca_feature_projection():
    base_dir=os.getcwd()
    list_of_df_dicts, scores_df = load_dfs()

    model_conversion_dict = {
        'linear': LinearRegression(),
        'ridge': Ridge(),
        'lasso': Lasso()
    }

    list_of_cat_cv_dicts = []

    for df_dict in list_of_df_dicts:
        list_of_cv_dicts = []
        X = df_dict['X']
        y = df_dict['y']

        for model_type in ['linear', 'ridge', 'lasso']:
            model_reg = model_conversion_dict[model_type]

            for n_features in range(2, 9):
                temp_pca = PCA(n_components=n_features, svd_solver='full')
                X_new = temp_pca.fit_transform(X)

                all_scores = cross_val_score(model_reg, X_new, y, cv=5)
                cross_val_avg = np.mean(all_scores)

                temp_dict = {
                    'model_type': model_type, 
                    'n_features': n_features, 
                    'all_cv_scores': all_scores,
                    'cross_val_avg': cross_val_avg
                    }
                list_of_cv_dicts.append(temp_dict)

        temp_df = pd.DataFrame(list_of_cv_dicts)
        temp_df_name = df_dict['name']

        list_of_cat_cv_dicts.append({'df': temp_df, 'df_name': temp_df_name})

    cv_results_path = os.path.join(base_dir, 'dim_redux_results', f'pca_all_categories.xlsx')

    with pd.ExcelWriter(cv_results_path, mode='a', if_sheet_exists='replace') as writer:
        for final_dict in list_of_cat_cv_dicts:
            final_dict['df'].to_excel(writer, sheet_name=f'{final_dict["df_name"]}')