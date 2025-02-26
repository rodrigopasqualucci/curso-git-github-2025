#LIBS
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import tempfile
import boto3
import joblib
from scipy.stats import ks_2samp
import random
from sklearn.metrics import roc_auc_score
from sklearn.feature_selection import RFECV
from sklearn.model_selection import train_test_split
import plotly.express as px
import plotly.graph_objects as go
from tqdm import tqdm
from typing import List, Dict, Any
from pprint import pprint
from boruta import BorutaPy
from lightgbm import LGBMClassifier
import catboost
from catboost import CatBoostClassifier, Pool, EFeaturesSelectionAlgorithm, EShapCalcType
from sklearn.ensemble import RandomForestClassifier
from feature_engine.selection import SmartCorrelatedSelection
from sklego.feature_selection import MaximumRelevanceMinimumRedundancy
from sklearn.model_selection import StratifiedShuffleSplit
import warnings
warnings.filterwarnings('ignore')
 
#MLFLOW
import mlflow
from typing import Optional
REMOTE_SERVER_URI = "https://mlflow.conecta360dados.com.br"
mlflow.set_tracking_uri(REMOTE_SERVER_URI)
 
#FUNCTIONS
#MLFLOW
def get_or_create_experiment(experiment_name: str, artifact_location: Optional[str]):
    """
    Retrieve the ID of an existing MLflow experiment or create a new one if it doesn't exist.
 
    This function checks if an experiment with the given name exists within MLflow.
    If it does, the function returns its ID. If not, it creates a new experiment
    with the provided name and returns its ID.
 
    Parameters:
    - experiment_name (str): Name of the MLflow experiment.
 
    Returns:
    - str: ID of the existing or newly created MLflow experiment.
    """
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment:
        return experiment.experiment_id
    else:
        return mlflow.create_experiment(experiment_name, artifact_location=artifact_location)
   
 
def feature_selection_plot(all_features_list, list_of_drops, base_report_feats_remove, name_target):
   
    df_temp = pd.DataFrame([elem for elem in all_features_list if elem not in list_of_drops+[name_target]], columns=['FEATURE_NAME'])
    df_join = df_temp.merge(base_report_feats_remove, on='FEATURE_NAME', how='left')
   
    # binary column
    df_join['DROP_COL'] = df_join['REASON_FOR_REMOVAL'].notna().astype(int)
   
    # Count the number of 0s and 1s in DROP_COL
    counts = df_join['DROP_COL'].value_counts().reset_index()
    counts.columns = ['Drop Status', 'Count']
    counts['Drop Status'] = counts['Drop Status'].map({0: 'Not Dropped', 1: 'Dropped'})
 
    # Create a bar plot using Plotly Express
    fig = px.bar(counts,
                 x='Drop Status',
                 y='Count',
                 #text='Count',
                 color='Drop Status',
                 color_discrete_map={'Not Dropped': 'lightgrey', 'Dropped': '#1f78b4'},
                 width=1000,
                 height=500)
 
    # Update layout for better readability
    #fig.update_traces(texttemplate='%{text}', textposition='inside')
    fig.update_layout(title_text='Comparison of Feature Drop Status',
                      xaxis_title='Drop Status',
                      yaxis_title='Count',
                      #uniformtext_minsize=8,
                      #uniformtext_mode='hide',
                     )
 
    # Show the plot
    #fig.show()
   
    return fig, df_join
 
def read_dataset(path_base, sep=None, header=None, fraction_sample=0.5, random_state=42, selected_cols=None):
 
    if path_base.endswith('.xlsx') or path_base.endswith('.xls'):
        raise ValueError("Unsupported file format: Excel files are not supported by this function")
   
    # Tenta ler como Parquet primeiro
    try:
        print('Attempting to read customer dataframe in PARQUET format')
        df = pd.read_parquet(path_base, columns=selected_cols).sample(frac=fraction_sample, random_state=random_state)
        return df
    except Exception as e:        
        print(f'Failed to read as PARQUET: {e}')
        # Tenta ler como CSV ou TXT
        try:
            print('Attempting to read customer dataframe in CSV or TXT format')
            if sep is not None:
                df = pd.read_csv(path_base, delimiter=sep, header=0).sample(frac=fraction_sample, random_state=random_state)
            else:
                df = pd.read_csv(path_base, header=0).sample(frac=fraction_sample, random_state=random_state)
            return df
        except Exception as e:
            print(f'Failed to read as CSV or TXT: {e}')
            raise
 
    return df  
 
def filter_binary_target(df, target_col='tg', target_0="None", target_1="None"):
    """
    Function to filter a DataFrame based on a binary target column.
    Args:
    - df (pandas.DataFrame): Pandas DataFrame to be filtered.
    - target_col (str): Name of the binary target column. Default is 'tg'.
    - target_0 (int, str or None, optional): Value representing the first binary target category.
        If None, assumes 0 as the first category. Default is None.
    - target_1 (int, str or None, optional): Value representing the second binary target category.
        If None, assumes 1 as the second category. Default is None.
    Returns:
    - pandas.DataFrame: Filtered Pandas DataFrame containing rows where the target column matches
        one of the specified binary target categories.
    - int: Total number of rows removed during the filtering process.
    """
    initial_rows = df.shape[0]
   
    # Verificar se target_0 e target_1 são None
    if target_0 == "None" and target_1 == "None":
        # Tentativa de conversão para numérico para lidar com casos de "0", "1" em formato string
        df[target_col] = pd.to_numeric(df[target_col], errors='coerce')                        
        # Filtrar para manter apenas 0 e 1
        df = df[df[target_col].isin([0, 1])]
        df.dropna(subset=[target_col], inplace=True)
        df[target_col] = df[target_col].astype(int)
        n_rows_removed = initial_rows - df.shape[0]
       
    else:
        # Converter todos os valores para string para facilitar a manipulação
        df[target_col] = df[target_col].astype(str).str.strip()
 
        # Substituir valores missing que estão como strings (ex: 'nan') por np.nan
        df[target_col].replace(['nan', 'NaN', 'None', 'Nan', ''], np.nan, inplace=True)
        df.dropna(subset=[target_col], inplace=True)
 
        # Mapeia valores para 0 e 1 com base em target_0 e target_1
        df[target_col] = df[target_col].apply(lambda x: 1 if x == str(target_1) else (0 if x == str(target_0) else None))
        initial_rows = df.shape[0]
               
        n_rows_removed = initial_rows - df.shape[0]
   
    return df, n_rows_removed
 
 
def find_and_remove_duplicates(df):
    """
    Function to keep duplicate rows in a separate variable and remove duplicates from the original variable.
   
    Args:
    - df (pandas.DataFrame): Input DataFrame.
   
    Returns:
    - pandas.DataFrame: DataFrame with duplicates removed.
    - pandas.DataFrame: DataFrame containing only the duplicate rows.
    """
    # Find duplicate rows
    duplicate_rows = df[df.duplicated(keep=False)]
   
    # Remove duplicates from the original DataFrame, keeping only the first occurrence
    filtered_df = df.drop_duplicates(keep='first')
   
    return filtered_df, duplicate_rows
 
def transform_to_regex_pattern(str_list=['vlr']):
    pattern = "|".join(re.escape(s) for s in str_list)
    return pattern
 
def data_type_transform(df, lst_drop_feats = ['cnpj_radical',], regex_lst= ['flag']):
   
    pattern = transform_to_regex_pattern(regex_lst)    
    lst_cols_str = list(df.filter(regex=r"{0}".format(pattern)).columns)
   
    for col in df.columns:
       
        if col in lst_cols_str:
            #print(col)
            df[col] = df[col].astype('object')
       
        elif col == target_column:
            df[col] = df[col].astype('int')
                   
        elif col not in lst_drop_feats and col not in lst_cols_str:
            df[col] = df[col].astype('float')
           
    return df
 
def remove_high_missing_features(df, target_col, threshold):
    remove_features = []  
 
    df_temp = df
    total_rows = df_temp.shape[0]    
    df_temp = df_temp.drop([target_col], axis=1, errors='ignore')
   
    for column in df_temp.columns:
        missing_percentage = df_temp[column].isnull().sum() / total_rows
       
        if missing_percentage > threshold:
            remove_features.append(column)
 
    return remove_features
 
 
# Função para calcular a correlação entre features
def calculate_correlations(df, column_target, threshold=0.8):
   
    # Filtrar apenas colunas numéricas
    df_numeric = df.select_dtypes(exclude=['object', 'string', 'category'])  
 
    # Removendo target se estiver presente  
    df_numeric = df_numeric.drop([column_target], axis=1, errors='ignore')
 
    # Calcular a matriz de correlação
    corr_matrix = df_numeric.corr()
    correlated_pairs = set()
    cont=1
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > threshold:
 
                correlated_pairs.add((corr_matrix.columns[i], corr_matrix.columns[j]))
               
                cont+=1
    return correlated_pairs
 
 
# Função para escolher a feature com maior AUC ou variância
def choose_best_feature_by_auc_or_var(X, y, correlated_pairs, criteria='auc'):
    remove_features = []
    auc_all = []
    var_all = []
    # Selecionar o critério
    if criteria == 'auc':
        print('...AUC method is starting off...it may take a few minutes...please wait...')
        for i, (feature1, feature2) in enumerate(tqdm(correlated_pairs, desc="Comparing AUCs")):
            auc1 = evaluate_feature_auc(X[feature1].values.reshape(-1, 1), y)
            auc2 = evaluate_feature_auc(X[feature2].values.reshape(-1, 1), y)
            #print(f'Rodada {i+1}: {feature1} AUC {auc1:.4f} ||| {feature2} AUC {auc2:.4f}', end='\r')
            auc_all.append((feature1, auc1, feature2, auc2))
            if auc1 >= auc2:            
                remove_features.append(feature2)
            else:            
                remove_features.append(feature1)
               
    elif criteria == 'var':        
        print('...Variance method is starting off...it may take a few minutes...please wait...')
        for i, (feature1, feature2) in enumerate(tqdm(correlated_pairs, desc="Comparing Variances")):
            var1 = X[feature1].var()
            var2 = X[feature2].var()
            #print(f'Rodada {i+1}: {feature1} VAR {var1:.4f} ||| {feature2} VAR {var2:.4f}', end='\r')
            var_all.append((feature1, var1, feature2, var2))
           
            if var1 >= var2:            
                remove_features.append(feature2)
            else:            
                remove_features.append(feature1)
               
    return remove_features
 
# Função para avaliar AUC de uma feature
def evaluate_feature_auc(X, y):
   
    # Converter para DataFrame para usar fillna
    X = pd.DataFrame(X)
    # Substitui NaN por a média da coluna
    X = X.fillna(X.mean())
 
    # Substitui valores infinitos por um valor grande, mas finito
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.mean())  # Agora preenche os antigos infinitos que se tornaram NaN
 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=10)
 
    model = LGBMClassifier(**{'force_col_wise': True, "verbosity": -1, 'random_state': 11})
    model.fit(X_train, y_train)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    return roc_auc_score(y_test, y_pred_proba)
 
def iv_woe(data, target, vars_, numeric_vars, categorical_vars, bins=10, show_woe=False):
    # Empty Dataframe
    newDF, woeDF = pd.DataFrame(), pd.DataFrame()
    iv_dict = {}
 
    # Drop columns safely
    numeric_vars = [col for col in numeric_vars if col not in [target]]
    categorical_vars = [col for col in categorical_vars if col not in [target]]
 
    # Run WOE and IV on all the independent variables
    for ivars in vars_:
        if ivars in numeric_vars:
            # Handling NaN values by filling them with the median of the column
            if data[ivars].isnull().sum() > 0:
                data[ivars] = data[ivars].fillna(data[ivars].median())
           
            # Using pd.qcut to bin the values
            try:
                binned_x = pd.qcut(data[ivars], bins, duplicates='drop')
                d0 = pd.DataFrame({'x': binned_x, 'y': data[target]})
            except ValueError as e:
                #print(f"Error processing {ivars}: {e}")
                continue
        else:
            d0 = pd.DataFrame({'x': data[ivars], 'y': data[target]})
       
        d0 = d0.astype({"x": str})
        d = d0.groupby("x", as_index=False, dropna=False).agg({"y": ["count", "sum"]})
        d.columns = ['Cutoff', 'N', 'Events']
        d['% of Events'] = np.maximum(d['Events'], 0.5) / d['Events'].sum()
        d['Non-Events'] = d['N'] - d['Events']
        d['% of Non-Events'] = np.maximum(d['Non-Events'], 0.5) / d['Non-Events'].sum()
        d['WoE'] = np.log(d['% of Non-Events']/d['% of Events'])
        d['IV'] = d['WoE'] * (d['% of Non-Events']-d['% of Events'])
        d.insert(loc=0, column='Variable', value=ivars)
        iv_dict[ivars] = [round(d['IV'].sum(), 6)]
       
        temp = pd.DataFrame({"Variable": [ivars], "IV": [d['IV'].sum()]}, columns=["Variable", "IV"])
        newDF = pd.concat([newDF, temp], axis=0)
        woeDF = pd.concat([woeDF, d], axis=0)
       
        # Show WOE Table
        if show_woe:
            print(d)
   
    iv_ = (pd.DataFrame(iv_dict).T.reset_index().set_axis(["var", "IV"], axis=1).sort_values(by="IV", ascending=False).reset_index(drop=True))
   
    return newDF
 
class RankCountVectorizer(object):
    """Vectorizes categorical variables by rank of magnitude.
    This is a hybrid between label vectorizing (assigning every variable a unique ID)
    and count vectorizing (replacing variables with their count in training set).
    Categorical variables are ranked by their count in train set. If a never-before-seen
    variable is in test set, it gets assigned `1`, and is treated like a rare variable in
    the train set. NaN's can be treated as specifically encoded, for instance with `-1`,
    or they can be set to `0`, meaning some algorithms will treat them as missing / ignore
    them. Linear algorithms should be able to work with labelcount encoded variables.
    They basically get treated as: how popular are these variables?
    Example:
        |cat|
        -----
        a
        a
        a
        a
        b
        c
        c
        NaN
        vectorizes to:
        |cat|
        -----
        3
        3
        3
        3
        1
        2
        2
        -1
    Attributes:
        verbose: An integer specifying level of verbosity. Default is `0`.
        set_nans: An integer for filling NaN values. Default is `-1`.
    """
 
    def __init__(self, verbose=0, set_nans=-999, min_count=40):
        self.verbose = verbose
        self.set_nans = set_nans
        self.min_count = min_count
 
    def __repr__(self):
        return "RankCountVectorizer(verbose=%s, set_nans=%s)" % (
            self.verbose,
            self.set_nans,
        )
 
    def fit(self, df, cols=[]):
        """Fits a vectorizer to a dataframe.
        Args:
            df: a Pandas dataframe.
            cols: a list (or 1-D Numpy array) of strings with column headers. Default is `[]`.
        """
        if self.verbose > 0:
            print(
                "Labelcount fitting columns: %s on dataframe shaped %sx%s"
                % (cols, df.shape[0], df.shape[1])
            )
 
        vec = {}
        for col in cols:
            vec[col] = {}
 
        for col in cols:
            if self.verbose > 0:
                print(
                    "Column: %s\tCardinality: %s" % (col.rjust(20), df[col].nunique())
                )
            d = df[col].value_counts()
            d = d[d > self.min_count]
            for i, k in enumerate(sorted(d.to_dict(), key=d.get)):
                vec[col][k] = i + 1
            vec[col][-999] = self.set_nans
        self.vec = vec
 
    def transform(self, df, cols=[]):
        """Transforms a dataframe with a vectorizer.
        Args:
            df: a Pandas dataframe.
            cols: a list (or 1-D Numpy array) of strings with column headers. Default is `[]`.
        Returns:
            df: a Pandas dataframe where specified columns are vectorized.
        Raises:
            AttributeError: Transformation was attempted before fitting the vectorizer.
        """
        try:
            self.vec
        except AttributeError:
            import sys
 
            sys.exit(
                "AttributeError. `self.vec` is not set. Use .fit() before transforming."
            )
 
        if self.verbose > 0:
            print(
                "Labelcount transforming columns: %s on dataframe shaped %sx%s"
                % (cols, df.shape[0], df.shape[1])
            )
 
        for col in cols:
            if self.verbose > 0:
                print(
                    "Column: %s\tCardinality: %s" % (col.rjust(20), df[col].nunique())
                )
            df[col].fillna(-999, inplace=True)
            df[col] = df[col].apply(
                lambda x: self.vec[col][x] if x in self.vec[col] else 1
            )
        return df
 
    def fit_transform(self, df, cols=[]):
        """Calls fit then calls transform in one line.
        Args:
            df: a Pandas dataframe.
            cols: a list (or 1-D Numpy array) of strings with column headers. Default is `[]`.
        Returns:
            df: a Pandas dataframe where specified columns are vectorized.
        """
        self.fit(df, cols)
        return self.transform(df, cols)
 
def drift_preprocessing(df, date_ref= 'date', n_recent_months=1):
 
    # Sort the unique dates in ascending order
    sorted_dates = sorted(df[date_ref].unique())
 
    # Select the most recent two dates
    most_recent_dates_list = sorted_dates[-n_recent_months:]
 
    # split into training drift sample and oot drigt sample
    base_train_drift = df[~df[date_ref].isin(most_recent_dates_list)]
    base_oot_drift = df[df[date_ref].isin(most_recent_dates_list)]
    print(f'Base_train_drift {base_train_drift.shape}')
    print(f'Base_oot_drift {base_oot_drift.shape}')
 
    return base_train_drift, base_oot_drift  
 
 
def create_target_drift(base_treino, base_oot):
    base_treino['target_drift'] = 0
    base_oot['target_drift'] = 1
    return base_treino, base_oot
 
 
def drift_analysis(BASE_TREINO, BASE_OOT, BASE_TEST, target_drift_name = 'target_drift', target_name_risk = 'target', cols_dropping=[]):
   
    list_feats_end = []
   
    # Passo 1: Criação da Target de Drift
    BASE_TREINO, BASE_OOT = create_target_drift(BASE_TREINO, BASE_OOT)
 
    BASE_TREINO[target_name_risk] = BASE_TREINO[target_name_risk].astype(int)
    BASE_OOT[target_name_risk] = BASE_OOT[target_name_risk].astype(int)
    BASE_TEST[target_name_risk] = BASE_TEST[target_name_risk].astype(int)
 
    # Passo 2: Inicialização de Variáveis
    ks = 0
    ks_anterior = 0
    tolerancia = 0.03      
 
    # Loop principal
    i = 1
    while True:
        print()        
        print(f"Drift round {i}", end='\r', flush=True)
 
        list_feats_start = list(BASE_TREINO.columns)
       
        # Passo 3: Lista de Features
        if i != 1:
            list_feats_start = [x for x in list_feats_start if x not in list_feats_end]
            #print(len(list_feats_start))
        else:
            pass
            #print(len(list_feats_start))
           
       
        # Passo 4: Junção das Bases de Dados
        BASE_DRIFT = pd.concat([BASE_TREINO, BASE_OOT], ignore_index=True)        
 
        # Passo 5: Modelo Inicial com LightGBM
        modelo_drift = LGBMClassifier(verbosity=-1, random_state=86)
        modelo_drift.fit(BASE_DRIFT[list_feats_start].drop(cols_dropping+['target_drift']+[target_name_risk], axis=1, errors='ignore'), BASE_DRIFT[target_drift_name])
        y_prob = modelo_drift.predict_proba(BASE_DRIFT[modelo_drift.feature_name_])[:, 1]
        auc_drift = roc_auc_score(BASE_DRIFT[target_drift_name], y_prob)
        #print(auc_drift)
 
        # Passo 6: Verificação da AUC
        if auc_drift > 0.55:
            # Passo 7: Remoção de Features
            #print(list_feats_start)
            #importancias_df, remove_feat = remover_feature_importante(modelo_drift)
            temp_imp = pd.DataFrame({'FEATURE_NAME': modelo_drift.feature_name_, 'FEATURE_IMPORTANCE': modelo_drift.feature_importances_})
            remove_feat = temp_imp.loc[temp_imp['FEATURE_IMPORTANCE'].idxmax()]['FEATURE_NAME']
            list_feats_end.append(remove_feat)
            #print(list_feats_end)
           
            # Passo 8: Treinamento do Modelo de Risco
            modelo_risco = LGBMClassifier(verbosity=-1, random_state=88)
            modelo_risco.fit(BASE_TREINO[list_feats_start].drop(cols_dropping+['target_drift']+[target_name_risk], axis=1, errors='ignore'), BASE_TREINO[target_name_risk])
            #print(modelo_risco)
            y_prob_test = modelo_risco.predict_proba(BASE_TEST[modelo_risco.feature_name_])[:, 1]
            #print(y_prob_test)
           
            # ks_test
            #ks = calcular_ks(BASE_TEST[target_name_risco], y_prob_test)
            ks = ks_score(BASE_TEST[target_name_risk], y_prob_test)
            #print(ks_anterior)
            #print(ks)
           
            #print()
            #print('*'*50)
 
            # Passo 9: Verificação do ks (ks_test)
            if ks > (ks_anterior - (ks_anterior*tolerancia)):
                # recomeça LOOPING
                ks_anterior = ks
               
            else:                
                break  # Condição falsa, sair do loop  
            i+=1
        else:
            break  # Condição falsa, sair do loop
 
    return list_feats_end
 
 
def create_feature_selection_report(cols2drop=[], reason_for_removal = 'This is the explanation for the element.'):
   
    # Example list whose length will determine the number of repetitions
    elements_list = cols2drop
 
    # The explanation string
    explanation_string = reason_for_removal
 
    # Create a new list with the explanation repeated for each element in elements_list
    repeated_explanations = [explanation_string for _ in elements_list]
   
    return (elements_list, repeated_explanations)
 
class RankCountVectorizer(object):
    """Vectorizes categorical variables by rank of magnitude.
    This is a hybrid between label vectorizing (assigning every variable a unique ID)
    and count vectorizing (replacing variables with their count in training set).
    Categorical variables are ranked by their count in train set. If a never-before-seen
    variable is in test set, it gets assigned `1`, and is treated like a rare variable in
    the train set. NaN's can be treated as specifically encoded, for instance with `-1`,
    or they can be set to `0`, meaning some algorithms will treat them as missing / ignore
    them. Linear algorithms should be able to work with labelcount encoded variables.
    They basically get treated as: how popular are these variables?
    Example:
        |cat|
        -----
        a
        a
        a
        a
        b
        c
        c
        NaN
        vectorizes to:
        |cat|
        -----
        3
        3
        3
        3
        1
        2
        2
        -1
    Attributes:
        verbose: An integer specifying level of verbosity. Default is `0`.
        set_nans: An integer for filling NaN values. Default is `-1`.
    """
 
    def __init__(self, verbose=0, set_nans=-999, min_count=40):
        self.verbose = verbose
        self.set_nans = set_nans
        self.min_count = min_count
 
    def __repr__(self):
        return "RankCountVectorizer(verbose=%s, set_nans=%s)" % (
            self.verbose,
            self.set_nans,
        )
 
    def fit(self, df, cols=[]):
        """Fits a vectorizer to a dataframe.
        Args:
            df: a Pandas dataframe.
            cols: a list (or 1-D Numpy array) of strings with column headers. Default is `[]`.
        """
        if self.verbose > 0:
            print(
                "Labelcount fitting columns: %s on dataframe shaped %sx%s"
                % (cols, df.shape[0], df.shape[1])
            )
 
        vec = {}
        for col in cols:
            vec[col] = {}
 
        for col in cols:
            if self.verbose > 0:
                print(
                    "Column: %s\tCardinality: %s" % (col.rjust(20), df[col].nunique())
                )
            d = df[col].value_counts()
            d = d[d > self.min_count]
            for i, k in enumerate(sorted(d.to_dict(), key=d.get)):
                vec[col][k] = i + 1
            vec[col][-999] = self.set_nans
        self.vec = vec
 
    def transform(self, df, cols=[]):
        """Transforms a dataframe with a vectorizer.
        Args:
            df: a Pandas dataframe.
            cols: a list (or 1-D Numpy array) of strings with column headers. Default is `[]`.
        Returns:
            df: a Pandas dataframe where specified columns are vectorized.
        Raises:
            AttributeError: Transformation was attempted before fitting the vectorizer.
        """
        try:
            self.vec
        except AttributeError:
            import sys
 
            sys.exit(
                "AttributeError. `self.vec` is not set. Use .fit() before transforming."
            )
 
        if self.verbose > 0:
            print(
                "Labelcount transforming columns: %s on dataframe shaped %sx%s"
                % (cols, df.shape[0], df.shape[1])
            )
 
        for col in cols:
            if self.verbose > 0:
                print(
                    "Column: %s\tCardinality: %s" % (col.rjust(20), df[col].nunique())
                )
            df[col].fillna(-999, inplace=True)
            df[col] = df[col].apply(
                lambda x: self.vec[col][x] if x in self.vec[col] else 1
            )
        return df
 
    def fit_transform(self, df, cols=[]):
        """Calls fit then calls transform in one line.
        Args:
            df: a Pandas dataframe.
            cols: a list (or 1-D Numpy array) of strings with column headers. Default is `[]`.
        Returns:
            df: a Pandas dataframe where specified columns are vectorized.
        """
        self.fit(df, cols)
        return self.transform(df, cols)
 
 
 
def read_and_save_into_s3(path='s3://mntz-datascience/',
                          main_path_s3 = 'path',
                          file_name='training',
                          remove_feat=['vlr']):
 
    temp = pd.read_parquet(path)
    #print(f'Inicial {temp.shape}')
    dt_f = temp.drop(remove_feat, axis=1)
    #print(f'Final {dt_f.shape}')
   
    dt_f.to_parquet(main_path_s3 + file_name + '.parquet')
 
def progressive_feature_selection(X_train, X_test, y_train, y_test, feature_list):
    """
    Function to progressively add features to the model and calculate AUC.
   
    Parameters:
    - model: The machine learning model to be used.
    - X: DataFrame with the features.
    - y: Target variable.
    - feature_list: List of feature names to be considered.
   
    Returns:
    - selected_features: List of features that were kept after the progressive addition.
    - auc_history: List of AUC scores after each iteration.
    """
    # Initialize variables
    selected_features = []
    auc_history = []
    best_auc = 0  # Start with 0 as the baseline AUC
 
    # Iterate over the features to progressively add them
    #for feature in feature_list:
    for feature in tqdm(feature_list, desc="Progressive Feature Selection"):
        # Temporarily add the feature to the selected features
        current_features = selected_features + [feature]
       
        # Add the model
        params = {"force_col_wise": True, "verbosity": -1}
        model = LGBMClassifier(**params).fit(X_train, y_train)
 
        # Train the model with the current set of features
        model.fit(X_train[current_features], y_train)
 
        # Predict probabilities and calculate AUC
        y_pred_proba = model.predict_proba(X_test[current_features])[:, 1]
        current_auc = roc_auc_score(y_test, y_pred_proba)
 
        # Compare the AUC: if it improves or stays the same, keep the feature
        if current_auc >= best_auc:
            selected_features.append(feature)  # Keep the feature
            best_auc = current_auc  # Update the best AUC
            #print(f"Feature '{feature}' added. AUC improved to: {best_auc:.4f}", end='\r')
        else:
            pass
            #print(f"Feature '{feature}' removed. AUC did not improve.", end= '')
 
        # Record the AUC history
        auc_history.append(best_auc)
 
    return selected_features, auc_history
 
def ks_score(y, y_pred):
    return (ks_2samp(y_pred[y == 1], y_pred[y != 1]).statistic)
 
 
def train_LGBM(X_train, y_train, X_test, y_test, X_val, y_val, params = {"force_col_wise": True, "verbosity": -1}):
 
    test_auc = np.zeros(1)
    val_auc = np.zeros(1)
    test_ks = np.zeros(1)
    val_ks = np.zeros(1)
 
    model = LGBMClassifier(**params).fit(X_train, y_train)
 
    y_pred_test = model.predict_proba(X_test)[:, 1]
    y_pred_val = model.predict_proba(X_val)[:, 1]
 
    test_auc = roc_auc_score(y_test, y_pred_test)
    val_auc = roc_auc_score(y_val, y_pred_val)
    test_ks = ks_score(y_test, y_pred_test)
    val_ks = ks_score(y_val, y_pred_val)
 
 
    metrics = {
        "test_auc": test_auc,
        "val_auc": val_auc,
        "test_ks": test_ks,
        "val_ks": val_ks,
        }
 
    results = {"N_features": len(X_train.columns),
               "test_auc": metrics['test_auc'],
               "val_auc": metrics['val_auc'],
               "test_ks": metrics["test_ks"],
               "val_ks": metrics["val_ks"],
               }
 
    #print(results)
 
    return results, model
 
 
def plot_progressive_curves(df_scores, col1='N_features', col2='test_auc', col3='val_auc', metric = 'AUC', color_1='darkblue', color_2='#1e90ff'):    
 
    fig1 = go.Scatter(x=df_scores[col1],
                      y=df_scores[col2]*100,
                      mode='lines', name= 'Test_'+metric, line=dict(dash='solid',color=color_1)
                      )
 
    fig2 = go.Scatter(x=df_scores[col1],
                      y=df_scores[col3]*100,
                      mode='lines', name= 'Val_'+metric, line=dict(dash='dot',color=color_2)
                      )
 
    fig = go.Figure([fig1, fig2])
 
 
    # Add titles and layout adjustments, with legend inside the plot
    fig.update_layout(
        xaxis_title='Número de Features',
        yaxis_title=f'Métrica {metric}',
        legend_title=f'Curvas de {metric}',
        legend=dict(
            orientation="h",  # Horizontal legend layout
            x=0.5,  # Center the legend horizontally
            y=1.1,  # Place the legend slightly above the plot
            xanchor='center',  # Center the legend with respect to the x position
            yanchor='bottom'   # Align the legend to the bottom of the y position
        ),
        width=1500,  # Width of the plot
        height=500  # Height of the plot
    )
 
    fig.show()
 
 
def map_path_s3(Bucket, Prefix):
   
    # MAPEAR pastas S3
    import boto3
    s3 = boto3.client('s3')
 
    # MAPEAR pastas S3
    filtered_files=[]
 
    response = s3.list_objects_v2(
        Bucket=Bucket,
        Prefix=Prefix
    )
 
    if 'Contents' in response:
            for obj in response['Contents']:
                file_name = obj['Key']
 
                # Ignora arquivos que contenham '_SUCCESS' no nome
                if '_SUCCESS' not in file_name:
                    filtered_files.append(file_name)
 
    else:
        print("O prefixo especificado não contém arquivos ou não foi possível listar os arquivos.")
 
    # Remove duplicatas usando set
    unique_paths = list(set(v.split('part')[0] for v in filtered_files))
 
    # Cria o dicionário com valores únicos
    filtered_files = {i: path for i, path in enumerate(unique_paths)}
   
    paths = {}
    for bs in ['base_treino', 'base_teste', 'base_validacao', 'base_oot']:
        # Filter the files and convert to a list
        filtered_list = list({k: v for k, v in filtered_files.items() if v.endswith(f"{bs}/")}.values())
        paths[bs] = filtered_list[0]  # Assign the first element
       
    # Join back the remaining parts
    s3_path = '/'.join(paths['base_treino'].split('/')[:-2]) + '/'
   
    return filtered_files, paths, s3_path
 
 
def _catboost_selector(model: CatBoostClassifier, train_pool: Pool, features_for_select: list, num_features_to_select: int = 25, steps: int = 50, train_final_model: bool = False, logging_level: str = 'Info') -> dict:
    """
    Select features using CatBoost's built-in feature selection method.
 
    Args:
        model (CatBoostClassifier): The CatBoost model to perform feature selection.
        train_pool (Pool): Data pool to train the feature selector.
        features_for_select (List[str]): List of feature names to consider for selection.
        num_features_to_select (int): Number of features to select.
        steps (int): Number of steps to perform in feature selection.
        train_final_model (bool): Whether to train the final model with the selected features.
        logging_level (str): Logging level to use during feature selection.
 
    Returns:
        Dict: A dictionary containing the names of the selected features.
    """
    model.set_params(train_dir="/tmp/catboost_info")
   
    return model.select_features(train_pool,
                                 features_for_select=features_for_select,
                                 num_features_to_select=num_features_to_select,
                                 steps=steps,
                                 algorithm=EFeaturesSelectionAlgorithm.RecursiveByShapValues,
                                 shap_calc_type=EShapCalcType.Regular,
                                 train_final_model=train_final_model,
                                 logging_level=logging_level
                                )
 
def _drop_correlated_features(X_train: pd.DataFrame, y_train: pd.Series, threshold: float = 0.8) -> list:
    """
    Drop highly correlated features based on the correlation threshold.
 
    Args:
        X_train (pd.DataFrame): The input features of the training data.
        y_train (pd.Series): The target variable of the training data.
        threshold (float): The correlation threshold to consider.
 
    Returns:
        List[str]: List of feature names after dropping correlated features.
    """
    feats = X_train.columns.tolist()
    tr = SmartCorrelatedSelection(variables=feats,
                                  method='spearman',
                                  threshold=threshold,
                                  missing_values="ignore",
                                  selection_method="model_performance",
                                  estimator=LGBMClassifier(n_estimators=200, random_state=0, n_jobs=-1, verbose=-1)
                                 )
 
    tr.fit(X_train, y_train)
 
    return [f for f in feats if f not in tr.features_to_drop_]
 
def _catboost_select_features(X_train: pd.DataFrame, y_train: pd.Series, features: list, n_feats_list: list) -> dict:
    """
    Wrapper function to perform feature selection using CatBoost for different numbers of top features.
 
    Args:
        X_train (pd.DataFrame): The input features of the training data.
        y_train (pd.Series): The target variable of the training data.
        features (List[str]): List of feature names to be used in the model.
        n_feats_list (List[int]): List specifying the number of top features to select in different runs.
 
    Returns:
        Dict[str, List[str]]: Dictionary where keys are feature set labels and values are lists of selected feature names.
    """
    print('Starting off CATBOOST Feature Selection. It may take a few minutes...please wait')
   
    if isinstance(features, pd.Index):
        features = features.tolist()
       
    catboost_features = {}      
    model = CatBoostClassifier(allow_writing_files=False, iterations=600, random_seed=0)      
    train_pool = Pool(X_train, y_train, feature_names=features)
 
    for n in n_feats_list:
        print(f'Number of features: {n}')
        summary = _catboost_selector(model = model,
                                     train_pool = train_pool,
                                     features_for_select = features,
                                     num_features_to_select=n,
                                     logging_level='Silent')
       
        catboost_features[f'catboost_top{n}'] = summary['selected_features_names']
        print(f'Top{n} features has been selected')
        print()
    print('...Catboost Selection Features is done!')
    print()
    return catboost_features
 
def _boruta_select_features(X_train: pd.DataFrame, y_train: pd.Series, features: list, perc_list: list) -> dict:
    """
    Perform feature selection using the Boruta algorithm with different percentile of relevant features.
 
    Args:
        X_train (pd.DataFrame): The input features of the training data.
        y_train (pd.Series): The target variable of the training data.
        perc_list (List): List of percentiles of relevant features to consider.
 
    Returns:
        Dict: Dictionary containing the selected features for each percentile.
    """
    print('Starting off BORUTA Feature Selection. It may take a few minutes...please wait')
 
    # preprocessing
    X_train=X_train[features]
    X_train.replace([np.inf, -np.inf], np.nan, inplace=True)
    X_train.fillna(-99999, inplace=True)
   
    boruta_features = {}
 
    # Create a random forest classifier
    #rf = RandomForestClassifier(n_jobs=-1, max_depth=6, random_state=0)
    model = LGBMClassifier(n_estimators=200, random_state=0, n_jobs=-1, verbose=-1)
 
    for perc in perc_list:
        print(f'Porcentage chosen: {perc}%')
        # Create a Boruta feature selection object        
        boruta = BorutaPy(model, n_estimators=200, verbose=0, random_state=0, perc=perc)
        #boruta = BorutaPy(model, n_estimators='auto', verbose=0, random_state=0, perc=perc)
 
 
        # Fit the Boruta algorithm to the data
        boruta.fit(X_train.fillna(-1).values, y_train)
 
        # Check the selected features
        selected_features_boruta = X_train.iloc[:,boruta.support_].columns.tolist()
        tentative_features_boruta = X_train.iloc[:,boruta.support_weak_].columns.tolist()
 
        boruta_features[f'boruta_{perc}perc'] = selected_features_boruta
 
        # Print the selected features
        print('Boruta features (perc = %d%%):' % perc)
        print('Selected features:', selected_features_boruta)
        print('Tentative features', tentative_features_boruta)
        print()
    print('...Boruta Feature Selection is done!')
    print()
 
    return boruta_features
 
def stratified_function(X_train, y_train):    
    # Assuming X_train and y_train are already defined
    n_samples = 32000
    sss = StratifiedShuffleSplit(n_splits=1, train_size=n_samples, random_state=1)
 
    # StratifiedShuffleSplit returns indices, so we index the original arrays
    for train_index, _ in sss.split(X_train, y_train):
        X_train_stratified = X_train.iloc[train_index]
        y_train_stratified = y_train.iloc[train_index]
        #X_test_stratified = X_train.iloc[test_index]
        #y_test_stratified = y_train.iloc[test_index]
    print(X_train.shape, y_train.shape)
   
    return X_train, y_train
 
def _mrmr_select_features(X_train: pd.DataFrame, y_train: pd.Series, features: list, n_feats_list: list) -> dict:
    """
    Perform feature selection using the Maximum Relevance Minimum Redundancy algorithm.
 
    Args:
        X_train (pd.DataFrame): The input features of the training data.
        y_train (pd.Series): The target variable of the training data.
        n_feats_list (List): List of numbers of top features to select.
 
    Returns:
        Dict: Dictionary containing the selected features for each number of top features.
    """
    print('Starting off MRMR Feature Selection. It may take a few minutes...please wait')
    # preprocessing    
    X_train=X_train[features]
    X_train.replace([np.inf, -np.inf], np.nan, inplace=True)
    X_train.fillna(-99999, inplace=True)
   
    mrmr_features = {}
    n_max = max(n_feats_list)    
    mrmr = MaximumRelevanceMinimumRedundancy(k=n_max, kind='auto', redundancy_func='p', relevance_func='f')
 
    try:
        # Attempt to fit mrmr on the full data
        mrmr.fit(X_train, y_train)
       
    except MemoryError:
        # If MemoryError occurs, use stratified sampling
        X_train, y_train = stratified_function(X_train, y_train)
        print(X_train.shape, y_train.shape)
        mrmr.fit(X_train, y_train)  
       
    for n in n_feats_list:
        print(f'Number of features: {n}')
        mrmr_features[f'mrmr_top{n}'] = X_train.columns[mrmr.selected_features_[:n]].tolist()
        print(f'Top{n} features has been selected')
        print()
    print('...MRMR Feature Selection is done!')
    print()
 
    return mrmr_features
 
def select_features(df_train: pd.DataFrame,
                    target_col: str,
                    filtered_features: list,
                    filtered_features_catboost: list,
                    filtered_features_boruta: list,
                    filtered_features_mrmr: list,
                    params: dict,
                    classifier: list) -> dict:
    """
    Coordinate the feature selection process using different methods (CatBoost, Boruta, and MRMR) specified in params.
 
    Args:
        df_train (pd.DataFrame): The training dataset.
        filtered_features (List[str]): List of pre-filtered feature names.
        params (Dict[str, Any]): Dictionary containing settings for different feature selection methods.
 
    Returns:
        Dict[str, List[str]]: A comprehensive dictionary of selected features by each method.
    """
 
    # Load parameters
    target_col = params['modeling']['target_col']
    #target_months = params['modeling']['target_months']
    catboost_n_feats_list = params['modeling']['feat_selection']['catboost_n_feats_list']    
 
    boruta_perc_list = params['modeling']['feat_selection']['boruta_perc_list']
 
    mrmr_n_feats_list = params['modeling']['feat_selection']['mrmr_n_feats_list']
 
    cols2drop = params['modeling']['features_to_drop']
    print('Columns to drop:')
    pprint([col for col in filtered_features if any(f in col for f in cols2drop)])
    print()
    #filtered_features = [col for col in filtered_features if any(f in col for f in cols2drop)]
    filtered_features = [col for col in filtered_features if col not in cols2drop]
 
 
    # Define the target column
    target = target_col
    #target = f'{target_col}_{target_months}m'
 
    # Define the features and target
    #X_train = df_train[filtered_features]
    #print(X_train.shape)
    #y_train = df_train[target]
    #print(y_train.shape)
 
    # Select features
    selected_features = {}
 
    # Select features based on the classifiers listed
    if 'catboost' in classifier:
 
        filtered_features_catboost = [feature for feature in filtered_features_catboost if feature != target]
 
        X_train = df_train[filtered_features_catboost]
        y_train = df_train[target]
        catboost_features = _catboost_select_features(X_train=X_train, y_train=y_train, features=filtered_features_catboost, n_feats_list=catboost_n_feats_list)
        selected_features.update(catboost_features)
        #selected_features['catboost'] = catboost_features
   
    if 'boruta' in classifier:
 
        filtered_features_boruta = [feature for feature in filtered_features_boruta if feature != target]
 
        X_train = df_train[filtered_features_boruta]
        y_train = df_train[target]
        boruta_features = _boruta_select_features(X_train=X_train, y_train=y_train, features=filtered_features_boruta, perc_list=boruta_perc_list)
        selected_features.update(boruta_features)
        #selected_features['boruta'] = boruta_features
   
    if 'mrmr' in classifier:
        # Suppress warnings
        warnings.filterwarnings("ignore")
 
        filtered_features_mrmr = [feature for feature in filtered_features_mrmr if feature != target]
 
        X_train = df_train[filtered_features_mrmr]
        y_train = df_train[target]
        mrmr_features = _mrmr_select_features(X_train=X_train, y_train=y_train, features=filtered_features_mrmr, n_feats_list=mrmr_n_feats_list)
        selected_features.update(mrmr_features)
        #selected_features['mrmr'] = mrmr_features
   
    return selected_features    
    #return {**catboost_features, **boruta_features, **mrmr_features}
 
def drop_correlated_features(df_train: pd.DataFrame, selected_features: Dict[str, List[str]], params: Dict[Any, Any]) -> Dict[str, List[str]]:
    """
    Drop correlated features from the selected feature sets.
 
    Args:
        X_train (pd.DataFrame): The input features of the training data.
        y_train (pd.Series): The target variable of the training data.
        selected_features (Dict[str, List[str]]): Dictionary containing the selected features by each method.
 
    Returns:
        Dict[str, List[str]]: Dictionary containing the uncorrelated features for each method.
    """
 
    # Load parameters
    target_col = params['modeling']['target_col']
    #target_months = params['modeling']['target_months']
    max_corr = params['modeling']['max_correlation']
 
    # Define the target column
    target = target_col
    #target = f'{target_col}_{target_months}m'
 
    # Define the features and target
    X_train = df_train.drop(columns=[target])
    y_train = df_train[target]
 
    uncorrelated_features = {}    
    for method, features in selected_features.items():
       
        try:
            uncorrelated_feats = _drop_correlated_features(X_train[features], y_train, max_corr)
            print(f'{method}: Dropped {len(features) - len(uncorrelated_feats)} correlated features')
            uncorrelated_features[method] = uncorrelated_feats
        except Exception as e:
            print(f'Error running the code for the method {method}: {str(e)}')
            uncorrelated_features[method] = features
            #pass
       
    #pprint(uncorrelated_features)
    return uncorrelated_features  
 
 
def save_pickle_into_s3(variables_list, path_s3_pickle="mntz-datascience", file_name='rfp_features_list.pkl'):
   
    s3_client = boto3.client('s3')
    # Remover o prefixo "s3://"
    path_without_prefix = path_s3_pickle.replace("s3://", "")
 
    # Separar o bucket_name e a key
    bucket_name, key = path_without_prefix.split("/", 1)
    #print(bucket_name, key)
 
    with tempfile.TemporaryFile() as fp:
        joblib.dump(variables_list, fp)    
        fp.seek(0)
        s3_client.put_object(Body=fp.read(), Bucket=bucket_name, Key=key + file_name)
        print()
        print('The pickle file has been saved into S3')
 
 
#PRINCIPAL FUNCTION
def run_feature_selection(
                          #Paths    
                          path_enriched_training_base='s3://mntz-datascience/',
                          header_training=None,
                          sep_training=None,
                          fraction_sample_train=1.0,
                          random_state_train=42,
   
                          path_enriched_test_base='s3://mntz-datascience/',
                          header_test=None,
                          sep_test=None,  
                          fraction_sample_test=1.0,
                          random_state_test=42,
 
                          path_enriched_oot_base='s3://mntz-datascience/',
                          header_oot=None,
                          sep_oot=None,  
                          fraction_sample_oot=1.0,
                          random_state_oot=42,
 
                          path_enriched_validation_base='s3://mntz-datascience/',
                          header_validation=None,
                          sep_validation=None,  
                          fraction_sample_validation=1.0,
                          random_state_validation=42,
 
                          # S3 Bucket  
                          path_s3 = 's3://mntz-datascience/',
                          name_file_report = 'report_removable variables',
 
                          #MLFlow
                          experiment_name = 'iac_dev', # nome da pasta no MLFlow                            
                          run_name = 'feature_selection', # nome do experimento
                                                 
                          #target
                          target_column = 'target',
                          target_0 = "None", # caso a target 0 seja uma string Ex: 'Bom pagador'
                          target_1 = "None", # caso a target 1 seja uma string Ex: 'Mau pagador'
 
                          # id column
                          id_column = 'cnpj', # nome da coluna ID (Ex: cnpj)
 
                          # list of removing feats
                          active_drop_cols_prefix_new = False,
                          drop_cols = ['cnpj_radical', 'anomes', '_ano', 'mes', 'target'],
 
                          # list of categorial features
                          #categorical_list = ['flag_socio_pf', 'porte'],
                         
                          # reference date for OOT drift
                          date_ref = 'anomes',
                          n_recent_months = 3, # last 3 months for Out-of-time set
 
                          # Critério para a seleção da feature (Correlação)
                          method_corr = 'var', # ou 'auc'
 
                          # thresholds
                          threshold_null = 0.9,                          
                          threshold_high_corr = 0.9,
                          threshold_iv = 0.02,
 
                          # AUC curves  
                          n_features_for_auc_curves = 200,
 
                          # Progressive Feature Elimination
                          n_features_pfe = 150,
 
                          skip_main_steps=False, #skip 4 main steps
                          activate_rfp_module=False, #activate RFP
                          rfp_params={}, #set params for RFP  
                          rfp_methods = ['catboost', 'boruta', 'mrmr'], #methods
 
                          save_into_s3=False, #save into S3  
                          save_into_mlflow=False, #save into S3
                         ):
    #MLFlow
    experiment_id = get_or_create_experiment(experiment_name=experiment_name, artifact_location=path_s3)
    with mlflow.start_run(run_name=run_name, # nome da execução
                          experiment_id=experiment_id,
                          description="Feature selection process from IAC",
                          nested=True # mais de uma execução dentro dela mesmo
                         ):
 
        if skip_main_steps == False:          
   
            print('The feature selection process is running. It may take a few minutes...')        
            print()
           
            print('Reading Training Sample...please wait...')
            # reading Train-Test sets
            df_train = read_dataset(path_base=path_enriched_training_base, header=header_training, sep=sep_training,
                                    fraction_sample=fraction_sample_train, random_state=random_state_train)
           
            if active_drop_cols_prefix_new == True:
                cols_w_new = [feature for feature in df_train.columns if "NEW__" in feature in feature]
                # Saving into S3
                df_w_new = pd.DataFrame(cols_w_new, columns=["COLS_NEW"])
                df_w_new.to_csv(path_s3+'features_with_prefix_new.csv', index=False, sep=';')
                del df_w_new
                data_tuple_new = create_feature_selection_report(cols2drop=cols_w_new,
                                                                 reason_for_removal = 'Features with prefix NEW__ have been intentionally removed to align with project requirements')
                df_train = df_train.drop(cols_w_new, axis=1)
 
            n_vol_train = df_train.shape
            all_features_list = list(df_train.columns)
            initial_features = [x for x in df_train.columns if x not in drop_cols]
            print(f'Training: {df_train.shape}')      
            print()
 
            # VERIFICAR SE PRECISA DESTA ETAPA
            #df_train = data_type_transform(df_train, ['cnpj_radical', 'anomes', '_ano', 'mes', 'target'], ['flag', 'port'])
           
            print('Checking for duplicates... If any exist, they will be removed.')        
            df_train, duplicate_rows = find_and_remove_duplicates(df_train)
            print(f'{duplicate_rows.shape[0]} duplicated row(s) has been detected and removed if needed')
            print()
            print('Filtering only rows that contain binary target')
            n_initial_train_rows = df_train.shape[0]
            df_train, n_rows_removed = filter_binary_target(df_train, target_col=target_column, target_0=target_0, target_1=target_1)
 
            print(f'{n_rows_removed} has been removed due to non-binary target')
            print()
           
            # Feature Selection
            print(f'Total number of features: {len(initial_features)}')
            #print(initial_features)
            print()
 
            # checking null
            #drop_null = values_filter(df_train[initial_features], threshold=threshold_null, option='null')
            print('Starting the selection of features...please wait...')
            print()
            print('Step One: Dropping Null Features [threshold: above 90%]')
            drop_null = remove_high_missing_features(df_train[initial_features], target_col=target_column,threshold=threshold_null)
            drop_null = list(set(drop_null))
            initial_features = [x for x in initial_features if x not in drop_null]    
            print(f'{len(drop_null)} has been removed. Remains {len(initial_features)} features')
            print()
            data_tuple1 = create_feature_selection_report(cols2drop=drop_null, reason_for_removal = 'High Percentage of Null Values (threshold: above 90%)')        
           
            # Dropping columns from Training set        
            df_train = df_train.drop(drop_null, axis=1)
            print(f'First_drop {df_train.shape}')
 
            # iv
            print('Step Two: Information Value (IV) Analysis [threshold: Less than 0.02]')
            num_feats = df_train[initial_features].select_dtypes(include=['int', 'float']).columns.tolist()
            cat_feats = df_train[initial_features].select_dtypes(include=['object']).columns.tolist()
           
 
            #print(num_feats)
            iv_table = iv_woe(data=df_train,
                            target=target_column,
                            vars_=initial_features,
                            numeric_vars=num_feats,
                            categorical_vars=cat_feats,
                            bins=10,
                            show_woe=False)    
 
            drop_iv = list(iv_table[iv_table['IV'] < threshold_iv]['Variable'])
            drop_iv = list(set(drop_iv))
            initial_features = [x for x in initial_features if x not in drop_iv]
            print(f'{len(drop_iv)} has been removed. Remains {len(initial_features)} features')
            print()
            data_tuple3 = create_feature_selection_report(cols2drop=drop_iv, reason_for_removal = 'Low Information Value (threshold: Less than 0.02)')
            df_train = df_train.drop(drop_iv, axis=1)
            print(f'Second_drop {df_train.shape}')
       
            # correlation
            print('Step Three: Correlation Study [threshold: above 0.9]')                
            correlated_pairs = calculate_correlations(df_train[initial_features], column_target=target_column, threshold=threshold_high_corr)                                                      
            drop_high_corr = choose_best_feature_by_auc_or_var(X=df_train[initial_features].drop([target_column], axis=1, errors='ignore'),
                                                            y=df_train[[target_column]],
                                                            correlated_pairs=correlated_pairs,
                                                            criteria=method_corr)
            drop_high_corr = list(set(drop_high_corr))                                                          
            initial_features = [x for x in initial_features if x not in drop_high_corr]    
            print(f'{len(drop_high_corr)} has been removed. Remains {len(initial_features)} features')
            print()
            data_tuple2 = create_feature_selection_report(cols2drop=drop_high_corr, reason_for_removal = 'High Correlation (threshold: above 0.9)')
            df_train = df_train.drop(drop_high_corr, axis=1)
            print(f'Third_drop {df_train.shape}')
 
            # Get numeric columns (int and float)
            numerical_feats = df_train.drop(drop_cols, axis=1, errors='ignore').select_dtypes(include=['int', 'float']).columns
            #print(numerical_feats)
            # Get categorical columns (object)
            categorical_feats = df_train.drop(drop_cols, axis=1, errors='ignore').select_dtypes(include=['object']).columns
            #print(categorical_feats)
 
            # categorical treatment    
            # Apply transformations into the categorical features for train-test df
            print('Encoding process: treatment for categorical features just started off on Train sample...please wait...')
            rc = RankCountVectorizer()
 
            # FIT
            df_train = rc.fit_transform(df_train, cols = categorical_feats)
 
            # PREDICT
            print()
            print('Uploading Test sample and applying Encoding process...please wait...')
            df_test = read_dataset(path_base=path_enriched_test_base, header=header_test, sep=sep_test,
                                   fraction_sample=fraction_sample_test, random_state=random_state_test)
           
            if active_drop_cols_prefix_new == True:
                cols_w_new = [feature for feature in df_test.columns if "NEW__" in feature in feature]
                df_w_new = pd.read_csv(path_s3+'features_with_prefix_new.csv', sep=';')
                cols_w_new = df_w_new['COLS_NEW'].to_list()
                del df_w_new
               
                df_test = df_test.drop(cols_w_new, axis=1)            
           
            # Manter as mesmas colunas que a base de treino após os primeiros fitros
            print(f'Shape treino {df_train.shape}')
            df_test = df_test[df_test.columns.intersection(df_train.columns)]
         
            print(f'Test: {df_test.shape}')
            print()
            df_test = rc.transform(df_test, cols = categorical_feats)  
            print('...the enconding process is done')
            print()
            # Replace "NaN" values with np.nan
            df_train[numerical_feats] = df_train[numerical_feats].applymap(lambda x: np.nan if x == "NaN" else x)
            df_test[numerical_feats] = df_test[numerical_feats].applymap(lambda x: np.nan if x == "NaN" else x)
 
            print('Step Four: Drift Analysis... It may take a few minutes')
            # Drift preprocessing
            base_train_drift, base_oot_drift = drift_preprocessing(df_train, date_ref= date_ref, n_recent_months=n_recent_months)
 
            # Drift analysis
            drop_drift = drift_analysis(base_train_drift, base_oot_drift, df_test, target_drift_name = 'target_drift', target_name_risk = target_column, cols_dropping=drop_cols)
            drop_drift = list(set(drop_drift))
            initial_features = [x for x in initial_features if x not in drop_drift]
            print(f'{len(drop_drift)} has been removed. Remains {len(initial_features)} features')
            print()
            data_tuple4 = create_feature_selection_report(cols2drop=drop_drift, reason_for_removal = 'Drift Treatment')
 
            # Report dropping features
            if active_drop_cols_prefix_new == True:
                feat_names = data_tuple1[0] + data_tuple2[0] + data_tuple3[0] + data_tuple4[0] + data_tuple_new[0]
                explanations = data_tuple1[1] + data_tuple2[1] + data_tuple3[1] + data_tuple4[1] + data_tuple_new[1]
               
           
            else:
                feat_names = data_tuple1[0] + data_tuple2[0] + data_tuple3[0] + data_tuple4[0]
                explanations = data_tuple1[1] + data_tuple2[1] + data_tuple3[1] + data_tuple4[1]
               
 
            base_report_feats_remove = pd.DataFrame({'FEATURE_NAME': feat_names,
                                                     'REASON_FOR_REMOVAL': explanations,}
                                                     )
 
            print('Saving the removal feature report into S3...please wait...')
            base_report_feats_remove.to_csv(path_s3+'feature_selection_four_steps.csv', index=False, sep=';')
            print()
            del base_report_feats_remove
 
            print('Removing all features from the Training Sample to compute the temporary volumetry...please wait...')
            df_train = df_train.drop(drop_drift, axis=1)
            print(f'Initial volumetry {n_vol_train}')
            print(f'Final volumetry {df_train.shape}: {len(initial_features)} features + {len(drop_cols)} from dropping variable(s)')
            print()
 
            del df_train
            del df_test
 
        else: # skip_main_steps == True:
           
            df_train = read_dataset(path_base=path_enriched_training_base, header=header_training, sep=sep_training,
                                    fraction_sample=fraction_sample_train, random_state=random_state_train,)            
           
 
            n_vol_train = df_train.shape
            all_features_list = list(df_train.columns)          
           
            base_report_feats_remove = pd.read_csv(path_s3+'feature_selection_four_steps.csv', sep=';')
            drop_all = base_report_feats_remove['FEATURE_NAME'].to_list()            
            explanation_all = base_report_feats_remove['REASON_FOR_REMOVAL'].to_list()
 
            initial_features = [x for x in all_features_list if x not in drop_cols]
            initial_features = [x for x in initial_features if x not in drop_all]
 
            print('...continue the Feature Selection Process, skipping the 4 main initial steps: Null, Correlation, IV and Drift...')
            df_train = df_train.drop(drop_all, axis=1)
                       
            print(f'Initial volumetry {n_vol_train}')
            print(f'Final volumetry {df_train.shape}: {len(initial_features)} features + {len(drop_cols)} from dropping variable(s)')
            del  df_train
            print()            
 
 
        df_train = read_dataset(path_base=path_enriched_training_base, header=header_training, sep=sep_training,
                                fraction_sample=fraction_sample_train, random_state=random_state_train,
                                selected_cols=initial_features + [target_column])
        df_test = read_dataset(path_base=path_enriched_test_base, header=header_test, sep=sep_test,
                                fraction_sample=fraction_sample_test, random_state=random_state_test,
                                selected_cols=initial_features + [target_column])
        df_val = read_dataset(path_base=path_enriched_validation_base, header=header_validation, sep=sep_validation,
                                fraction_sample=fraction_sample_validation, random_state=random_state_validation,
                                selected_cols=initial_features + [target_column])
        #df_oot = read_dataset(path_base=path_enriched_oot_base, header=header_oot, sep=sep_oot,
        #                        fraction_sample=fraction_sample_oot, random_state=random_state_oot,
        #                        selected_cols=initial_features + [target_column])
 
 
        X_train = df_train.drop(drop_cols, axis=1, errors='ignore')
        y_train = df_train[target_column]
        X_test = df_test.drop(drop_cols, axis=1, errors='ignore')
        y_test = df_test[target_column]
        X_val = df_val.drop(drop_cols, axis=1, errors='ignore')
        y_val = df_val[target_column]
 
        print("Initiating the baseline model to complete the final stage of feature selection...please wait")
        print()
        print("Checking for any categorical features. If found, the encoding process will begin.")
       
        categorical_cols = list(X_train.select_dtypes(include=['object', 'string', 'category']).columns)
        #print(categorical_cols)
 
        if len(categorical_cols) > 0:
            print(f'{len(categorical_cols)} categorical features have been detected. The enconding process is starting...please wait')
            print()
            rc = RankCountVectorizer()
            X_train = rc.fit_transform(X_train, cols=categorical_cols)
            X_train.fillna(np.nan, inplace=True)
            X_test = rc.fit_transform(X_test, cols=categorical_cols)
            X_test.fillna(np.nan, inplace=True)
            X_val = rc.fit_transform(X_val, cols=categorical_cols)
            X_val.fillna(np.nan, inplace=True)
 
        else:
            print('No categorical features are present.')
            print()
 
        print("Preprocessing for the final step... this may take a few minutes.")
        model = LGBMClassifier(**{'force_col_wise': True, "verbosity": -1,})
        print(X_train.shape, y_train.shape)
        model.fit(X_train, y_train)
        print()
        print('Preparing to plot the curves: AUC & KS versus number of features')
        print()
        importances = pd.DataFrame({'Feature': model.feature_name_, 'Importance': model.feature_importances_})
        importances = importances.sort_values(by='Importance', ascending=False, ignore_index=True)
        importances.to_csv(path_s3+'base_feature_importance.csv', index=False, sep=';')
        list_feats = list(importances['Feature'])
        #print(len(list_feats))
 
        # Progressive Modeling
        all_metrics = []
        #for i in range(1, len(list_feats[:50]) + 1):
        if n_features_for_auc_curves != None:
            n_feats_for_curves= n_features_for_auc_curves
        else:
            n_feats_for_curves = len(initial_features)
       
         
        for i in tqdm(range(1, len(list_feats[:n_feats_for_curves]) + 1), desc=f"Progress Bar for {n_feats_for_curves} features"):
            #print("Added" , X_train.iloc[:,i-1].name)    
 
            # Train model
            results, _ = train_LGBM(X_train.iloc[:,:i],
                                    y_train,
                                    X_test.iloc[:,:i],
                                    y_test,
                                    X_val.iloc[:,:i],
                                    y_val,
                                    params = {"force_col_wise": True, "verbosity": -1}
                                    )
 
            all_metrics.append(results)
       
        df_scores = pd.DataFrame(all_metrics)
 
        plot_progressive_curves(df_scores, col1='N_features', col2='test_auc', col3='val_auc', metric = 'AUC')
        plot_progressive_curves(df_scores, col1='N_features', col2='test_ks', col3='val_ks', metric = 'KS', color_1='#1B4D3E', color_2='#3CB371')
 
        print()
        print('Step Five: Progressive Feature Elimination. It may take a few minutes, please wait.')
        print()
       
        # Call the function
 
        if n_features_pfe != None:
            n_features_pfe_curves= n_features_pfe
        else:
            n_features_pfe_curves = len(initial_features)
 
        final_features, auc_history = progressive_feature_selection(X_train=X_train,
                                                                    X_test=X_test,
                                                                    y_train=y_train,
                                                                    y_test=y_test,
                                                                    feature_list=list_feats[:n_features_pfe_curves])        
 
        drop_rfecv = [feature for feature in initial_features if feature not in final_features]
        #print(drop_rfecv)
        drop_rfecv = list(set(drop_rfecv))
        #optimal_num_features = rfecv.n_features_
        #print(f"Optimal number of features: {optimal_num_features} final variables")
        initial_features = [x for x in initial_features if x not in drop_rfecv]
        print(f'{len(drop_rfecv)} has been removed. Remains {len(initial_features)} final features')
        save_pickle_into_s3(variables_list=initial_features, path_s3_pickle=path_s3, file_name='features_list_pfe.pkl')  
        data_tuple5 = create_feature_selection_report(cols2drop=drop_rfecv, reason_for_removal = 'Progressive Feature Elimination process')
 
        if skip_main_steps == False:
            feat_names = feat_names + list(data_tuple5[0])
            explanations = explanations + list(data_tuple5[1])
           
 
        else:
            feat_names = drop_all + list(data_tuple5[0])  
            explanations = explanation_all + list(data_tuple5[1])
       
           
        # Create a DataFrame
        base_report_feats_remove = pd.DataFrame({'FEATURE_NAME': feat_names,
                                                 'REASON_FOR_REMOVAL': explanations,})        
       
        # Plot distribution        
        fig, df_drops_binary = feature_selection_plot(all_features_list, list_of_drops=drop_cols, base_report_feats_remove=base_report_feats_remove, name_target=target_column)
 
       
        uncorrelated_features = None
 
        if activate_rfp_module==True:
            print()
            print('RFP process is activated. It may take a few minutes, please wait')            
            del df_train
            del df_test
            del df_val
 
            # feature importance dataframe
            imp = pd.read_csv(path_s3+'feature_selection_four_steps.csv', sep=';')
            drop_all = imp['FEATURE_NAME'].to_list()            
            temp_initial_features = [x for x in all_features_list if x not in drop_cols]
            initial_features = [x for x in temp_initial_features if x not in drop_all]
 
            if active_drop_cols_prefix_new == True:
                df_w_new = pd.read_csv(path_s3+'features_with_prefix_new.csv', sep=';')
                cols_w_new = df_w_new['COLS_NEW'].to_list()
                del df_w_new
                initial_features = [x for x in initial_features if x not in cols_w_new]                              
 
            #print(imp.shape)
            df_train = read_dataset(path_base=path_enriched_training_base, header=header_training, sep=sep_training,
                                fraction_sample=fraction_sample_train, random_state=random_state_train,
                                selected_cols=initial_features + [target_column])
            print(f' Volumetria Treino: {df_train.shape}')
           
 
            # Lista de features filtradas
            filtered_features = list(df_train.columns)
            categorical_cols = list(df_train[filtered_features].drop(columns=target_column,axis=1).select_dtypes(include=['object', 'string', 'category']).columns)
            rc = RankCountVectorizer()
            df_train = rc.fit_transform(df_train, cols=categorical_cols)
 
            # Chamando a função para selecionar as features
            selected_features = select_features(df_train=df_train,
                                                target_col=target_column,
                                                filtered_features=filtered_features,
                                                filtered_features_catboost=filtered_features,
                                                filtered_features_boruta=filtered_features,
                                                filtered_features_mrmr=list(importances['Feature'])[:350], #MRMR has a limitation for the feature space that's 350 is used
                                                params=rfp_params,
                                                classifier = rfp_methods,)
 
            # Removendo features correlacionadas
            uncorrelated_features = drop_correlated_features(df_train=df_train, selected_features=selected_features, params=rfp_params)          
            save_pickle_into_s3(variables_list=uncorrelated_features, path_s3_pickle=path_s3, file_name='rfp_features_list.pkl')
           
            print()
        else:
            pass      
 
        # Save into S3
        if save_into_s3:
            print('Saving the removal feature report and all datasets with final features into S3...please wait...')
            base_report_feats_remove.to_csv(path_s3+name_file_report+'.csv', index=False, sep=';')
 
            for file_, path_ in [(f'treino_top{len(final_features)}_final_features', path_enriched_training_base),
                                 (f'teste_top{len(final_features)}_final_features', path_enriched_test_base),
                                 (f'validacao_top{len(final_features)}_final_features', path_enriched_validation_base),
                                 (f'oot_top{len(final_features)}_final_features', path_enriched_oot_base)]:
 
                #print(file_, path_)                                      
                read_and_save_into_s3(path=path_,
                                      main_path_s3=path_s3,
                                      file_name=file_,
                                      remove_feat=list(set(list(base_report_feats_remove['FEATURE_NAME']))))
           
        # Save into MLFlow
        if save_into_mlflow:
 
            base_report_feats_remove.to_csv('/tmp/'+name_file_report+'.csv', index=False, sep=';')
            print('...the report has been saved into S3')
            print()
            print('Saving the removal feature report and artifacts into MLFlow...please wait...')
            #mlflow.log_artifact(base_report_feats_remove, "explainability/")
            mlflow.log_artifact('/tmp/'+name_file_report+'.csv', "explainability/")
            mlflow.log_dict(base_report_feats_remove.to_dict(orient="records"), "explainability/removal_features_report.json")
            mlflow.log_figure(fig, "explainability/dropping_features_plot.html")
            print('...the report has been saved into MLFlow')
            print()
       
        print('DONE!')
       
        return base_report_feats_remove, importances, final_features, selected_features, uncorrelated_features
 
