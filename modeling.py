import pandas as pd
import numpy as np
import shap
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import plot_confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn.metrics import roc_auc_score
import optuna
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import brier_score_loss
from sklearn.isotonic import IsotonicRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import QuantileTransformer
from scipy.stats import ks_2samp
 
import joblib
from copy import deepcopy
 
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
 
import warnings
warnings.filterwarnings("ignore")
 
pd.set_option('display.max_rows', None)  # Show all rows
pd.set_option('display.max_columns', None)  # Show all columns
pd.set_option('max_colwidth', None) # Show all columns width
 
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
 
def select_targets(df, target_column = 'target'):    
    n_row_start = df.shape[0]
    #print(f'Initial volumetry {df.shape}')
    df[target_column] = pd.to_numeric(df[target_column], errors='coerce')
    df_t = df[(df[target_column] == 0) | (df[target_column] == 1)]
    n_row_end = df_t.shape[0]
    #print(f'Final volumetry {df.shape} ||| {n_row_start-n_row_end} row(s) that contain(s) non-binary target has(ve) been removed')
    print(f'{n_row_start-n_row_end} row(s) that contain(s) non-binary target has(ve) been removed')
   
    return df_t
 
 
def stratified_sample(df, target_col, fraction=0.5, random_state=10):
   
    """
    Create a stratified sample from a pandas DataFrame.
 
    Parameters:
    df (pd.DataFrame): The DataFrame from which to sample.
    stratify_column (str): The column to stratify on.
    fraction (float): The fraction of the data to sample (between 0 and 1).
    random_state (int, optional): Seed for the random number generator.
 
    Returns:
    pd.DataFrame: A stratified sample of the DataFrame.
    """
    stratified_sample_df, _ = train_test_split(
        df,
        test_size=fraction,
        stratify=df[target_col],
        random_state=random_state
    )
   
    return stratified_sample_df
 
 
def read_dataframe(PATH_FILE, target_col, fraction, random_state, sep_or_delimiter='\t', sample_method=False):
 
    """
    Reads a file and returns its contents as a pandas DataFrame.
 
    This function determines the file type based on the file extension and uses the appropriate
    pandas function to read the file. Supported file types are CSV, Excel (XLSX/XLS), Parquet, and TXT.
 
    Parameters:
    PATH_BASE (str): The base directory path where the file is located.
    PATH_FILE (str): The name of the file to read, including its extension.
    sep_or_delimiter (str): The separator or delimiter to use when reading CSV and TXT files.
                            Default is tab ('\t').
 
    Returns:
    pd.DataFrame: The data from the file as a pandas DataFrame.
 
    Raises:
    ValueError: If the file format is not supported.
    """
    if PATH_FILE.endswith('.csv'):
        df = pd.read_csv(PATH_FILE, sep=sep_or_delimiter)
        #print(df.columns)
        if sample_method:
            df=stratified_sample(df, target_col, fraction, random_state)
    elif PATH_FILE.endswith('.xlsx') or PATH_FILE.endswith('.xls'):
        df = pd.read_excel(PATH_FILE)
        #print(df.columns)
        if sample_method:
            df=stratified_sample(df, target_col, fraction, random_state)
    elif PATH_FILE.endswith('.parquet'):
        df = pd.read_parquet(PATH_FILE)
        #print(df.columns)
        if sample_method:
            df=stratified_sample(df, target_col, fraction, random_state)
    elif PATH_FILE.endswith('.txt'):
        df = pd.read_csv(PATH_FILE, delimiter=sep_or_delimiter)
        #print(df.columns)
        if sample_method:
            df=stratified_sample(df, target_col, fraction, random_state)
    else:
        try:
            df = pd.read_parquet(PATH_FILE)
            if sample_method:
                df=stratified_sample(df, target_col, fraction, random_state)
 
        except:
            raise ValueError(f"Unsupported file format: {PATH_FILE}")
   
    return df
 
 
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
 
 
def eval_model(model, X, y):
   
    prob = model.predict_proba(X)[:,1]
    #probabilities = model.predict_proba(X)[:,1]
   
    return roc_auc_score(y, prob)
 
 
def get_preprocess_lgbm(X_train, X_test, X_val, X_oot, categorical_cols, date_col='ref_date', add_date_col = False):
   
    if add_date_col==True:        
               
        # Save the date_ref column
        date_ref_train = X_train[date_col].reset_index(drop=True)
        date_ref_test = X_test[date_col].reset_index(drop=True)
        date_ref_val = X_val[date_col].reset_index(drop=True)
        date_ref_oot = X_oot[date_col].reset_index(drop=True)
   
        # Remove the date_ref column from training and testing data
        X_train = X_train.drop(columns=[date_col], axis=1).reset_index(drop=True)
        X_test = X_test.drop(columns=[date_col], axis=1).reset_index(drop=True)
        X_val = X_val.drop(columns=[date_col], axis=1).reset_index(drop=True)
        X_oot = X_oot.drop(columns=[date_col], axis=1).reset_index(drop=True)
       
        # col names
        name_features= X_train.columns
   
    rc = RankCountVectorizer()
    X_train = rc.fit_transform(X_train, cols=categorical_cols)
    X_train.fillna(np.nan, inplace=True)
   
    X_test = rc.transform(X_test, cols=categorical_cols)
    X_test.fillna(np.nan, inplace=True)
   
    X_val = rc.transform(X_val, cols=categorical_cols)
    X_val.fillna(np.nan, inplace=True)
   
    X_oot = rc.transform(X_oot, cols=categorical_cols)
    X_oot.fillna(np.nan, inplace=True)    
   
   
    if add_date_col==True:
       
        # Convert scaled arrays back to DataFrames
        X_train = pd.DataFrame(X_train, columns=name_features)
        X_test = pd.DataFrame(X_test, columns=name_features)
        X_val = pd.DataFrame(X_val, columns=name_features)
        X_oot = pd.DataFrame(X_oot, columns=name_features)
       
        # Reinsert the date_ref column
        X_train[date_col] = date_ref_train
        X_test[date_col] = date_ref_test
        X_val[date_col] = date_ref_val
        X_oot[date_col] = date_ref_oot
   
    return X_train, X_test, X_val, X_oot
 
def get_preprocess_xgboost(X_train, X_test, X_val, X_oot, categorical_cols, date_col='ref_date', add_date_col = False):
       
    if add_date_col==True:
 
        # Save the date_ref column
        date_ref_train = X_train[date_col].reset_index(drop=True)
        date_ref_test = X_test[date_col].reset_index(drop=True)
        date_ref_val = X_val[date_col].reset_index(drop=True)
        date_ref_oot = X_oot[date_col].reset_index(drop=True)
   
        # Remove the date_ref column from training and testing data
        X_train = X_train.drop(columns=[date_col], axis=1).reset_index(drop=True)
        X_test = X_test.drop(columns=[date_col], axis=1).reset_index(drop=True)
        X_val = X_val.drop(columns=[date_col], axis=1).reset_index(drop=True)
        X_oot = X_oot.drop(columns=[date_col], axis=1).reset_index(drop=True)
       
        # col names      
        name_features = X_train.columns
   
    rc = RankCountVectorizer()        
    X_train = rc.fit_transform(X_train, cols = categorical_cols)
    X_test = rc.transform(X_test, cols=categorical_cols)
    X_val = rc.transform(X_val, cols=categorical_cols)
    X_oot = rc.transform(X_oot, cols=categorical_cols)
   
    # inf  
    X_train.replace(np.inf, np.nan, inplace=True)
    X_train.replace(-np.inf, np.nan, inplace=True)
    # inf
    X_test.replace(np.inf, np.nan, inplace=True)
    X_test.replace(-np.inf, np.nan, inplace=True)
    # inf
    X_val.replace(np.inf, np.nan, inplace=True)
    X_val.replace(-np.inf, np.nan, inplace=True)
    # inf
    X_oot.replace(np.inf, np.nan, inplace=True)
    X_oot.replace(-np.inf, np.nan, inplace=True)
   
    # scaler
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)  
    X_val = scaler.transform(X_val)
    X_oot = scaler.transform(X_oot)
   
   
    if add_date_col==True:        
        # Convert scaled arrays back to DataFrames
        X_train = pd.DataFrame(X_train, columns=name_features)
        X_test = pd.DataFrame(X_test, columns=name_features)
        X_val = pd.DataFrame(X_val, columns=name_features)
        X_oot = pd.DataFrame(X_oot, columns=name_features)
       
        # Reinsert the date_ref column
        X_train[date_col] = date_ref_train
        X_test[date_col] = date_ref_test
        X_val[date_col] = date_ref_val
        X_oot[date_col] = date_ref_oot
   
    return X_train, X_test, X_val, X_oot
 
def get_preprocess_catboost(X_train, X_test, X_val, X_oot, date_col='ref_date', add_date_col = False):  
   
    if add_date_col==True:
       
        # Save the date_ref column
        date_ref_train = X_train[date_col].reset_index(drop=True)
        date_ref_test = X_test[date_col].reset_index(drop=True)
        date_ref_val = X_val[date_col].reset_index(drop=True)
        date_ref_oot = X_oot[date_col].reset_index(drop=True)
   
        # Remove the date_ref column from training and testing data
        X_train = X_train.drop(columns=[date_col], axis=1).reset_index(drop=True)
        X_test = X_test.drop(columns=[date_col], axis=1).reset_index(drop=True)
        X_val = X_val.drop(columns=[date_col], axis=1).reset_index(drop=True)
        X_oot = X_oot.drop(columns=[date_col], axis=1).reset_index(drop=True)        
               
        # col names
        name_features= X_train.columns
       
        # Convert scaled arrays back to DataFrames
        X_train = pd.DataFrame(X_train, columns=name_features)
        X_test = pd.DataFrame(X_test, columns=name_features)
        X_val = pd.DataFrame(X_val, columns=name_features)
        X_oot = pd.DataFrame(X_oot, columns=name_features)
       
        # Reinsert the date_ref column
        X_train[date_col] = date_ref_train
        X_test[date_col] = date_ref_test
        X_val[date_col] = date_ref_val
        X_oot[date_col] = date_ref_oot      
 
    cat_features = np.where(X_train.dtypes == np.object)[0]      
    cat_vars = list(X_train.select_dtypes(include='object'))
    X_train[cat_vars] = X_train[cat_vars].fillna('NA')
    X_test[cat_vars] = X_test[cat_vars].fillna('NA')
    X_val[cat_vars] = X_val[cat_vars].fillna('NA')
    X_oot[cat_vars] = X_oot[cat_vars].fillna('NA')        
                   
    return X_train, X_test, X_val, X_oot, cat_features
 
def get_preprocess_logit(X_train, X_test, X_val, X_oot, categorical_cols, date_col='ref_date', add_date_col = False):  
   
    if add_date_col==True:
       
        # Save the date_ref column
        date_ref_train = X_train[date_col].reset_index(drop=True)
        date_ref_test = X_test[date_col].reset_index(drop=True)
        date_ref_val = X_val[date_col].reset_index(drop=True)
        date_ref_oot = X_oot[date_col].reset_index(drop=True)
   
        # Remove the date_ref column from training and testing data
        X_train = X_train.drop(columns=[date_col], axis=1).reset_index(drop=True)
        X_test = X_test.drop(columns=[date_col], axis=1).reset_index(drop=True)
        X_val = X_val.drop(columns=[date_col], axis=1).reset_index(drop=True)
        X_oot = X_oot.drop(columns=[date_col], axis=1).reset_index(drop=True)
       
        # col names
        name_features= X_train.columns
       
    rc = RankCountVectorizer()        
    X_train = rc.fit_transform(X_train, cols = categorical_cols)
    X_train.replace(np.inf, np.nan, inplace=True)
    X_train.replace(-np.inf, np.nan, inplace=True)
 
    X_test = rc.transform(X_test, cols = categorical_cols)
    X_test.replace(np.inf, np.nan, inplace=True)
    X_test.replace(-np.inf, np.nan, inplace=True)
   
    X_val = rc.transform(X_val, cols = categorical_cols)
    X_val.replace(np.inf, np.nan, inplace=True)
    X_val.replace(-np.inf, np.nan, inplace=True)
   
    X_oot = rc.transform(X_oot, cols = categorical_cols)
    X_oot.replace(np.inf, np.nan, inplace=True)
    X_oot.replace(-np.inf, np.nan, inplace=True)
 
    # Replace NaN values in multiple columns with -99_999
    X_train.fillna(-99999, inplace=True)
    X_test.fillna(-99999, inplace=True)
    X_val.fillna(-99999, inplace=True)
    X_oot.fillna(-99999, inplace=True)
    # Scaler
    scaler = MaxAbsScaler()
    X_train = scaler.fit_transform(X_train)    
    X_test = scaler.transform(X_test)
    X_val = scaler.transform(X_val)
    X_oot = scaler.transform(X_oot)
   
    if add_date_col==True:
       
        # Convert scaled arrays back to DataFrames
        X_train = pd.DataFrame(X_train, columns=name_features)
        X_test = pd.DataFrame(X_test, columns=name_features)
        X_val = pd.DataFrame(X_val, columns=name_features)
        X_oot = pd.DataFrame(X_oot, columns=name_features)
       
        # Reinsert the date_ref column
        X_train[date_col] = date_ref_train
        X_test[date_col] = date_ref_test
        X_val[date_col] = date_ref_val
        X_oot[date_col] = date_ref_oot
   
    return X_train, X_test, X_val, X_oot
 
 
def baseline_model(df_train, df_test,
                   df_val, df_oot,
                   features, target_col, categorical_cols, classifier = 'logit', date_col='data_ref'):    
 
    features.remove(date_col)
    X_train = df_train[features].copy()
    y_train = df_train[target_col].copy()
    X_test = df_test[features].copy()
    y_test = df_test[target_col].copy()
    X_val = df_val[features].copy()
    y_val = df_val[target_col].copy()
    X_oot = df_oot[features].copy()
    y_oot = df_oot[target_col].copy()
   
    if classifier == 'lgbm':                
        (X_train, X_test, X_val, X_oot) = get_preprocess_lgbm(X_train=X_train, X_test=X_test,
                                                              X_val=X_val, X_oot=X_oot,
                                                              categorical_cols=categorical_cols)
        model = LGBMClassifier(**{'force_col_wise': True, "verbosity": -1,})
   
    elif classifier == 'xgboost':
        (X_train, X_test, X_val, X_oot) = get_preprocess_xgboost(X_train=X_train, X_test=X_test,
                                                                 X_val=X_val, X_oot=X_oot,
                                                                 categorical_cols=categorical_cols)
        model = XGBClassifier()
       
    elif classifier == 'catboost':
        (X_train, X_test, X_val, X_oot, cat_features) = get_preprocess_catboost(X_train=X_train, X_test=X_test,
                                                                                X_val=X_val, X_oot=X_oot)
        model = CatBoostClassifier(cat_features=cat_features, allow_writing_files=False, verbose=False)
       
    elif classifier == 'logit':    
        (X_train, X_test, X_val, X_oot) = get_preprocess_logit(X_train=X_train, X_test=X_test,
                                                               X_val=X_val, X_oot=X_oot,
                                                               categorical_cols=categorical_cols)
        model = LogisticRegression()          
       
    else:
        raise ValueError("Invalid classifier type. Supported types are 'lgbm', 'xgboost', 'catboost', and 'logit'.")
   
    model.fit(X_train, y_train)
   
    bl_auc_treino = eval_model(model=model, X=X_train, y=y_train)
    print(f'[Baseline] auc_train {bl_auc_treino}')  
   
    bl_auc_teste = eval_model(model=model, X=X_test, y=y_test)
    print(f'[Baseline] auc_test {bl_auc_teste}')
 
    # NOVAS FEATURE
    bl_auc_validacao = eval_model(model=model, X=X_val, y=y_val)
    print(f'[Baseline] auc_val {bl_auc_validacao}')
 
    bl_auc_oot = eval_model(model=model, X=X_oot, y=y_oot)
    print(f'[Baseline] auc_oot {bl_auc_oot}')
 
    bl_auc_mean = (bl_auc_treino+bl_auc_teste)/2
    bl_auc_std = np.std([bl_auc_treino,bl_auc_teste])
    bl_auc_penalizado = bl_auc_mean - 2*abs(bl_auc_std)
    #print(f'[Baseline] auc_penalizado {bl_auc_penalizado}')
 
    return bl_auc_treino, bl_auc_teste, bl_auc_validacao, bl_auc_oot, bl_auc_penalizado, model, features
 
 
def get_params(trial, classifier='lgbm'):
   
    if classifier == 'lgbm':
        params = {"learning_rate": trial.suggest_loguniform("learning_rate", 0.01, 0.2),
                  "n_estimators": trial.suggest_int("n_estimators", 100, 1000, 100),
                  "num_leaves": trial.suggest_int("num_leaves", 6, 150),
                  "max_depth": trial.suggest_int("max_depth", 5, 10),
                  "min_child_samples": trial.suggest_int("min_child_samples", 50, 500, 5),
                  "min_child_weight": trial.suggest_uniform("min_child_weight", 0.0001, 1000),
                  "boosting_type": "gbdt",
                  "objective": "binary",
                  "colsample_bytree": trial.suggest_uniform("colsample_bytree", 0.4, 1.0),
                  "subsample": trial.suggest_uniform("subsample", 0.5, 1.0),
                  "min_split_gain": trial.suggest_uniform("min_split_gain", 0.01, 0.1),
                  'force_col_wise': True,
                  "verbosity": -1,
                  'random_state': 42,}
       
    elif classifier == 'xgboost':
        params = {"learning_rate": trial.suggest_loguniform("learning_rate", 0.01, 0.2),
                  "n_estimators": trial.suggest_int("n_estimators", 100, 1000, 100),
                  "max_depth": trial.suggest_int("max_depth", 3, 10),
                  "min_child_weight": trial.suggest_uniform("min_child_weight", 0.0001, 1000),
                  "gamma": trial.suggest_uniform("gamma", 0.01, 1.0),
                  "subsample": trial.suggest_uniform("subsample", 0.5, 1.0),
                  "colsample_bytree": trial.suggest_uniform("colsample_bytree", 0.4, 1.0),
                  "reg_alpha": trial.suggest_uniform("reg_alpha", 1e-5, 100),
                  "reg_lambda": trial.suggest_uniform("reg_lambda", 1e-5, 100),
                  "objective": "binary:logistic",
                  "eval_metric": "logloss",
                  'random_state': 42,}
       
       
    elif classifier == 'catboost':
        cat_features=[]
        params = {'depth': trial.suggest_int('depth', 2, 10),
                  'bagging_temperature': trial.suggest_float('bagging_temperature', 0.1, 10),
                  'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
                  'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.3, 1.0),
                  'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 0.01, 100),
                  'iterations': trial.suggest_int('iterations', 10, 1000),
                  'random_strength': trial.suggest_float('random_strength', 0, 100),
                  'verbose': False,
                  'bootstrap_type': 'Bayesian',
                  'cat_features': cat_features,
                  'nan_mode': 'Min',
                  'allow_writing_files': False,
                  'random_state': 42,}  
       
       
    elif classifier == 'logit':
        params = {'C': trial.suggest_float('C', 1e-5, 1000),  # Regularization strength
                  'penalty': trial.suggest_categorical('penalty', ['l1', 'l2', 'elasticnet']),  # Type of regularization
                  'solver': 'saga',  # Optimization algorithm
                  'class_weight': trial.suggest_categorical('class_weight', [None, 'balanced']),  # Class weights
                  'max_iter': trial.suggest_int('max_iter', 100, 800),  # Maximum number of iterations
                  'tol': trial.suggest_float('tol', 1e-5, 1e-2),  # Tolerance for stopping criteria
                  'fit_intercept': trial.suggest_categorical('fit_intercept', [True, False]),  # Include intercept term
                  'l1_ratio': trial.suggest_float('l1_ratio', 0, 1),
                  'random_state': 42,}
 
   
    return params
 
 
def objective(trial, df_train, df_test, df_val, df_oot, features, target_col, categorical_cols, classifier='lgbm', date_col='data_ref'):    
 
    features.remove(date_col)
 
    X_train = df_train[features].copy()
    y_train = df_train[target_col].copy()
    X_test = df_test[features].copy()
    y_test = df_test[target_col].copy()
    X_val = df_val[features].copy()
    y_val = df_val[target_col].copy()
    X_oot = df_oot[features].copy()
    y_oot = df_oot[target_col].copy()
   
    if classifier in ['lgbm']:        
        (X_train, X_test, X_val, X_oot) = get_preprocess_lgbm(X_train=X_train, X_test=X_test,
                                                              X_val=X_val, X_oot=X_oot,
                                                              categorical_cols=categorical_cols)
        model = LGBMClassifier(**get_params(trial=trial, classifier=classifier))
   
    elif classifier in ['xgboost']:
        (X_train, X_test, X_val, X_oot) = get_preprocess_xgboost(X_train=X_train, X_test=X_test,
                                                                 X_val=X_val, X_oot=X_oot,
                                                                 categorical_cols=categorical_cols)      
        # convert to dataframe
        X_train = pd.DataFrame(X_train, columns=features)
        X_test = pd.DataFrame(X_test, columns=features)
        model = XGBClassifier(**get_params(trial=trial, classifier=classifier))
       
    elif classifier in ['catboost']:                
        (X_train, X_test, X_val, X_oot, cat_features) = get_preprocess_catboost(X_train=X_train, X_test=X_test,
                                                                                X_val=X_val, X_oot=X_oot)
        params=get_params(trial=trial, classifier=classifier)
        cat_features = np.where(X_train.dtypes == np.object)[0]
        params['cat_features'] = cat_features # atualiza o dicionário de hyperparâmetros
        model = CatBoostClassifier(**params)        
       
    elif classifier in ['logit']:  
        (X_train, X_test, X_val, X_oot) = get_preprocess_logit(X_train=X_train, X_test=X_test,
                                                               X_val=X_val, X_oot=X_oot,
                                                               categorical_cols=categorical_cols)      
        # convert to dataframe
        X_train = pd.DataFrame(X_train, columns=features)
        X_test = pd.DataFrame(X_test, columns=features)
        model = LogisticRegression(**get_params(trial=trial, classifier=classifier))          
               
    else:
        raise ValueError("Invalid classifier type. Supported types are 'lgbm', 'xgboost', 'catboost', and 'logit'.")
 
    # Seleção de variáveis    
    model.fit(X_train, y_train)
   
    if classifier in ['xgboost', 'catboost']:
        feature_importance = get_shap_importances(model, X_train, features=features, model_name = classifier)
   
    if classifier == 'lgbm':
        feature_importance = get_shap_importances(model, X_train, features=features, model_name = classifier)
       
    elif classifier == 'logit':
        feature_importance = get_shap_importances(model, X_train, features=features, model_name = classifier)
   
    n_features = trial.suggest_int('n_features', 1, X_train.shape[1])
    sel_features = list(feature_importance.sort_values('feature_importance_vals',ascending=False)['col_name'].values[0:n_features])
 
    trial.set_user_attr("sel_features", sel_features)
   
    # Para rodar os experimentos após seleção de features é necessário atualizar os parâmetros; no caso do Catboost é preciso atualizar o cat_features
    if classifier == 'catboost':
        # atualizar cat_features
        cat_features = np.where(X_train[sel_features].dtypes == np.object)[0]
        params['cat_features'] = cat_features # atualiza o dicionário de hyperparâmetros
        model = CatBoostClassifier(**params)
 
    else:
        pass    
 
    model.fit(X_train[sel_features], y_train)
   
    y_pred_train = model.predict_proba(X_train[sel_features])[:, 1]
    y_pred_test = model.predict_proba(X_test[sel_features])[:, 1]
    y_pred_val = model.predict_proba(X_val[sel_features])[:, 1]
    y_pred_oot = model.predict_proba(X_oot[sel_features])[:, 1]
   
    auc_train = roc_auc_score(y_train, y_pred_train)
    auc_test = roc_auc_score(y_test, y_pred_test)
    auc_val = roc_auc_score(y_val, y_pred_val)
    auc_oot = roc_auc_score(y_oot, y_pred_oot)
    print('auc_train', auc_train)
    print('auc_test', auc_test)
    print('auc_val', auc_val)
    print('auc_oot', auc_oot)
 
   
    # Set the AUC values as user attributes of the trial
    trial.set_user_attr('auc_train', auc_train)
    trial.set_user_attr('auc_test', auc_test)
    trial.set_user_attr('auc_val', auc_val)
    trial.set_user_attr('auc_oot', auc_oot)
 
    mean_auc = (auc_train + auc_test)/2
    dp_auc = np.std([auc_train, auc_test])
    penalized_auc = mean_auc - 5*abs(dp_auc)    
   
    return penalized_auc#, auc_train, auc_test
    #return {'penalized_auc': penalized_auc, 'auc_train': auc_train, 'auc_test': auc_test}
 
 
def get_shap_importances(model, X, features=['valor_pago'], model_name = 'xgboost'):
 
    if model_name in ['xgboost', 'catboost']:        
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)    
        vals= np.abs(shap_values).mean(0)
        feature_importance = pd.DataFrame(list(zip(X.columns,vals)),columns=['col_name','feature_importance_vals'])
        feature_importance = feature_importance.sort_values(by=['feature_importance_vals'], ascending=False,inplace=False)
       
    elif model_name in ['lgbm']:        
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        vals= np.abs(shap_values[1]).mean(0)
        feature_importance = pd.DataFrame(list(zip(X.columns,vals)),columns=['col_name','feature_importance_vals'])
        feature_importance = feature_importance.sort_values(by=['feature_importance_vals'], ascending=False,inplace=False)
   
    elif model_name in ['logit']:
        feature_importance = model.coef_[0]
        feature_importance = pd.DataFrame({'col_name': features, 'feature_importance_vals': feature_importance})
        feature_importance = feature_importance.sort_values(by=['feature_importance_vals'], ascending=False, inplace=False)        
   
    return feature_importance
 
 
def get_best_model(report_df, metric = 'AUC_TEST'):      
   
    report_df = report_df[report_df['TRIAL_NUMBER'] != 'baseline'].sort_values(by=[metric], ascending=False).head(1)
   
    return report_df
 
 
def predict(model, x):
    return model.predict_proba(x)[:, 1]
 
 
def compute_metrics_over_time(X_train, X_test, X_val, X_oot, target_col):
    df_metric_ks_train, df_metric_auc_train = None, None
    df_metric_ks_test, df_metric_auc_test = None, None
    df_metric_ks_val, df_metric_auc_val = None, None
    df_metric_ks_oot, df_metric_auc_oot = None, None
   
    try:
        print()
        try:
            df_metric_ks_train = ks_over_time(X_train, target_col, 'PROBA')
            df_metric_auc_train = auc_over_time(X_train, target_col, 'PROBA')
            #print("Train metrics calculated successfully.")
        except Exception as e:
            print(f"Error calculating metrics for training data: {str(e)}")
 
        try:
            df_metric_ks_test = ks_over_time(X_test, target_col, 'PROBA')
            df_metric_auc_test = auc_over_time(X_test, target_col, 'PROBA')
            #print("Test metrics calculated successfully.")
        except Exception as e:
            print(f"Error calculating metrics for test data: {str(e)}")
 
        try:
            df_metric_ks_val = ks_over_time(X_val, target_col, 'PROBA')
            df_metric_auc_val = auc_over_time(X_val, target_col, 'PROBA')
            #print("Validation metrics calculated successfully.")
        except Exception as e:
            print(f"Error calculating metrics for validation data: {str(e)}")
 
        try:
            df_metric_ks_oot = ks_over_time(X_oot, target_col, 'PROBA')
            df_metric_auc_oot = auc_over_time(X_oot, target_col, 'PROBA')
            #print("OOT metrics calculated successfully.")
        except Exception as e:
            print(f"Error calculating metrics for OOT data: {str(e)}")
   
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
   
    return (df_metric_ks_train, df_metric_auc_train, df_metric_ks_test, df_metric_auc_test,
            df_metric_ks_val, df_metric_auc_val, df_metric_ks_oot, df_metric_auc_oot)
 
 
def ks_stats(y, y_pred):
    return ks_2samp(y_pred[y == 1], y_pred[y != 1]).statistic
 
 
def print_overall_metrics(model, X_train, X_test, X_val, X_oot, COL_TARGET):  
   
    try:
        try:
            auc_train = roc_auc_score(X_train[COL_TARGET], X_train['PROBA'])
            auc_train= auc_train*100
            print(f"AUC Treino = {auc_train:.2f}")
        except Exception as e:
            print(f"AUC Treino = Error calculating AUC for training data: {str(e)}")
       
        try:
            auc_test = roc_auc_score(X_test[COL_TARGET], X_test['PROBA'])
            auc_test=auc_test*100
            print(f"AUC Teste = {auc_test:.2f}")
        except Exception as e:
            print(f"AUC Teste = Error calculating AUC for test data: {str(e)}")
       
        try:
            auc_val = roc_auc_score(X_val[COL_TARGET], X_val['PROBA'])
            auc_val=auc_val*100
            print(f"AUC Validação = {auc_val:.2f}")
        except Exception as e:
            print(f"AUC Validação = Error calculating AUC for validation data: {str(e)}")
       
        try:
            auc_oot = roc_auc_score(X_oot[COL_TARGET], X_oot['PROBA'])
            auc_oot=auc_oot*100
            print(f"AUC OOT = {auc_oot:.2f}")
        except Exception as e:
            print(f"AUC OOT = Error calculating AUC for OOT data: {str(e)}")
       
        print()
       
        try:
            ks_train = ks_stats(X_train[COL_TARGET], X_train['PROBA'])
            ks_train=ks_train*100
            print(f"KS Treino = {round(ks_train, 2)}")
        except Exception as e:
            print(f"KS Treino = Error calculating KS for training data: {str(e)}")
       
        try:
            ks_test = ks_stats(X_test[COL_TARGET], X_test['PROBA'])
            ks_test=ks_test*100
            print(f"KS Teste = {round(ks_test, 2)}")
        except Exception as e:
            print(f"KS Teste = Error calculating KS for test data: {str(e)}")
       
        try:
            ks_val = ks_stats(X_val[COL_TARGET], X_val['PROBA'])
            ks_val=ks_val*100
            print(f"KS Validação = {round(ks_val, 2)}")
        except Exception as e:
            print(f"KS Validação = Error calculating KS for validation data: {str(e)}")
       
        try:
            ks_oot = ks_stats(X_oot[COL_TARGET], X_oot['PROBA'])
            ks_oot=ks_oot*100
            print(f"KS OOT = {round(ks_oot, 2)}")
        except Exception as e:
            print(f"KS OOT = Error calculating KS for OOT data: {str(e)}")
   
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
       
    return auc_train, auc_test, auc_val, auc_oot, ks_train, ks_test, ks_val, ks_oot
 
def edit_params(model_name, params):
   
    if model_name in ['catboost']:
        final_params = deepcopy(params)
        try:
            del final_params['n_features']
        except:
            pass
       
    elif model_name in ['xgboost', 'lgbm']:
        final_params = deepcopy(params)
        try:
            del final_params['n_features']
        except:
            pass
       
    elif model_name in ['logit']:
        final_params = deepcopy(params)
        try:
            del final_params['n_features']
        except:
            pass
        # Update the solver to 'saga'
        final_params['solver'] = 'saga'
   
    return final_params      
 
 
def train_best_model(X_train, y_train, parameters={}, features=[], classifier='lgbm'):
   
    if classifier == 'lgbm':
        parameters.update({"force_col_wise": True, "verbosity": -1})
        model = LGBMClassifier(**parameters)
 
    elif classifier == 'xgboost':
        model = XGBClassifier(**parameters)
       
    elif classifier == 'catboost':
        parameters.update({"allow_writing_files": False, "verbose": False})        
        cat_features = np.where(X_train[features].dtypes == np.object)[0]
        parameters['cat_features'] = cat_features # atualiza o dicionário de hyperparâmetros        
        model = CatBoostClassifier(**parameters)
       
    elif classifier == 'logit':
        model = LogisticRegression(**parameters)        
       
    else:
        raise ValueError("Invalid classifier type. Supported types are 'lgbm', 'catboost', 'logit' and 'xgboost'.")
       
    model.fit(X_train[features], y_train)
   
    return model
 
 
def binary_ks_curve(y_true, y_probas):
    """This function generates the points necessary to calculate the KS
    Statistic curve.
 
    Args:
        y_true (array-like, shape (n_samples)): True labels of the data.
 
        y_probas (array-like, shape (n_samples)): Probability predictions of
            the positive class.
 
    Returns:
        thresholds (numpy.ndarray): An array containing the X-axis values for
            plotting the KS Statistic plot.
 
        pct1 (numpy.ndarray): An array containing the Y-axis values for one
            curve of the KS Statistic plot.
 
        pct2 (numpy.ndarray): An array containing the Y-axis values for one
            curve of the KS Statistic plot.
 
        ks_statistic (float): The KS Statistic, or the maximum vertical
            distance between the two curves.
 
        max_distance_at (float): The X-axis value at which the maximum vertical
            distance between the two curves is seen.
 
        classes (np.ndarray, shape (2)): An array containing the labels of the
            two classes making up `y_true`.
 
    Raises:
        ValueError: If `y_true` is not composed of 2 classes. The KS Statistic
            is only relevant in binary classification.
    """
    y_true, y_probas = np.asarray(y_true), np.asarray(y_probas)
    lb = LabelEncoder()
    encoded_labels = lb.fit_transform(y_true)
 
    if len(lb.classes_) != 2:
        raise ValueError('Cannot calculate KS statistic for data with '
                         '{} category/ies'.format(len(lb.classes_)))
 
    idx = encoded_labels == 0
    data1 = np.sort(y_probas[idx])
    data2 = np.sort(y_probas[np.logical_not(idx)])
 
    ctr1, ctr2 = 0, 0
    thresholds, pct1, pct2 = [], [], []
    while ctr1 < len(data1) or ctr2 < len(data2):
 
        # Check if data1 has no more elements
        if ctr1 >= len(data1):
            current = data2[ctr2]
            while ctr2 < len(data2) and current == data2[ctr2]:
                ctr2 += 1
 
        # Check if data2 has no more elements
        elif ctr2 >= len(data2):
            current = data1[ctr1]
            while ctr1 < len(data1) and current == data1[ctr1]:
                ctr1 += 1
 
        else:
            if data1[ctr1] > data2[ctr2]:
                current = data2[ctr2]
                while ctr2 < len(data2) and current == data2[ctr2]:
                    ctr2 += 1
 
            elif data1[ctr1] < data2[ctr2]:
                current = data1[ctr1]
                while ctr1 < len(data1) and current == data1[ctr1]:
                    ctr1 += 1
 
            else:
                current = data2[ctr2]
                while ctr2 < len(data2) and current == data2[ctr2]:
                    ctr2 += 1
                while ctr1 < len(data1) and current == data1[ctr1]:
                    ctr1 += 1
 
        thresholds.append(current)
        pct1.append(ctr1)
        pct2.append(ctr2)
 
    thresholds = np.asarray(thresholds)
    #print(thresholds)
    pct1 = np.asarray(pct1) / float(len(data1))
    pct2 = np.asarray(pct2) / float(len(data2))
 
    if thresholds[0] != 0:
        thresholds = np.insert(thresholds, 0, [0.0])
        pct1 = np.insert(pct1, 0, [0.0])
        pct2 = np.insert(pct2, 0, [0.0])
 
    if thresholds[-1] != 1:
        thresholds = np.append(thresholds, [1.0])
        pct1 = np.append(pct1, [1.0])
        pct2 = np.append(pct2, [1.0])
 
    differences = pct1 - pct2
    ks_statistic, max_distance_at = (np.max(differences),
                                     thresholds[np.argmax(differences)])
 
    return ks_statistic, max_distance_at
 
def ks_over_time(X, target_col= 'target', target_estimate = 'PROB'):
   
    # Initialize df_metricas before the loop
    df_metric_ks = pd.DataFrame()
 
    for x in np.sort(X['data_ref1'].unique()):
        #print(x)
 
        df_i_datarisk = X[X['data_ref1'] == x].copy()
        #print(df_i_datarisk.shape)
        # Calculate KS and create a new DataFrame for the current month's data
 
        new_row = pd.DataFrame({
            'MONTH': [str(x)],
            'KS': binary_ks_curve(df_i_datarisk[target_col], df_i_datarisk[target_estimate])[0],
            'THRESHOLD': binary_ks_curve(df_i_datarisk[target_col], df_i_datarisk[target_estimate])[1]
        })
 
        # Append the new row to df_metricas using concat
        df_metric_ks = pd.concat([df_metric_ks, new_row], ignore_index=True)
        df_metric_ks['THRESHOLD'] = df_metric_ks['THRESHOLD']*1000
 
    return df_metric_ks
 
 
def auc_over_time(X, target_col='target', target_estimate='PROB'):
    # Initialize df_metricas before the loop
    df_metric_auc = pd.DataFrame()
 
    for x in np.sort(X['data_ref1'].unique()):
        # Filter the data for the current month
        df_i_datarisk = X[X['data_ref1'] == x].copy()
 
        # Calculate AUC for the current month's data
        auc = roc_auc_score(df_i_datarisk[target_col], df_i_datarisk[target_estimate])
 
        # Create a new DataFrame for the current month's data
        new_row = pd.DataFrame({
            'MONTH': [str(x)],
            'AUC': [auc]
        })
 
        # Append the new row to df_metricas using concat
        df_metric_auc = pd.concat([df_metric_auc, new_row], ignore_index=True)
 
    return df_metric_auc
 
 
def plot_correlation_matrix(df_base, model_features, style="seaborn", plot_size=(10, 8)):
    with plt.style.context(style=style):
       
        fig, ax = plt.subplots(figsize=plot_size)
       
        df = df_base[model_features].copy()
 
        # Calculate the correlation matrix
        corr = df.corr()
 
        # Generate a mask for the upper triangle
        mask = np.triu(np.ones_like(corr, dtype=bool))
 
        # Draw the heatmap with the mask and correct aspect ratio
        sns.heatmap(
            corr,
            mask=mask,
            cmap="coolwarm",
            vmax=0.3,
            center=0,
            square=True,
            linewidths=0.5,
            annot=True,
            fmt=".2f",
            xticklabels=True,
            yticklabels=True
        )
 
        ax.set_title("Feature Correlation Matrix", fontsize=14)
        plt.tight_layout()
 
    plt.close(fig)
   
    return fig
 
 
def plot_ratings(model, df, model_features, COLUNA_TARGET, quantiles=10, figsize=(12, 6)):
    # Calculate probabilities and categorize
   
    prob_test = (1 - model.predict_proba(df[model_features])[:, 1]) * 1000
   
    try:
        prob_test_cat = pd.qcut(prob_test, q=quantiles)
    except ValueError as e:
        print(f"Error: {e}")        
        prob_test_cat = pd.qcut(prob_test, q=quantiles, labels=False, duplicates='drop')
   
    # Create DataFrame for plotting
    scores_val_test = pd.DataFrame({
        'score': prob_test_cat, #+ 1,  # Shift scores by 1 to start from 1 instead of 0
        'target': df[COLUNA_TARGET].values.astype(int)
    })
   
    # Compute mean target by score
    score_means = scores_val_test.groupby('score')['target'].mean().reset_index()
    score_means['score'] = [f'R{i+1}' for i in range(len(score_means))] # Label scores as R1, R2, etc.
   
   
    relatorio = (scores_val_test.groupby("score")["target"].count()).reset_index()
    relatorio.columns = ["score", "Volume"]
    relatorio['score'] = [f'R{i+1}' for i in range(len(relatorio))] # Label scores as R1, R2, etc.
       
 
    # DEFAULT  
    fig1, ax = plt.subplots(figsize=figsize)
    ax=sns.barplot(x='score', y='target', data=score_means, color='#2b3f4f')
    ax.set_title('Inadimplencia por Rating', fontsize=12)
    ax.set_ylabel('Taxa de inadimplência', fontsize=10)
    ax.set_xlabel('Ratings', fontsize=10)
    ax.grid(True, linestyle=':', linewidth='0.5', color='grey')
   
    # Add labels to each bar
    ax.bar_label(ax.containers[0], label_type='center', fmt='%.3f', padding=1, fontsize=8,
                 fontweight='bold',
                 color='white'
                )
    #plt.show()
   
    # VOLUME
    fig2, ax = plt.subplots(figsize=figsize)
    ax=sns.barplot(x='score', y='Volume', data=relatorio, color="#2077b4")
    ax.set_title('Volume por Cluster', fontsize=14)
    ax.set_ylabel('Volume de registros', fontsize=10)
    ax.set_xlabel('Ratings', fontsize=10)
    ax.grid(True, linestyle=':', linewidth='0.5', color='grey')
    # Add labels to each bar
    ax.bar_label(ax.containers[0], label_type='center', fmt='%.0f', padding=1, fontsize=8,
                 fontweight='bold',
                 color='white'
                )
   
    #sns.color_palette("coolwarm")
 
   
    #plt.show();
   
    return fig1, fig2
    #return ax, score_means, relatorio
 
 
def plot_importance(model_type, model, df, model_features, figsize=(12, 12)):
    if model_type == 'logit':
        imp = pd.DataFrame(model.coef_[0], df[model_features].columns).reset_index()
    else:
        imp = pd.DataFrame(model.feature_importances_, df[model_features].columns).reset_index()
       
    imp.columns = ['var', 'imp']
    imp = imp.sort_values('imp',ascending=False)
    imp['imp'] = (imp['imp'] / imp['imp'].sum()) * 100
    #print(imp['imp'].sum())
 
    sns.set_style('whitegrid')
    fig, ax = plt.subplots(figsize=figsize)
 
    ax = sns.barplot(y="var", x="imp", data=imp.iloc[:80,:], color = '#2b3f4f')
 
    ax.set_xlabel('Importância (em %)', fontsize=16)
    ax.set_ylabel('', fontsize=16)
 
    #ax.set_xticks([x for x in range(0, int(((imp['imp'].max()*3)//2)*2), 2)])
 
    ax.tick_params(axis='both', which='major', labelsize=6)
 
    ax.bar_label(ax.containers[0], label_type='edge', fmt='%.2f', fontsize=8, color='white'
                 #fontweight='bold'
                )
 
    #plt.show()
   
    return fig
 
 
def get_shap_imp(shap_values, model_features,classifier):
   
    if classifier == 'logit':
        rf_resultX=pd.DataFrame(shap_values.values, columns=model_features)
        vals = np.abs(rf_resultX.values).mean(0)
 
        shap_importance = pd.DataFrame(
            list(zip(model_features, vals)),
            columns=['col_name', 'feature_importance_vals']
        )
        shap_importance = shap_importance.sort_values('feature_importance_vals', ascending=False)
        shap_importance['idx'] = shap_importance.index
       
    else:
        rf_resultX = pd.DataFrame(shap_values, columns=model_features)
        vals = np.abs(rf_resultX.values).mean(0)
 
        shap_importance = pd.DataFrame(
            list(zip(model_features, vals)),
            columns=['col_name', 'feature_importance_vals'])
        shap_importance = shap_importance.sort_values('feature_importance_vals', ascending=False)
        shap_importance['idx'] = shap_importance.index
   
    return shap_importance
 
def get_shap_info(model, df, model_features, classifier):
   
    if classifier == 'logit':
       
        explainer = shap.Explainer(model, df[model_features], feature_names=model_features)
        shap_values = explainer(df[model_features])        
       
    else:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(df[model_features])
   
    if classifier == 'lgbm':
        shap_values = shap_values[1]
   
    elif classifier in ['xgboost', 'catboost']:
       
        shap_values = shap_values    
 
    shap_importance = get_shap_imp(shap_values, model_features,classifier)
       
    return (explainer, shap_values, shap_importance)
 
 
 
def plot_summary_shap(model, df, model_features, classifier):
    explainer, shap_values, shap_importance = get_shap_info(model, df, model_features, classifier)
 
    # Create a new figure
    fig, ax = plt.subplots()
    #plt.figure()
   
    # Generate the SHAP summary plot
    ax=shap.summary_plot(
        shap_values,
        df[model_features],
        plot_size=(12, 6),
        #max_display=30,
        show=True,  # Show the plot immediately
        #plot_type='dot'
    )
   
    plt.close()
   
    return fig
 
 
def plot_dependence_plots(model, df, target_col, shap_values, features= ['valor_pago', 'valor_recebido']):
    idx = 0
    for var in features:
        fig,ax=plt.subplots(1,2,figsize=(24,8))
        shap.dependence_plot(ind = var,
                             shap_values = shap_values[1],
                             features = df[features] ,
                             interaction_index=idx,
                             xmax=df[var].quantile(0.75),
                             ax=ax[0],
                             show=False) #  o ponto principal ta aqui, que é o xmin e xmax de cada plot
        ax[0].axhspan(0, shap_values[1][:, idx].max(), facecolor="#e45b5bff", alpha=0.15)
        ax[0].axhspan(shap_values[1][:, idx].min(), 0, facecolor="#4fae4eff", alpha=0.15)
 
        shap.dependence_plot(ind = var,
                             shap_values = shap_values[1],
                             features = df[features],
                             interaction_index=idx,
                             xmin=df[var].quantile(0.75),
                             xmax=df[var].quantile(0.99),
                             ax=ax[1],
                             show=False)
        ax[1].axhspan(0, shap_values[1][:, idx].max(), facecolor="#e45b5bff", alpha=0.15)
        ax[1].axhspan(shap_values[1][:, idx].min(), 0, facecolor="#4fae4eff", alpha=0.15)
        plt.show()
        idx += 1
 
 
def plot_score_distribution(model, df, COL_TARGET, COL_PROB, model_features, figsize=(12, 6)):
   
    df_plot = df.copy()
 
    #df_plot['prob'] = model.predict_proba(df_plot[model_features])[:, 1]
    df_plot['score'] = (1 - model.predict_proba(df_plot[model_features])[:, 1]) * 1000
 
    sns.set_style('whitegrid')
    fig, ax = plt.subplots(figsize=figsize)
 
    sns.distplot(x=df_plot.loc[df_plot[COL_TARGET] == 0]['score'], ax=ax, label='Goods', hist=True)
    sns.distplot(x=df_plot.loc[df_plot[COL_TARGET] == 1]['score'], ax=ax, label='Bads', hist=True)
 
    ax.set_xticks([x for x in range(0, 1001, 100)])
    ax.set_title(f'Distribuição dos Scores (Sem Calibração)', fontsize=16)
    plt.legend(fontsize=14)
   
    print('Sem Calibração')
    print(f"AUC: {roc_auc_score(df_plot[COL_TARGET], df_plot[COL_PROB])*100:.2f}")
    print(f"KS: {ks_2samp(df_plot[COL_PROB][(df_plot[COL_TARGET] == 0)], df_plot[COL_PROB][(df_plot[COL_TARGET] == 1)]).statistic*100:.2f}")
    print(f"Brier Score: {brier_score_loss(df_plot[COL_TARGET], df_plot[COL_PROB]):.6f}")
   
    display(df_plot.groupby(COL_TARGET)['score'].describe(percentiles=[x/10 for x in range(1, 10)]))
    #plt.show()
    return fig
 
   
def plot_isotonic_regression_distribution(model, df_train, df_test, COL_TARGET, COL_PROB, model_features, figsize=(12,6)):
    df_calibrated = df_train[[COL_TARGET]].copy()
    df_calibrated[COL_PROB] = model.predict_proba(df_train[model_features])[:, 1]
 
    iso_reg = IsotonicRegression(y_min=df_calibrated[COL_PROB].min(), y_max=df_calibrated[COL_PROB].max(), out_of_bounds='clip')
    iso_reg.fit(df_calibrated[COL_PROB], df_calibrated[COL_TARGET])
 
    df_plot = df_test.copy()
 
    df_plot[COL_PROB] = model.predict_proba(df_plot[model_features])[:, 1]
    df_plot['score'] = (1 - df_plot[COL_PROB]) * 1000
    df_plot['calibrated_prob'] = iso_reg.predict(df_plot[COL_PROB])
    df_plot['calibrated_score'] = (1 - df_plot['calibrated_prob']) * 1000
 
    sns.set_style('whitegrid')
    fig, ax = plt.subplots(figsize=figsize)
 
    sns.distplot(x=df_plot.loc[df_plot[COL_TARGET] == 0]['calibrated_score'], ax=ax, label='Goods', hist=True)
    sns.distplot(x=df_plot.loc[df_plot[COL_TARGET] == 1]['calibrated_score'], ax=ax, label='Bads', hist=True)
   
    ax.set_xticks([x for x in range(0, 1001, 100)])
    ax.set_title(f'Distribuição dos Scores (Com Regressão Isotônica)', fontsize=16)
    plt.legend(fontsize=14)
   
    print('Regressão Isotônica')
    print(f"AUC: {roc_auc_score(df_plot[COL_TARGET], df_plot['calibrated_prob'])*100:.2f}")
    print(f"KS: {ks_2samp(df_plot['calibrated_prob'][(df_plot[COL_TARGET] == 0)], df_plot['calibrated_prob'][(df_plot[COL_TARGET] == 1)]).statistic*100:.2f}")
    print(f"Brier Score: {brier_score_loss(df_plot[COL_TARGET], df_plot['calibrated_prob']):.6f}")
   
    display(df_plot.groupby(COL_TARGET)['calibrated_score'].describe(percentiles=[x/10 for x in range(1, 10)]))
   
    #plt.show()
   
    return fig
 
 
def plot_sigmoid_calibration_distribution(model, df_train, df_test, COL_TARGET, COL_PROB, model_features, figsize=(12,6)):
    calibrated_clf = CalibratedClassifierCV(model, cv="prefit")
    calibrated_clf.fit(df_train[model_features], df_train[[COL_TARGET]])
 
    df_plot = df_test.copy()
 
    df_plot[COL_PROB] = calibrated_clf.predict_proba(df_plot[model_features])[:, 1]
    df_plot['score'] = (1 - df_plot[COL_PROB]) * 1000
 
    sns.set_style('whitegrid')
    fig, ax = plt.subplots(figsize=figsize)
 
    sns.distplot(x=df_plot.loc[df_plot[COL_TARGET] == 0]['score'], ax=ax, label='Goods', hist=True)
    sns.distplot(x=df_plot.loc[df_plot[COL_TARGET] == 1]['score'], ax=ax, label='Bads', hist=True)
   
    ax.set_xticks([x for x in range(0, 1001, 100)])
    ax.set_title(f'Distribuição dos Scores (Calibração Sigmóide)', fontsize=16)
    plt.legend(fontsize=14)
   
    print('Calibração Sigmóide')
    print(f"AUC: {roc_auc_score(df_plot[COL_TARGET], df_plot[COL_PROB])*100:.2f}")
    print(f"KS: {ks_2samp(df_plot[COL_PROB][(df_plot[COL_TARGET] == 0)], df_plot[COL_PROB][(df_plot[COL_TARGET] == 1)]).statistic*100:.2f}")
    print(f"Brier Score: {brier_score_loss(df_plot[COL_TARGET], df_plot[COL_PROB]):.6f}")
   
    display(df_plot.groupby(COL_TARGET)['score'].describe(percentiles=[x/10 for x in range(1, 10)]))
    #plt.show()
   
    return fig
 
 
def plot_quantile_transform_distribution(model, df_train, df_test, COL_TARGET, COL_PROB, model_features, figsize=(12,6)):
    qt = QuantileTransformer()
   
    score_train = model.predict_proba(df_train[model_features])[:, 1]
    score_train = qt.fit_transform(score_train.reshape((-1, 1)))
   
    df_plot = df_test.copy()
   
    df_plot[COL_PROB] = model.predict_proba(df_plot[model_features])[:, 1]
    df_plot['score'] = (1 - df_plot[COL_PROB]) * 1000
   
    df_plot["calibrated_prob"] = qt.transform(df_plot[COL_PROB].values.reshape((-1, 1)))
    df_plot["calibrated_score"] = (1 - df_plot["calibrated_prob"]) * 1000
 
    sns.set_style('whitegrid')
    fig, ax = plt.subplots(figsize=figsize)
 
    sns.distplot(x=df_plot.loc[df_plot[COL_TARGET] == 0]['calibrated_score'], ax=ax, label='Goods', hist=True)
    sns.distplot(x=df_plot.loc[df_plot[COL_TARGET] == 1]['calibrated_score'], ax=ax, label='Bads', hist=True)
 
    ax.set_xticks([x for x in range(0, 1001, 100)])
    ax.set_title(f'Distribuição dos Scores (Quantile Transformer)', fontsize=16)
    plt.legend(fontsize=14)
   
    print('Quantile Transformer')
    print(f"AUC: {roc_auc_score(df_plot[COL_TARGET], df_plot['calibrated_prob'])*100:.2f}")
    print(f"KS: {ks_2samp(df_plot['calibrated_prob'][(df_plot[COL_TARGET] == 0)], df_plot['calibrated_prob'][(df_plot[COL_TARGET] == 1)]).statistic*100:.2f}")
    print(f"Brier Score: {brier_score_loss(df_plot[COL_TARGET], df_plot['calibrated_prob']):.6f}")
 
    display(df_plot.groupby(COL_TARGET)['calibrated_score'].describe(percentiles=[x/10 for x in range(1, 10)]))
   
    #plt.show()
   
    return fig, df_plot
 
 
def get_ks_statistic(scores, target):
    df_ks = pd.DataFrame()
    df_ks['score'] = scores
    df_ks['target'] = target.astype('int').values
    ks = ks_2samp(df_ks['score'][df_ks['target'] == 0], df_ks['score'][df_ks['target'] == 1]).statistic * 100
   
    return ks
 
def default_plot(df, COL_TARGET, COL_SAFRA, figsize=(16, 9)):
 
    df_comp_temp = df.copy()
    df_comp_temp[['calibrated_score', COL_TARGET]] = df_comp_temp[['calibrated_score', COL_TARGET]].astype('float')
 
    df_results = {'modelo': [], COL_SAFRA: [], 'qtd': [], 'inad': [], 'auc': [], 'ks': []}
    safras_lista = sorted(df_comp_temp[COL_SAFRA].unique().tolist())
 
    for col in ['score']:
        for safra in safras_lista:
 
            df_safra = df_comp_temp[(df_comp_temp[COL_SAFRA]==safra)]
            if df_safra.shape[0] == 0:
                continue
            score = df_safra[col]
 
            qtd_contratos = df_safra.shape[0]
            inad = np.round(df_safra[COL_TARGET].mean() * 100, 2)
 
            auc = 100 - roc_auc_score(df_safra[COL_TARGET].astype('int'), score) * 100
            ks = get_ks_statistic(score, df_safra[COL_TARGET])
 
            for i, k in enumerate(df_results.keys()): df_results[k].append([col, safra, qtd_contratos, inad, auc, ks][i])
    df_results = pd.DataFrame(df_results)
 
    sns.set_style('whitegrid')
    fig, ax = plt.subplots(figsize=figsize)
 
    sns.barplot(ax=ax, x=COL_SAFRA, y="qtd", data=df_results, color='#4e79a7', label='Total de CNPJs')
    #ax.axvspan(safras_lista.index(CORTE_OOT)-0.5, len(safras_lista)-0.5, alpha=0.2, color='#65ab7c', label='Período out-of-time')
 
    ax2 = ax.twinx()
 
    sns.lineplot(x=COL_SAFRA, y='inad', data=df_results, marker='o', ms=6, lw=1, ls='--', color='#d62728', markeredgecolor=None, label='Taxa de Inadimplência', ax=ax2)
 
    for x, y in zip(df_results[COL_SAFRA], df_results['inad']):
        ax2.annotate(f'{y:.1f}%', xy=(x, y), xytext=(0, 10), textcoords='offset points', ha='center', fontsize=10, color='white')
 
    ax.set_title(f'Total de CNPJs e Inadimplencia por Safra\nCNPJs Cobertos: {df_comp_temp.shape[0]:,}', fontsize=16, pad=15)
    ax2.set_yticks([x for x in range(0, int(np.round((df_results['inad'].mean()*3), -1)+1), int(np.round((df_results['inad'].mean()*3), -1)/10))])
    ax.set_yticks([x for x in range(0, int(np.round((df_results['qtd'].mean()*2), -1)+1), int(np.round((df_results['qtd'].mean()*2), -1)/10))])
    ax2.set_ylabel('Taxa de Inadimplência (em %)', fontsize=14, labelpad=15)
    ax.set_ylabel('Total de CNPJs', fontsize=14, labelpad=15)
    ax2.set_xlabel('')
    ax.set_xlabel('')
    ax2.grid(None)
 
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc=0, fontsize=16)
 
    ax.bar_label(ax.containers[0], label_type='edge', fmt='%.0f', padding=6, fontsize=10, fontweight='semibold', color='black')
   
    # Rotate X-axis labels
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0, ha='center', fontsize=8)
   
    #plt.show()
   
    return fig, df_results
 
 
def auc_ks_plot(df_results, COL_TARGET, COL_SAFRA):
    sns.set_style('whitegrid')
    fig, ax = plt.subplots(figsize=(16, 9))
 
    sns.lineplot(x=COL_SAFRA, y='auc', data=df_results, marker='o', ms=6, lw=3, ls='--', color='#4e79a7', markeredgecolor=None, label='AUC', ax=ax)
    sns.lineplot(x=COL_SAFRA, y='ks', data=df_results, marker='o', ms=6, lw=3, ls='--', color='#d62728', markeredgecolor=None, label='KS', ax=ax)
    #ax.axvspan(safras_lista.index(CORTE_OOT)-0.5, len(safras_lista)-0.5, alpha=0.2, color='#65ab7c', label='Período out-of-time')
 
    for x, y in zip(df_results[COL_SAFRA], df_results['auc']):
        ax.annotate(f'{y:.2f}', xy=(x, y), xytext=(0, 10), textcoords='offset points', ha='center', fontsize=10)
 
    for x, y in zip(df_results[COL_SAFRA], df_results['ks']):
        ax.annotate(f'{y:.2f}', xy=(x, y), xytext=(0, 10), textcoords='offset points', ha='center', fontsize=10)
 
    ax.set_yticks([x for x in range(0, 101, 10)])
    ax.set_ylabel('Valor da Métrica', fontsize=14, labelpad=15)
    ax.set_xlabel('')
    ax.set_title('AUC e KS por Safra', fontsize=16, pad=15)
   
    plt.legend(loc='upper right', fontsize=14)
   
    # Rotate the X-axis labels
    plt.xticks(rotation=0, fontsize=8)
 
    #plt.show()
   
    return fig
     
 
def plot_conf_matrix(model, X_test, y_test, color='viridis', title='Confusion Matrix'):
    """
    Train the model and plot the confusion matrix.
    Parameters:
    model: The machine learning model to be trained.
    X_test: Testing data features.
    y_test: Testing data labels.
    color: Colormap for the confusion matrix.
    """
    fig, ax = plt.subplots()
    disp = plot_confusion_matrix(model, X_test, y_test, cmap=color, ax=ax)
    ax.set_title(title)  # Add the title
    plt.grid(False)
    return fig
 
def run_classification_report(y_test, y_pred, output_dict=False):
    # Generate classification report as a dictionary
    report_dict = classification_report(y_test, y_pred, output_dict=output_dict)
 
    # Convert the dictionary to a pandas DataFrame
    report_df = pd.DataFrame(report_dict).transpose()
    return report_df
 
 
def plot_graphs(model_type, model, X_train, X_test, X_val, X_oot, features, n_quantile, target_col, y_test, y_val, y_oot, best_model):
   
    if model_type in ['logit', 'xgboost']:
        features = model.feature_names_in_
       
    elif model_type == 'catboost':        
        features = model.feature_names_
       
    elif model_type == 'lgbm':
        features=model.feature_name_        
       
    fig_plot_1_test=plot_conf_matrix(model, X_test[features], y_test, color='Blues', title='Confusion Matrix [TEST]')
    fig_plot_1_val=plot_conf_matrix(model, X_val[features], y_val, color='Blues', title='Confusion Matrix [VAL]')
    fig_plot_1_oot=plot_conf_matrix(model, X_oot[features], y_oot, color='Blues', title='Confusion Matrix [OOT]')
 
    fig_plot_2_test=plot_correlation_matrix(X_test, features[:20],
                                            #figsize=(20, 10)
                                           )
    fig_plot_2_val=plot_correlation_matrix(X_val, features[:20],
                                           #figsize=(20, 10)
                                          )
    fig_plot_2_oot=plot_correlation_matrix(X_oot, features[:20],
                                           #figsize=(20, 10)
                                          )
 
    fig_plot_3_test, fig_plot_4_test = plot_ratings(model, X_test, model_features = features, COLUNA_TARGET=target_col, quantiles=n_quantile, figsize=(10, 4))
    fig_plot_3_val, fig_plot_4_val = plot_ratings(model, X_val, model_features = features, COLUNA_TARGET=target_col, quantiles=n_quantile, figsize=(10, 4))
    fig_plot_3_oot, fig_plot_4_oot = plot_ratings(model, X_oot, model_features = features, COLUNA_TARGET=target_col, quantiles=n_quantile, figsize=(10, 4))
       
    fig_plot_5_test = plot_importance(model_type, model, X_test, model_features=features, figsize=(25, 15))
    fig_plot_5_val = plot_importance(model_type, model, X_val, model_features=features, figsize=(25, 15))
    fig_plot_5_oot = plot_importance(model_type, model, X_oot, model_features=features, figsize=(25, 15))
   
    #explainer, shap_values, shap_importance = get_shap_info(model=model, df=X_test, model_features=features, classifier=model_type)
    fig_plot_6_test = plot_summary_shap(model, X_test, model_features=features, classifier=model_type)
    fig_plot_6_val = plot_summary_shap(model, X_val, model_features=features, classifier=model_type)
    fig_plot_6_oot = plot_summary_shap(model, X_oot, model_features=features, classifier=model_type)
   
    fig_plot_7_test = plot_score_distribution(model, X_test, COL_TARGET=target_col, COL_PROB='PROBA', model_features=features, figsize=(12, 6))
    fig_plot_7_val = plot_score_distribution(model, X_val, COL_TARGET=target_col, COL_PROB='PROBA', model_features=features, figsize=(12, 6))
    fig_plot_7_oot = plot_score_distribution(model, X_oot, COL_TARGET=target_col, COL_PROB='PROBA', model_features=features, figsize=(12, 6))
       
    fig_plot_8_test = plot_isotonic_regression_distribution(model, X_train, X_test, COL_TARGET=target_col, COL_PROB='PROBA', model_features=features, figsize=(12, 6))
    fig_plot_8_val = plot_isotonic_regression_distribution(model, X_train, X_test, COL_TARGET=target_col, COL_PROB='PROBA', model_features=features, figsize=(12, 6))
    fig_plot_8_oot = plot_isotonic_regression_distribution(model, X_train, X_val, COL_TARGET=target_col, COL_PROB='PROBA', model_features=features, figsize=(12, 6))    
   
    fig_plot_9_test = plot_sigmoid_calibration_distribution(model, X_train, X_test, COL_TARGET=target_col, COL_PROB='PROBA', model_features=features, figsize=(12, 6))
    fig_plot_9_val = plot_sigmoid_calibration_distribution(model, X_train, X_val, COL_TARGET=target_col, COL_PROB='PROBA', model_features=features, figsize=(12, 6))
    fig_plot_9_oot = plot_sigmoid_calibration_distribution(model, X_train, X_oot, COL_TARGET=target_col, COL_PROB='PROBA', model_features=features, figsize=(12, 6))
       
    fig_plot_10_test, df_plot_quantile_test = plot_quantile_transform_distribution(model, X_train, X_test, COL_TARGET=target_col, COL_PROB='PROBA', model_features=features, figsize=(12, 6))
    fig_plot_10_val, df_plot_quantile_val = plot_quantile_transform_distribution(model, X_train, X_val, COL_TARGET=target_col, COL_PROB='PROBA', model_features=features, figsize=(12, 6))
    fig_plot_10_oot, df_plot_quantile_oot = plot_quantile_transform_distribution(model, X_train, X_oot, COL_TARGET=target_col, COL_PROB='PROBA', model_features=features, figsize=(12, 6))
 
    fig_plot_11_test, df_results_test = default_plot(df_plot_quantile_test, COL_TARGET=target_col, COL_SAFRA='data_ref1', figsize=(20, 10))
    fig_plot_11_val, df_results_val = default_plot(df_plot_quantile_val, COL_TARGET=target_col, COL_SAFRA='data_ref1', figsize=(20, 10))
    fig_plot_11_oot, df_results_oot = default_plot(df_plot_quantile_oot, COL_TARGET=target_col, COL_SAFRA='data_ref1', figsize=(20, 10))
   
    fig_plot_12_test = auc_ks_plot(df_results_test, COL_TARGET=target_col, COL_SAFRA='data_ref1')
    fig_plot_12_val = auc_ks_plot(df_results_val, COL_TARGET=target_col, COL_SAFRA='data_ref1')
    fig_plot_12_oot = auc_ks_plot(df_results_oot, COL_TARGET=target_col, COL_SAFRA='data_ref1')
   
    # declarar o RETURN para que a imagem seja salva e inclusa no MLFlow
    return (fig_plot_1_test, fig_plot_1_val, fig_plot_1_oot, fig_plot_2_test, fig_plot_2_val, fig_plot_2_oot, fig_plot_3_test, fig_plot_4_test, fig_plot_3_val, fig_plot_4_val,
            fig_plot_3_oot, fig_plot_4_oot, fig_plot_5_test, fig_plot_5_val, fig_plot_5_oot, fig_plot_6_test, fig_plot_6_val, fig_plot_6_oot, fig_plot_7_test, fig_plot_7_val,
            fig_plot_7_oot, fig_plot_8_test, fig_plot_8_val, fig_plot_8_oot, fig_plot_9_test, fig_plot_9_val, fig_plot_9_oot,
            fig_plot_10_test, fig_plot_10_val, fig_plot_10_oot, fig_plot_11_test, fig_plot_11_val, fig_plot_11_oot,
            fig_plot_12_test, fig_plot_12_val, fig_plot_12_oot)
 
def convert_ndarray_to_list(d):
    if isinstance(d, dict):
        return {k: convert_ndarray_to_list(v) for k, v in d.items()}
    elif isinstance(d, list):
        return [convert_ndarray_to_list(i) for i in d]
    elif isinstance(d, np.ndarray):
        return d.tolist()
    else:
        return d
 
 
# FUNÇÃO PRINCIPAL
def run_modeling(PATH_FILE_dropping_base_report = 's3://',
                 PATH_FILE_training = 's3://',
                 PATH_FILE_test = 's3://',
                 PATH_FILE_validation = 's3://',
                 PATH_FILE_oot = 's3://',
                 sep_or_delimiter = None,
                 sample_method = False,
                 fraction_sample = 1.0,
                 random_state_sample = 10,  
                 target_col = 'TARGET',
                 date_col = 'date_ref',
                 dropping_feats = ['INDEX'],
 
                 # baseline
                 run_baseline = True,                
 
                 # optuna
                 run_optuna = True,
                 n_trials_optuna = 10,
 
                 list_of_estimators = ['logit'],
                 
                 #MLFlow
                 experiment_name = 'ia', # nome da pasta no MLFlow                            
                 run_name = 'model', # nome do experimento
                 path_s3 = 's3://',
                 name_file_report = 'report',
                 
                 save_into_s3=False, #save into S3
                 save_into_mlflow=False, #save into MLFlow
                ):
       
    #MLFlow
    # Experimento principal
    experiment_id = get_or_create_experiment(experiment_name=experiment_name, artifact_location=path_s3)
    with mlflow.start_run(run_name=run_name, # nome da execução
                          experiment_id=experiment_id,
                          description="Modeling process from IAC",
                          nested=True # mais de uma execução dentro dela mesmo
                         ) as run:                  
   
        print('The modeling process is running. It may take a few minutes...')        
        print()  
        print('Reading Dataframes...please wait...')
        #drop_cols = read_dataframe(PATH_FILE=PATH_FILE_dropping_base_report, sep_or_delimiter=sep_or_delimiter)
        drop_cols = read_dataframe(PATH_FILE=PATH_FILE_dropping_base_report,
                                   sep_or_delimiter=sep_or_delimiter,
                                   target_col=target_col,
                                   fraction=fraction_sample,
                                   random_state=random_state_sample,
                                   sample_method=sample_method)
                                   
        drop_cols=list(drop_cols['FEATURE_NAME'])
        final_dropping_feats =  drop_cols + dropping_feats
 
        # Customer database
        #df_customer = read_dataframe(PATH_FILE=PATH_FILE_customer_base,
        #                             sep_or_delimiter=sep_or_delimiter)  
 
        # Training set
        #df_train = read_dataframe(PATH_FILE=PATH_FILE_training, sep_or_delimiter=sep_or_delimiter)
        df_train = read_dataframe(PATH_FILE=PATH_FILE_training,
                                   sep_or_delimiter=sep_or_delimiter,
                                   target_col=target_col,
                                   fraction=fraction_sample,
                                   random_state=random_state_sample,
                                   sample_method=sample_method)
        print()
        print('Checking target format in Training set....')    
        df_train = select_targets(df=df_train, target_column=target_col)
        df_train=df_train.drop(final_dropping_feats, axis=1, errors='ignore')
        print()
 
        # Test set
        #df_test = read_dataframe(PATH_FILE=PATH_FILE_test, sep_or_delimiter=sep_or_delimiter).drop(final_dropping_feats, axis=1)
        df_test = read_dataframe(PATH_FILE=PATH_FILE_test,
                                   sep_or_delimiter=sep_or_delimiter,
                                   target_col=target_col,
                                   fraction=fraction_sample,
                                   random_state=random_state_sample,
                                   sample_method=sample_method)
        print('Checking target format in Test set....')
        df_test = select_targets(df=df_test, target_column=target_col)
        df_test=df_test.drop(final_dropping_feats, axis=1, errors='ignore')
        print()
 
        # Validation set
        #df_val = read_dataframe(PATH_FILE=PATH_FILE_validation, sep_or_delimiter=sep_or_delimiter).drop(final_dropping_feats, axis=1)
        df_val = read_dataframe(PATH_FILE=PATH_FILE_validation,
                                   sep_or_delimiter=sep_or_delimiter,
                                   target_col=target_col,
                                   fraction=fraction_sample,
                                   random_state=random_state_sample,
                                   sample_method=sample_method)                                  
        print('Checking target format in Validation set....')  
        df_val = select_targets(df=df_val, target_column=target_col)
        df_val=df_val.drop(final_dropping_feats, axis=1, errors='ignore')
        print()
        # OOT set
        #df_oot = read_dataframe(PATH_FILE=PATH_FILE_oot, sep_or_delimiter=sep_or_delimiter).drop(final_dropping_feats, axis=1)
        df_oot = read_dataframe(PATH_FILE=PATH_FILE_oot,
                                   sep_or_delimiter=sep_or_delimiter,
                                   target_col=target_col,
                                   fraction=fraction_sample,
                                   random_state=random_state_sample,
                                   sample_method=sample_method)
        print('Checking target format in OOT set....')                            
        df_oot = select_targets(df=df_oot, target_column=target_col)
        df_oot=df_oot.drop(final_dropping_feats, axis=1, errors='ignore')
        print()
        #print(f'Customer dataframe: {df_customer.shape}')
        print(f'Training: {df_train.shape}')
        print(f'Test: {df_test.shape}')
        print(f'Validation: {df_val.shape}')
        print(f'OOT: {df_oot.shape}')
        print()    
 
        # Initialize an empty DataFrame to store the results
        results_report = pd.DataFrame(columns=['MODEL_NAME', 'TRIAL_NUMBER', 'AUC_TRAINING', 'AUC_TEST', 'AUC_PENALIZED', 'AUC_VAL', 'AUC_OOT', 'PARAMS', 'FEATURES'])    
        print('Modeling is in progress. It may take a few minutes to finish...please wait...')
        print()
 
        best_params_dict = {}
        trials_dict = {}
        baseline_models = {}
        for cls in list_of_estimators:
 
            print(f'Algorithm: {cls}')
 
            categorical_cols = list(df_train.select_dtypes(include=['object']).columns)
            categorical_cols.remove(date_col)
 
            if run_baseline:
                study = None
                #trials_dict = None  
 
                # baseline
                (bl_auc_treino, bl_auc_teste, bl_auc_validacao, bl_auc_oot,
                 bl_auc_penalizado, model, features) = baseline_model(df_train=df_train,
                                                                      df_test=df_test,
                                                                      df_val=df_val,
                                                                      df_oot=df_oot,
                                                                      features=list(df_train.drop(target_col,axis=1).columns),
                                                                      target_col=target_col,
                                                                      categorical_cols=categorical_cols,
                                                                      classifier = cls,
                                                                      date_col=date_col)
                baseline_models[cls] = model
                #print(f'Penalized AUC: {best_trial.values[0]}, AUC Train: {best_trial.user_attrs["auc_train"]}, AUC Test: {best_trial.user_attrs["auc_test"]}')
 
                # Append the trial information to the DataFrame
                results_report = results_report.append({'MODEL_NAME': cls,
                                                        'TRIAL_NUMBER': 'baseline',
                                                        'AUC_TRAINING': bl_auc_treino,
                                                        'AUC_TEST': bl_auc_teste,
                                                        'AUC_VAL': bl_auc_validacao,
                                                        'AUC_OOT': bl_auc_oot,
                                                        'AUC_PENALIZED': bl_auc_penalizado,
                                                        'PARAMS': model.get_params(),
                                                        'FEATURES': features,
                                                    },
                                                    ignore_index=True)
                print()
 
            if run_optuna:
                print('Hiperparameter tuning has been set. It may take a few minutes to finish...please wait...')
                print()        
                #Optuna    
                best_params_dict[cls] = {}
                trials_dict[cls] = {}
 
                study = optuna.create_study(direction='maximize')
                study.optimize(lambda trial: objective(trial,
                                                    df_train=df_train,
                                                    df_test=df_test,
                                                    df_val=df_val,
                                                    df_oot=df_oot,
                                                    features=list(df_train.drop(target_col, axis=1).columns),
                                                    target_col=target_col,
                                                    categorical_cols=categorical_cols,
                                                    classifier=cls,
                                                    date_col=date_col),
                            n_trials=n_trials_optuna)
 
                trials_dict[cls] = study.trials
                best_params_dict[cls] = study.best_params
                print()
 
            if run_optuna:
                # Iterate over the trials dictionary
                for classifier, trials in trials_dict.items():
                    for trial in trials:
 
                        # Extract trial-specific AUC values
                        auc_train = trial.user_attrs["auc_train"]
                        auc_test = trial.user_attrs["auc_test"]
                        auc_val = trial.user_attrs["auc_val"]
                        auc_oot = trial.user_attrs["auc_oot"]
                        features_list = trial.user_attrs["sel_features"]
 
                        # Extract trial information
                        trial_number = trial.number
                        trial_value = trial.value
                        trial_params = trial.params
 
                        # Append the trial information to the DataFrame
                        results_report = results_report.append({'MODEL_NAME': classifier,
                                                                'TRIAL_NUMBER': trial_number,
                                                                'AUC_TRAINING': auc_train,
                                                                'AUC_TEST': auc_test,
                                                                'AUC_VAL': auc_val,
                                                                'AUC_OOT': auc_oot,
                                                                'AUC_PENALIZED': trial_value,                                                  
                                                                'PARAMS': trial_params,
                                                                'FEATURES':features_list,
                                                            },
                                                            ignore_index=True)
 
                print('The results of hiperparameter tuning is done')
                print()              
       
        # Save into S3
        if save_into_s3:
            print('Salving the modeling report into S3...please wait...')
            results_report.to_csv(path_s3+name_file_report+'.csv', index=False, sep=';')
            print()
           
        #MLFlow
        if save_into_mlflow:            
            results_report.to_csv('/tmp/'+name_file_report+'.csv', index=False, sep=';')
            print('...the report has been saved into S3')
            print()
            print('Salving the modeling report and artifacts into MLFlow...please wait...')                                        
            mlflow.log_artifact('/tmp/'+name_file_report+'.csv', "explainability/")
 
            #mlflow.log_dict(results_report.to_dict(orient="records"), "explainability/hiperparam_tuning_result.json")
           
            results_dict = results_report.to_dict(orient="records")
            results_dict = convert_ndarray_to_list(results_dict)
            mlflow.log_dict(results_dict, "explainability/hiperparam_tuning_result.json")
            print('...the report has been saved into MLFlow')
            print()
        print('DONE!')
 
        return (df_train, df_test, df_val, df_oot,                
                results_report, baseline_models,
                trials_dict, study,
                experiment_id, run.info.run_id)
 
#FUNÇÃO SECUNDÁRIA
def run_best_model(results_report, df_train, df_test, df_val, df_oot, n_quantile, target_col='target', date_col='data', run_id='', save_into_mlflow=False, get_best_model_manual=False):
       
 
    print('The BEST MODEL run is starting off. It may take a few minutes...')
   
    if get_best_model_manual:
        #print('MANUAL')
       
        best_model = results_report
        params = results_report['PARAMS'].array[0]
        features = results_report['FEATURES'].to_list()[0]
        model_name = results_report['MODEL_NAME'].array[0]
   
    else:
        #print('AUTOMATICO')      
        best_model = get_best_model(results_report, metric = 'AUC_OOT')
        params = best_model['PARAMS'].array[0]
        features = best_model['FEATURES'].to_list()[0]
        model_name = best_model['MODEL_NAME'].array[0]
 
    #print(model_name, params)
    final_params = edit_params(model_name, params)    
    #print(final_params)
 
    X_train = df_train[[date_col]+ features].copy()
    y_train = df_train[target_col].copy()
 
    X_test = df_test[[date_col]+ features].copy()
    y_test = df_test[target_col].copy()
 
    X_val = df_val[[date_col]+ features].copy()
    y_val = df_val[target_col].copy()
 
    X_oot = df_oot[[date_col]+ features].copy()
    y_oot = df_oot[target_col].copy()
 
    categorical_cols = list(X_train[features].select_dtypes(include=['object']).columns)
   
    if model_name == 'logit':
        (X_train_scaled, X_test_scaled,
         X_val_scaled, X_oot_scaled) = get_preprocess_logit(X_train[features+[date_col]], X_test[features+[date_col]],
                                                               X_val[features+[date_col]], X_oot[features+[date_col]],
                                                               categorical_cols, date_col=date_col, add_date_col=True)      
               
    if model_name == 'xgboost':
        (X_train_scaled, X_test_scaled,
        X_val_scaled, X_oot_scaled) = get_preprocess_xgboost(X_train[features+[date_col]], X_test[features+[date_col]],
                                                             X_val[features+[date_col]], X_oot[features+[date_col]],
                                                             categorical_cols, date_col=date_col, add_date_col=True)
    if model_name == 'catboost':
        (X_train_scaled, X_test_scaled,
         X_val_scaled, X_oot_scaled, cat_features) = get_preprocess_catboost(X_train[features+[date_col]], X_test[features+[date_col]],
                                                                             X_val[features+[date_col]], X_oot[features+[date_col]],
                                                                             date_col=date_col, add_date_col=True)                
   
    if model_name == 'lgbm':
        (X_train_scaled, X_test_scaled,
         X_val_scaled, X_oot_scaled) = get_preprocess_lgbm(X_train[features+[date_col]], X_test[features+[date_col]],
                                                              X_val[features+[date_col]], X_oot[features+[date_col]],
                                                              categorical_cols, date_col=date_col, add_date_col=True)
 
 
    # .FIT
    model = train_best_model(X_train_scaled, y_train, parameters=final_params, features=features, classifier=model_name)
 
    # .PREDICT
    X_train_scaled['PROBA'] = model.predict_proba(X_train_scaled[features])[:, 1]
    X_train_scaled['PRED'] = model.predict(X_train_scaled[features])
    X_train_scaled[target_col] = y_train.reset_index(drop=True)
 
    X_test_scaled['PROBA'] = model.predict_proba(X_test_scaled[features])[:, 1]
    X_test_scaled['PRED'] = model.predict(X_test_scaled[features])
    X_test_scaled[target_col] = y_test.reset_index(drop=True)
 
    X_val_scaled['PROBA'] = model.predict_proba(X_val_scaled[features])[:, 1]
    X_val_scaled['PRED'] = model.predict(X_val_scaled[features])
    X_val_scaled[target_col] = y_val.reset_index(drop=True)
 
    try:
        X_oot_scaled['PROBA'] = model.predict_proba(X_oot_scaled[features])[:, 1]
        X_oot_scaled['PRED'] = model.predict(X_oot_scaled[features])
        X_oot_scaled[target_col] = y_oot.reset_index(drop=True)
       
    except Exception as e:
        print(f"There was an issue with the OOT dataset: {str(e)}. Please check if the dataset is properly formatted and contains the required features and target columns.")
 
    # PRINT ALL METRICS
    (auc_train_unique, auc_test_unique, auc_val_unique, auc_oot_unique,
     ks_train_unique, ks_test_unique, ks_val_unique, ks_oot_unique) = print_overall_metrics(model=model, X_train=X_train_scaled, X_test=X_test_scaled, X_val=X_val_scaled, X_oot=X_oot_scaled, COL_TARGET=target_col)
   
 
 
    # METRICS OVER TIME
    (df_metric_ks_train, df_metric_auc_train, df_metric_ks_test, df_metric_auc_test,
     df_metric_ks_val, df_metric_auc_val, df_metric_ks_oot, df_metric_auc_oot) = compute_metrics_over_time(X_train_scaled, X_test_scaled, X_val_scaled, X_oot_scaled, target_col)
   
   
    # ALL CLASSIFICATION METRICS
    report_df_test = run_classification_report(y_test=y_test, y_pred=X_test_scaled['PRED'], output_dict=True)
    report_df_val = run_classification_report(y_test=y_val, y_pred=X_val_scaled['PRED'], output_dict=True)
    report_df_oot = run_classification_report(y_test=y_oot, y_pred=X_oot_scaled['PRED'], output_dict=True)
   
    # PLOTS
    # declarar aqui o gráfico de interesse para que seja incluído no MLFlow
    (fig_plot_1_test, fig_plot_1_val, fig_plot_1_oot, fig_plot_2_test, fig_plot_2_val, fig_plot_2_oot, fig_plot_3_test, fig_plot_4_test, fig_plot_3_val, fig_plot_4_val,
            fig_plot_3_oot, fig_plot_4_oot, fig_plot_5_test, fig_plot_5_val, fig_plot_5_oot, fig_plot_6_test, fig_plot_6_val, fig_plot_6_oot, fig_plot_7_test, fig_plot_7_val,
            fig_plot_7_oot, fig_plot_8_test, fig_plot_8_val, fig_plot_8_oot, fig_plot_9_test, fig_plot_9_val, fig_plot_9_oot,
            fig_plot_10_test, fig_plot_10_val, fig_plot_10_oot, fig_plot_11_test, fig_plot_11_val, fig_plot_11_oot,
            fig_plot_12_test, fig_plot_12_val, fig_plot_12_oot) = plot_graphs(model_type=model_name,
                                                                              model=model,
                                                                              X_train=X_train_scaled,
                                                                              X_test=X_test_scaled,
                                                                              X_val=X_val_scaled,
                                                                              X_oot=X_oot_scaled,
                                                                              features=features,
                                                                              n_quantile=n_quantile,
                                                                              target_col=target_col,                                                                          
                                                                              y_test=y_test,
                                                                              y_val=y_val,
                                                                              y_oot=y_oot,
                                                                              best_model=best_model)  
 
   #MLFlow
    if save_into_mlflow:
        print()
        print('Save the artifacts into MLFlow...please wait...')
        print()
        df_metric_auc_train.to_csv('/tmp/df_metric_auc_train.csv', index=False, sep=';')
        df_metric_ks_train.to_csv('/tmp/df_metric_ks_train.csv', index=False, sep=';')
       
        # CLASSIFICATION METRICS
        report_df_test.to_csv('/tmp/classification_report_[test].csv', index=False, sep=';')
        report_df_val.to_csv('/tmp/classification_report_[val].csv', index=False, sep=';')
        report_df_oot.to_csv('/tmp/classification_report_[oot].csv', index=False, sep=';')
       
        # MLFLOW (continuação da run_modeling)
        # Adicionar gráficos ao MLFlow
        with mlflow.start_run(run_id=run_id) as run:
           
           
            # Adicione seus gráficos aqui, por exemplo:
           
            # CLASSIFICATION METRICS
            mlflow.log_artifact('/tmp/classification_report_[test].csv', "reports/")
            mlflow.log_dict(report_df_test.to_dict(orient="records"), "reports/classification_report_[test].json")
            mlflow.log_artifact('/tmp/classification_report_[val].csv', "reports/")
            mlflow.log_dict(report_df_val.to_dict(orient="records"), "reports/classification_report_[val].json")            
            mlflow.log_artifact('/tmp/classification_report_[oot].csv', "reports/")
            mlflow.log_dict(report_df_oot.to_dict(orient="records"), "reports/classification_report_[oot].json")
           
            #AUC
            mlflow.log_artifact('/tmp/df_metric_auc_train.csv', "reports/")
            mlflow.log_dict(df_metric_auc_train.to_dict(orient="records"), "reports/auc_train_overtime.json")
            mlflow.log_metric("AUC_TRAIN", auc_train_unique)
            mlflow.log_metric("AUC_TEST", auc_test_unique)
            mlflow.log_metric("AUC_VAL", auc_val_unique)
            mlflow.log_metric("AUC_OOT", auc_oot_unique)
           
            #KS
            mlflow.log_artifact('/tmp/df_metric_ks_train.csv', "reports/")
            mlflow.log_dict(df_metric_ks_train.to_dict(orient="records"), "reports/ks_train_overtime.json")            
            mlflow.log_metric("KS_TRAIN", ks_train_unique)
            mlflow.log_metric("KS_TEST", ks_test_unique)
            mlflow.log_metric("KS_VAL", ks_val_unique)
            mlflow.log_metric("KS_OOT", ks_oot_unique)
                         
           
            # Gráficos
            mlflow.log_figure(fig_plot_1_test, "plots/test/1_confusion_matrix.png")
            mlflow.log_figure(fig_plot_1_val, "plots/val/1_confusion_matrix.png")
            mlflow.log_figure(fig_plot_1_oot, "plots/oot/1_confusion_matrix.png")          
           
            mlflow.log_figure(fig_plot_2_test, "plots/test/2_correlation_matrix.png")
            mlflow.log_figure(fig_plot_2_val, "plots/val/2_correlation_matrix.png")
            mlflow.log_figure(fig_plot_2_oot, "plots/oot/2_correlation_matrix.png")
           
            mlflow.log_figure(fig_plot_3_test, "plots/test/3_ranking.png")
            mlflow.log_figure(fig_plot_3_val, "plots/val/3_ranking.png")
            mlflow.log_figure(fig_plot_3_oot, "plots/oot/3_ranking.png")
           
            mlflow.log_figure(fig_plot_4_test, "plots/test/4_volume_by_ranking.png")
            mlflow.log_figure(fig_plot_4_val, "plots/val/4_volume_by_ranking.png")
            mlflow.log_figure(fig_plot_4_oot, "plots/oot/4_volume_by_ranking.png")
           
            mlflow.log_figure(fig_plot_5_test, "plots/test/5_feature_importance.png")
            mlflow.log_figure(fig_plot_5_val, "plots/val/5_feature_importance.png")
            mlflow.log_figure(fig_plot_5_oot, "plots/oot/5_feature_importance.png")
           
            mlflow.log_figure(fig_plot_6_test, "plots/test/6_SHAP.png")
            mlflow.log_figure(fig_plot_6_val, "plots/val/6_SHAP.png")
            mlflow.log_figure(fig_plot_6_oot, "plots/oot/6_SHAP.png")
           
            mlflow.log_figure(fig_plot_7_test, "plots/test/7_score_distribution_no_calibration.png")
            mlflow.log_figure(fig_plot_7_val, "plots/val/7_score_distribution_no_calibration.png")
            mlflow.log_figure(fig_plot_7_oot, "plots/oot/7_score_distribution_no_calibration.png")            
           
            mlflow.log_figure(fig_plot_8_test, "plots/test/8_isotonic_regression_distribution.png")
            mlflow.log_figure(fig_plot_8_val, "plots/val/8_isotonic_regression_distribution.png")
            mlflow.log_figure(fig_plot_8_oot, "plots/oot/8_isotonic_regression_distribution.png")
           
            mlflow.log_figure(fig_plot_9_test, "plots/test/9_sigmoid_calibration_distribution.png")
            mlflow.log_figure(fig_plot_9_val, "plots/val/9_sigmoid_calibration_distribution.png")
            mlflow.log_figure(fig_plot_9_oot, "plots/oot/9_sigmoid_calibration_distribution.png")
           
            mlflow.log_figure(fig_plot_10_test, "plots/test/10_quantile_transform_distribution.png")
            mlflow.log_figure(fig_plot_10_val, "plots/val/10_quantile_transform_distribution.png")
            mlflow.log_figure(fig_plot_10_oot, "plots/oot/10_quantile_transform_distribution.png")
           
            mlflow.log_figure(fig_plot_11_test, "plots/test/11_default_overtime.png")
            mlflow.log_figure(fig_plot_11_val, "plots/val/11_default_overtime.png")
            mlflow.log_figure(fig_plot_11_oot, "plots/oot/11_default_overtime.png")
           
            mlflow.log_figure(fig_plot_12_test, "plots/test/12_auc_ks_overtime.png")
            mlflow.log_figure(fig_plot_12_val, "plots/val/12_auc_ks_overtime.png")
            mlflow.log_figure(fig_plot_12_oot, "plots/oot/12_auc_ks_overtime.png")
 
            # Se você tiver gráficos salvos em arquivos, você pode logá-los também
            # mlflow.log_artifact("path/to/your/plot.png")
 
    print('DONE!')
    return (X_train, y_train, X_test, y_test, X_val, y_val, X_oot, y_oot, X_train_scaled, X_test_scaled, X_val_scaled, X_oot_scaled, model,
            auc_train_unique, auc_test_unique, auc_val_unique, auc_oot_unique, ks_train_unique, ks_test_unique, ks_val_unique, ks_oot_unique,
            df_metric_ks_train, df_metric_auc_train, df_metric_ks_test, df_metric_auc_test, df_metric_ks_val, df_metric_auc_val, df_metric_ks_oot,
            df_metric_auc_oot, features, target_col, best_model)
 