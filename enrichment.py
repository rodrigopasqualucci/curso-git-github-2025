#Libs
from itertools import chain
import sys
from functools import reduce
from operator import or_
from datetime import datetime
from typing import Optional
from collections import Counter
from dateutil.relativedelta import relativedelta
 
#MLFLOW
import mlflow
from typing import Optional
from mlflow.data.spark_dataset import SparkDataset
REMOTE_SERVER_URI = "https://mlflow.conecta360dados.com.br"
mlflow.set_tracking_uri(REMOTE_SERVER_URI)
 
#PYSPARK
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.sql.types import *
app_spark = 'ia_customizacao'
spark = (
    SparkSession.builder.appName(app_spark)
    .config(
        "spark.hadoop.hive.metastore.client.factory.class",
        "com.amazonaws.glue.catalog.metastore.AWSGlueDataCatalogHiveClientFactory"
    )
    .enableHiveSupport()
    .getOrCreate()
)
spark.conf.set("spark.sql.adaptive.enabled", "true")
 
#SERVERLESS
import argparse
parser = argparse.ArgumentParser(description='Parametrização do processo .')
parser.add_argument("--customer_data_path", type=str)
parser.add_argument("--customer_sep_or_delimiter", type=str)
parser.add_argument("--nuclea_lake_path", type=str)
parser.add_argument("--enriched_dataset_stage_1_path", type=str, default='')
parser.add_argument("--cnpj_column", type=str)
parser.add_argument("--cnpj_type", type=str)
parser.add_argument("--date_column", type=str)
parser.add_argument("--date_format", type=str)
parser.add_argument("--target_column", type=str)
parser.add_argument("--target_0", type=str, default="")
parser.add_argument("--target_1", type=str, default="")
parser.add_argument("--random_sample", type=int, default=0)
parser.add_argument("--fraction", type=float)
parser.add_argument("--sample_by_target", type=int, default=0)
parser.add_argument("--fraction_stratify", type=float)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--n_recent_months_oot", type=int)
parser.add_argument("--select_enrich_bases", type=str, nargs="*")
parser.add_argument("--activate_feature_selection", type=int, default=0)
parser.add_argument("--final_feature_columns", type=str, nargs="*")
parser.add_argument("--save_into_s3", type=int, default=0)
parser.add_argument("--save_into_mlflow", type=int, default=0)
parser.add_argument("--experiment_name", type=str, default='')
parser.add_argument("--run_name", type=str, default='')
 
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
 
 
def read_dataset(file_path, sep_or_delimiter=None):
   
    if file_path.endswith('.xlsx') or file_path.endswith('.xls'):
        raise ValueError("Unsupported file format: Excel files are not supported by Spark")
 
    # Try reading as Parquet first
    try:
        print('Attempting to read customer dataframe in PARQUET format')
        df = spark.read.parquet(file_path)
        return df
    except Exception as e:        
        print(f'Failed to read as PARQUET')
       
        # Try reading as CSV or TXT
        try:
            print('Attempting to read customer dataframe in CSV or TXT format')
            if sep_or_delimiter is not None:
                df = spark.read.option("delimiter", sep_or_delimiter).option("header", True).csv(file_path)
            else:
                df = spark.read.csv(file_path, header=True)
            return df
       
        except Exception as e:
            print(f'Failed to read as CSV or TXT')
 
 
def remove_duplicates(df, col_id = 'cd_cnpj_basi', date_col='dt_situ_cada'):
 
    # Define a window specification
    #window_spec = Window.partitionBy(col_id).orderBy(F.desc(date_col))
 
    # Keep the last record within each group
    #df = df.withColumn("row_num", F.row_number().over(window_spec)).filter(F.col("row_num") == 1).drop("row_num")
    #df = df.select("*", F.row_number().over(window_spec).alias("row_num")).filter(F.col("row_num") == 1).drop("row_num")
    df = df.dropDuplicates([col_id, date_col])
   
    return df
 
def convert_date(df: DataFrame, date_column: str, date_format: str) -> DataFrame:
    """
    Convert a date column to a standard date format (YYYY-MM-DD) based on the given date format.
 
    :param df: Input DataFrame
    :param date_column: Name of the date column to be converted
    :param date_format: Format of the input date (e.g., 'yyyyMM', 'yyyy-MM-dd', 'dd/MM/yyyy', 'MM/dd/yyyy')
    :return: DataFrame with the date column converted to the standard format
    """      
    spark.conf.set("spark.sql.legacy.timeParserPolicy", "LEGACY")
   
    # Cast 'date_column' to StringType
    df=df.withColumn(date_column, F.col(date_column).cast(StringType()))    
       
    # Define the new column transformation based on the date format
    if date_format == "yyyyMM":    
       
       
 
        # Convert YYYYMM to YYYY-MM-DD format
        formatted_date_column = F.to_date(
            F.concat(
                F.col(date_column).substr(1, 4),
                F.lit("-"),
                F.col(date_column).substr(5, 2),
                F.lit("-01")
            ),
            "yyyy-MM-dd"
        )
       
    elif date_format == "yyyy-MM":
       
       
        # Convert YYYYMM to YYYY-MM-DD format
        formatted_date_column = F.to_date(
                F.concat(
                    F.col(date_column).substr(1, 4),
                    F.lit("-"),
                    F.col(date_column).substr(6, 2),
                    F.lit("-01")
                ),
                "yyyy-MM-dd"
            )                
       
    elif date_format == "yyyyMMdd":
        # Convert YYYYMMDD to YYYY-MM-DD format
        formatted_date_column = F.date_format(F.to_date(F.col(date_column), "yyyyMMdd"), "yyyy-MM-dd")
       
    elif date_format == "dd/MM/yyyy":
        # Convert DD/MM/YYYY to YYYY-MM-DD format
        formatted_date_column = F.date_format(F.to_date(F.col(date_column), "dd/MM/yyyy"), "yyyy-MM-dd")
       
    elif date_format == "MM/dd/yyyy":
        # Convert MM/DD/YYYY to YYYY-MM-DD format
        formatted_date_column = F.date_format(F.to_date(F.col(date_column), "MM/dd/yyyy"), "yyyy-MM-dd")
       
    elif date_format == "yyyy-MM-dd":
        # Already in YYYY-MM-DD format, no conversion needed
        formatted_date_column = F.date_format(F.to_date(F.col(date_column), "yyyy-MM-dd"), "yyyy-MM-dd")
       
    else:
        # If format is unknown, keep the column as is
        formatted_date_column = F.col(date_column)
 
    # Use select to apply the transformation and keep all existing columns
    df = df.select(
        "*",  # Keep all existing columns
        formatted_date_column.alias('formatted_date').cast(DateType())  # Replace the original date column with the formatted date
    ).drop(date_column).withColumnRenamed('formatted_date', date_column)
 
    return df
 
 
def read_dataset_with_index_columns(file_path, random_sample, session_name = 'Materialidade', sep_or_delimiter=';', fraction=0.0005, seed = 42):
    """
    Function to read a CSV dataset and convert column names to uppercase.
 
    Args:
    - spark: SparkSession object.
    - file_path: Path to the CSV dataset.
 
    Returns:
    - PySpark DataFrame with column names in uppercase.
    """  
   
    # Read the CSV dataset with header
    df = read_dataset(file_path, sep_or_delimiter)      
   
    if random_sample:        
        df = df.sample(fraction=fraction, seed=seed)
   
    df = df.withColumn("INDEX", F.monotonically_increasing_id())
   
    # Number of columns from original Customer Dataset
    n_cols_original_customer_dataset = len(df.columns)
   
    # Number of columns from original Customer Dataset
    n_rows_original_customer_dataset = df.count()
   
    return df, n_cols_original_customer_dataset, n_rows_original_customer_dataset
 
 
def add_cnpj_type(df, id_customer_col='id', type_of_entity='completo'):    
    type_of_entity = type_of_entity.lower()    
    df_t = (df.withColumn("num_digitos_cnpj", F.length(F.upper(F.col(id_customer_col))))
              .withColumn('tipo_cnpj', F.lit(type_of_entity))
              #.withColumn("tipo_cnpj", F.when(F.col("num_digitos_cnpj") >= 11, 'completo').otherwise('radical'))
              )
 
    return df_t
 
def remove_caracteres_zeros_esquerda(df, id_customer_col='id', cnpj_type_str='completo'):
 
    type_entity = cnpj_type_str.lower()
    print(type_entity)
 
    if type_entity == 'radical':
        df_rz = (df
                .withColumn('radical_cnpj', F.regexp_replace(F.col(id_customer_col), r"\D", ""))
                .withColumn('radical_cnpj', F.regexp_replace(F.col(id_customer_col), r'^[0]*', ''))
                )              
 
    elif type_entity == 'completo':
        df_rz = (df
                .withColumn('completo_cnpj', F.regexp_replace(F.col(id_customer_col), r"\D", ""))            
                .withColumn('completo_cnpj', F.regexp_replace(F.col(id_customer_col), r'^[0]*', ''))
                         )        
   
    return df_rz
 
def convert_year_month_col(df, date_column_col = 'date', date_format='yyyyMM'):
    df = convert_date(df, date_column=date_column_col, date_format=date_format)
 
    return df.withColumn('data_ref1', F.trunc(F.to_date(date_column_col, date_format), 'month'))    
 
def filter_binary_target(df: DataFrame, target_col: str = 'tg', positive_label: str = 'Bom pagador', negative_label: str = 'Mau pagador') -> DataFrame:
    """
    Ensure that a DataFrame contains a binary target column with values 0 and 1.
   
    Args:
    - df (pyspark.sql.DataFrame): Input PySpark DataFrame.
    - target_col (str): The name of the target column. Default is 'tg'.
    - positive_label (str): The string label to be converted to 1. Default is 'Mau pagador'.
    - negative_label (str): The string label to be converted to 0. Default is 'Bom pagador'.
   
    Returns:
    - pyspark.sql.DataFrame: A DataFrame with a binary target column.
    """
   
    # Verifica se a coluna é do tipo string
    if df.schema[target_col].dataType == StringType():
        # Se a coluna contém strings que representam os rótulos de classificação
        if positive_label != "None" and negative_label != "None":
 
            df = df.withColumn(
                target_col,
                F.when(F.col(target_col) == positive_label, 0)
                 .when(F.col(target_col) == negative_label, 1)
            )
        else:
            # Caso contrário, se a coluna contiver strings '1' ou '0', converte para int
            df = df.withColumn(
                target_col,
                F.when(F.col(target_col) == "1", 1)
                 .when(F.col(target_col) == "0", 0)
            )
 
    # Verifica se a coluna é de tipo numérico inteiro
    elif df.schema[target_col].dataType in [ByteType(), ShortType(), IntegerType(), LongType()]:
        # Filtra apenas os valores binários (0 ou 1)
        df = df.filter((F.col(target_col) == 0) | (F.col(target_col) == 1))
   
    # Verifica se a coluna é de tipo numérico de ponto flutuante ou decimal
    elif df.schema[target_col].dataType in [FloatType(), DoubleType(), DecimalType()]:
        # Converte para inteiro e filtra os valores binários (0 ou 1)
        df = df.withColumn(target_col, F.col(target_col).cast(IntegerType()))
        # Nova feature
        #df = df.withColumn('iac_target_binary', F.col(target_col).cast(IntegerType()))
 
        df = df.filter((F.col(target_col) == 0) | (F.col(target_col) == 1))
 
    return df
 
# lista de features
columns_mat_90 = [
    #5M
    'count_m05',
    'perc_good_m05',
    'atraso_medio_m05',
    'atraso_minimo_m05',
    'vl_pg_em_dia_m05',
    'vl_total_m05',
    #4M
    'count_m04',
    'perc_good_m04',
    'atraso_medio_m04',
    'atraso_minimo_m04',
    'vl_pg_em_dia_m04',
    'vl_total_m04',
    #3M
    'count_m03',
    'perc_good_m03',
    'atraso_medio_m03',
    'atraso_minimo_m03',
    'vl_pg_em_dia_m03',
    'vl_total_m03',
    #2M
    'count_m02',
    'perc_good_m02',
    'atraso_medio_m02',
    'atraso_minimo_m02',
    'vl_pg_em_dia_m02',
    'vl_total_m02',
    #1M
    'count_m01',
    'perc_good_m01',
    'atraso_medio_m01',
    'atraso_minimo_m01',
    'vl_pg_em_dia_m01',
    'vl_total_m01',
   
    # 3M,6M,12M
    'valor_recebido_boletos_3M',
    'valor_recebido_boletos_6M',
    'valor_recebido_boletos_12M',
    'total_cartao_recebido_3M',
    'total_cartao_recebido_6M',
    'total_cartao_recebido_12M',
    'total_ted_recebido_3M',
    'total_ted_recebido_6M',
    'total_ted_recebido_12M',
    'total_recebido_3M',
    'total_recebido_6M',
    'total_recebido_12M',
    # cadastral
    #'secao_cnae',
    #'divisao_cnae',
    #'grupo_cnae',
    #'classe_cnae',
    #'subclasse_cnae',
    #'nm_porte_empr',
    #'flag_socio_pj',
    #'flag_socio_pf',
    #'flag_socio_estr',
    #'flag_radical_tem_filial',
    #'flag_filial'
]
 
# lista de features
columns_hefindahl = [
    'ind_Hefindahl_H_paga_1m',
    'ind_Hefindahl_H_bene_1m',
    'ind_Hefindahl_H_paga_2m',
    'ind_Hefindahl_H_bene_2m',
    'ind_Hefindahl_H_paga_3m',
    'ind_Hefindahl_H_bene_3m',
    'ind_Hefindahl_H_paga_6m',
    'ind_Hefindahl_H_bene_6m',
    'ind_Hefindahl_H_paga_12m',
    'ind_Hefindahl_H_bene_12m'
]
 
# lista de features
columns_indice_liquidez = [
    'cedente_indice_liquidez_1m',
    'cedente_indice_liquidez_2m',
    'cedente_indice_liquidez_3m',
    'cedente_indice_liquidez_6m',
    'sacado_indice_liquidez_1m',
    'sacado_indice_liquidez_2m',
    'sacado_indice_liquidez_3m',
    'sacado_indice_liquidez_6m'
]
 
# lista de features
columns_ranking_pcr = [
    'vl_pago_boletos_ranking',
    'vl_pago_boletos_sum_2m_ranking',
    'vl_pago_boletos_sum_3m_ranking',
    'vl_pago_boletos_sum_6m_ranking',
    'vl_pago_boletos_sum_9m_ranking',
    'vl_pago_boletos_sum_12m_ranking',
    'vl_recebido_boletos_ranking',
    'vl_recebido_boletos_sum_2m_ranking',
    'vl_recebido_boletos_sum_3m_ranking',
    'vl_recebido_boletos_sum_6m_ranking',
    'vl_recebido_boletos_sum_9m_ranking',
    'vl_recebido_boletos_sum_12m_ranking'
]
 
# lista de features
columns_trend = [
    'vl_recebido_cartao_credito_avg_1m_2m_trend',
    'vl_recebido_cartao_credito_avg_1m_3m_trend',
    'vl_recebido_cartao_credito_avg_1m_6m_trend',
    'vl_recebido_cartao_credito_avg_1m_12m_trend',
    'vl_recebido_cartao_credito_avg_2m_3m_trend',
    'vl_recebido_cartao_credito_avg_2m_6m_trend',
    'vl_recebido_cartao_credito_avg_2m_12m_trend',
    'vl_recebido_cartao_credito_avg_3m_6m_trend',
    'vl_recebido_cartao_credito_avg_3m_12m_trend',
    'vl_recebido_cartao_credito_avg_6m_12m_trend',
    'vl_recebido_cartao_debito_avg_1m_2m_trend',
    'vl_recebido_cartao_debito_avg_1m_3m_trend',
    'vl_recebido_cartao_debito_avg_1m_6m_trend',
    'vl_recebido_cartao_debito_avg_1m_12m_trend',
    'vl_recebido_cartao_debito_avg_2m_3m_trend',
    'vl_recebido_cartao_debito_avg_2m_6m_trend',
    'vl_recebido_cartao_debito_avg_2m_12m_trend',
    'vl_recebido_cartao_debito_avg_3m_6m_trend',
    'vl_recebido_cartao_debito_avg_3m_12m_trend',
    'vl_recebido_cartao_debito_avg_6m_12m_trend',
    'vl_recebido_cartao_antecipacao_avg_1m_2m_trend',
    'vl_recebido_cartao_antecipacao_avg_1m_3m_trend',
    'vl_recebido_cartao_antecipacao_avg_1m_6m_trend',
    'vl_recebido_cartao_antecipacao_avg_1m_12m_trend',
    'vl_recebido_cartao_antecipacao_avg_2m_3m_trend',
    'vl_recebido_cartao_antecipacao_avg_2m_6m_trend',
    'vl_recebido_cartao_antecipacao_avg_2m_12m_trend',
    'vl_recebido_cartao_antecipacao_avg_3m_6m_trend',
    'vl_recebido_cartao_antecipacao_avg_3m_12m_trend',
    'vl_recebido_cartao_antecipacao_avg_6m_12m_trend',
    'vl_recebido_cartao_total_avg_1m_2m_trend',
    'vl_recebido_cartao_total_avg_1m_3m_trend',
    'vl_recebido_cartao_total_avg_1m_6m_trend',
    'vl_recebido_cartao_total_avg_1m_12m_trend',
    'vl_recebido_cartao_total_avg_2m_3m_trend',
    'vl_recebido_cartao_total_avg_2m_6m_trend',
    'vl_recebido_cartao_total_avg_2m_12m_trend',
    'vl_recebido_cartao_total_avg_3m_6m_trend',
    'vl_recebido_cartao_total_avg_3m_12m_trend',
    'vl_recebido_cartao_total_avg_6m_12m_trend',
]
 
def get_feature_table_with_shift(entity_column,
                                 path,
                                 features,
                                 dates,
                                 feature_reference_date_column,
                                 feature_reference_date_format,
):
    df_temp = get_feature(path=path,
                          unique_dates=dates,
                          feature_reference_date_column=feature_reference_date_column,
                          feature_reference_date_format=feature_reference_date_format,
    )
 
    #preprocessing
    df_temp = df_temp.select(entity_column,
                             *features,
                             F.add_months(F.col("mes"),1).alias('data_ref1')
                             )  
   
    df_temp = df_temp.withColumn(entity_column, F.col(entity_column).cast(StringType()))
    #df_temp = df_temp.withColumn(entity_column, F.col(entity_column).cast(LongType()))
    #df_temp = df_temp.withColumn(entity_column, F.col(entity_column).cast(DecimalType(38, 0)))
    #df_temp = df_temp.withColumn(entity_column, F.col(entity_column).cast(IntegerType()))
    #print(df_temp.printSchema())
    return df_temp
 
def get_feature(path,
                unique_dates,
                feature_reference_date_column,
                feature_reference_date_format,):
       
        df_feature = spark.read.parquet(path)
        feature_filter = _get_filter_by_column(df_feature.columns, unique_dates)
       
        df_feature = df_feature.withColumn(
            feature_reference_date_column,
            F.to_date(feature_reference_date_column, feature_reference_date_format),
        )
 
       
        df_feature = df_feature.filter(feature_filter)
        return df_feature
 
def _get_filter_by_column(
    product_columns, unique_dates
):
 
    if set(product_columns).intersection(["partition_0", "partition_1"]):
        conditions = [(F.col("partition_1") == date.month) & (F.col("partition_0") == date.year) for date in unique_dates]
        months_filter = reduce(or_, conditions)
        return months_filter
 
    if set(product_columns).intersection(["mes"]):
        return F.col("mes").isin(unique_dates)
 
    if set(product_columns).intersection(["year", "month"]):
        conditions = [(F.col("month") == date.month) & (F.col("year") == date.year) for date in unique_dates]
        months_filter = reduce(or_, conditions)        
        return months_filter
   
@F.udf
def mode(x):
    """
    Gets the most frequent element inside a list. Used in spark with `collect_list` aggregate function.
    Args:
        x: A list.
    Returns:
        The most frequent item inside list
    """
    try:
        return Counter(x).most_common(1)[0][0]
    except:
        return None
   
 
def materialidade_90_dias_radical(path = 's3://mntz-datascience/xxx/',
                                  features = ['valor_recebido_boletos_3M'],
                                  safras=['202304'],
                                  ):
 
    df_temp = get_feature(path=path,
                          unique_dates=safras,
                          feature_reference_date_column='date_ref',
                          feature_reference_date_format='yyyyMM',
                          )
 
    #preprocessing
    df_temp = (df_temp.select(*features, F.col('cnpj_radical').alias('cnpj_radical'), F.col('date_ref').alias('data_ref1'))              
                .withColumn('count_good_m01', F.col('count_m01') * F.col('perc_good_m01'))
                .withColumn('count_good_m02', F.col('count_m02') * F.col('perc_good_m02'))
                .withColumn('count_good_m03', F.col('count_m03') * F.col('perc_good_m03'))
                .withColumn('count_good_m04', F.col('count_m04') * F.col('perc_good_m04'))
                .withColumn('count_good_m05', F.col('count_m05') * F.col('perc_good_m05'))  
                .groupBy("cnpj_radical", "data_ref1")
                .agg(
                     #F.mean("delta_medio_m05").alias("delta_medio_m05"),
                     F.mean("atraso_medio_m05").alias("atraso_medio_m05"),
                     F.sum("count_good_m05").alias("count_good_m05"),
                     F.sum("count_m05").alias("count_m05"),
                     F.sum('vl_total_m05').alias('vl_total_m05'),
                     F.sum(f'vl_pg_em_dia_m05').alias(f'vl_pg_em_dia_m05'),
                     (F.sum(f'vl_pg_em_dia_m05')/F.sum('vl_total_m05')).alias('perc_vl_pg_m05'),
                     (F.sum(f'count_good_m05')/F.sum('count_m05')).alias('perc_good_m05'),
                     F.min('atraso_minimo_m05').alias('atraso_minimo_m05'),
 
                     #F.mean("delta_medio_m04").alias("delta_medio_m04"),
                     F.mean("atraso_medio_m04").alias("atraso_medio_m04"),
                     F.sum("count_good_m04").alias("count_good_m04"),
                     F.sum("count_m04").alias("count_m04"),
                     F.sum('vl_total_m04').alias('vl_total_m04'),
                     F.sum(f'vl_pg_em_dia_m04').alias(f'vl_pg_em_dia_m04'),
                     (F.sum(f'vl_pg_em_dia_m04')/F.sum('vl_total_m04')).alias('perc_vl_pg_m04'),
                     (F.sum(f'count_good_m04')/F.sum('count_m04')).alias('perc_good_m04'),
                     F.min('atraso_minimo_m04').alias('atraso_minimo_m04'),
               
                     #F.mean("delta_medio_m03").alias("delta_medio_m03"),
                     F.mean("atraso_medio_m03").alias("atraso_medio_m03"),
                     F.sum("count_good_m03").alias("count_good_m03"),
                     F.sum("count_m03").alias("count_m03"),
                     F.sum('vl_total_m03').alias('vl_total_m03'),
                     F.sum(f'vl_pg_em_dia_m03').alias(f'vl_pg_em_dia_m03'),
                     (F.sum(f'vl_pg_em_dia_m03')/F.sum('vl_total_m03')).alias('perc_vl_pg_m03'),
                     (F.sum(f'count_good_m03')/F.sum('count_m03')).alias('perc_good_m03'),
                     F.min('atraso_minimo_m03').alias('atraso_minimo_m03'),
                   
                     #F.mean("delta_medio_m02").alias("delta_medio_m02"),
                     F.mean("atraso_medio_m02").alias("atraso_medio_m02"),
                     F.sum("count_good_m02").alias("count_good_m02"),
                     F.sum("count_m02").alias("count_m02"),
                     F.sum('vl_total_m02').alias('vl_total_m02'),
                     F.sum(f'vl_pg_em_dia_m02').alias(f'vl_pg_em_dia_m02'),
                     (F.sum(f'vl_pg_em_dia_m02')/F.sum('vl_total_m02')).alias('perc_vl_pg_m02'),
                     (F.sum(f'count_good_m02')/F.sum('count_m02')).alias('perc_good_m02'),
                     F.min('atraso_minimo_m02').alias('atraso_minimo_m02'),
                   
                     #F.mean("delta_medio_m01").alias("delta_medio_m01"),
                     F.mean("atraso_medio_m01").alias("atraso_medio_m01"),
                     F.sum("count_good_m01").alias("count_good_m01"),
                     F.sum("count_m01").alias("count_m01"),
                     F.sum('vl_total_m01').alias('vl_total_m01'),
                     F.sum(f'vl_pg_em_dia_m01').alias(f'vl_pg_em_dia_m01'),
                     (F.sum(f'vl_pg_em_dia_m01')/F.sum('vl_total_m01')).alias('perc_vl_pg_m01'),
                     (F.sum(f'count_good_m01')/F.sum('count_m01')).alias('perc_good_m01'),
                     F.min('atraso_minimo_m01').alias('atraso_minimo_m01'),                  
                     
                     F.sum(f'valor_recebido_boletos_3M').alias(f'valor_recebido_boletos_3M'),
                     F.sum(f'valor_recebido_boletos_6M').alias(f'valor_recebido_boletos_6M'),
                     F.sum(f'valor_recebido_boletos_12M').alias(f'valor_recebido_boletos_12M'),
                     F.sum(f'total_cartao_recebido_3M').alias(f'total_cartao_recebido_3M'),
                     F.sum(f'total_cartao_recebido_6M').alias(f'total_cartao_recebido_6M'),
                     F.sum(f'total_cartao_recebido_12M').alias(f'total_cartao_recebido_12M'),
                     F.sum(f'total_ted_recebido_3M').alias(f'total_ted_recebido_3M'),
                     F.sum(f'total_ted_recebido_6M').alias(f'total_ted_recebido_6M'),
                     F.sum(f'total_ted_recebido_12M').alias(f'total_ted_recebido_12M'),
                     F.sum(f'total_recebido_3M').alias(f'total_recebido_3M'),
                     F.sum(f'total_recebido_6M').alias(f'total_recebido_6M'),
                     F.sum(f'total_recebido_12M').alias(f'total_recebido_12M'),
   
                     # MODE                                      
                     #mode(F.collect_list(F.col("nm_porte_empr"))).alias("nm_porte_empr"),
                     #mode(F.collect_list(F.col("flag_radical_tem_filial"))).alias("flag_radical_tem_filial"),
                     #mode(F.collect_list(F.col("flag_socio_pj"))).alias("flag_socio_pj"),
                     #mode(F.collect_list(F.col("flag_socio_pf"))).alias("flag_socio_pf"),
                     #mode(F.collect_list(F.col("flag_socio_estr"))).alias("flag_socio_estr"),
                )
              )
 
    df_temp = df_temp.withColumn('cnpj_radical', F.col('cnpj_radical').cast(StringType())).withColumn('data_ref1', F.col('data_ref1').cast(StringType()))  
    #print(df_temp.printSchema())
    return df_temp  
 
def materialidade_90_dias_completo(path = 's3://mntz-datascience/xxx/',
                                   features = ['valor_recebido_boletos_3M'],
                                   safras=['202304'],
                                   ):
 
    df_temp = get_feature(path=path,
                          unique_dates=safras,
                          feature_reference_date_column='date_ref',
                          feature_reference_date_format='yyyyMM',
                          )
 
    df_temp = df_temp.select(*features, F.col('paga_cpf_cnpj').alias('cnpj'), F.col('date_ref').alias('data_ref1'))    
    df_temp = df_temp.withColumn('cnpj',F.col('cnpj').cast(StringType())).withColumn('data_ref1', F.col('data_ref1').cast(StringType()))            
    #print(df_temp.printSchema())              
    return df_temp
 
def split_abt_into_samples(abt_df,
                           safras=['202101', '202102', '202103'],
                           n_recent_months=3,
                           col_data='ANOMES',
                           col_id = 'id',):  
       
    CORTE_OOT = sorted(safras)[-n_recent_months:]    
    df_ = abt_df.where(F.col(col_data) < CORTE_OOT[0])
    df_oot = abt_df.where(F.col(col_data) >= CORTE_OOT[0])
 
    cnpj_lista = df_.select(col_id).distinct()  
    cnpj_treino, cnpj_ = cnpj_lista.randomSplit([0.7, 0.3], seed=42)
    cnpj_test, cnpj_val  = cnpj_.randomSplit([0.5, 0.5], seed=42)
   
    df_treino = df_.join(cnpj_treino, on=col_id, how='inner')
    df_treino = df_treino.dropDuplicates()
 
    df_test = df_.join(cnpj_test, on=col_id, how='inner')
    df_test = df_test.dropDuplicates()
 
    df_val = df_.join(cnpj_val, on=col_id, how='inner')  
    df_val = df_val.dropDuplicates()
 
    return df_treino, df_test, df_val, df_oot
 
 
#def add_new_features(df):
#    df_f = (df.withColumn('flag_null_features',
#                          F.when((F.col('count_m01') <= 0.0) &
#                                 (F.col('count_m02') <= 0.0) &
#                                 (F.col('count_m03') <= 0.0), 1).otherwise(0))
#              .withColumn("target", F.col("target").cast("double"))
#              .withColumn('porte', F.when(F.col('nm_porte_empr') == 'Microempreendedor Individual', 0)
#                          .when(F.col('nm_porte_empr') == '01 - Micro empresa', 1)
#                          .when(F.col('nm_porte_empr') == '03 - Empresa de pequeno porte', 3)
#                          .when(F.col('nm_porte_empr') == '05 - Demais', 5)
#                          .otherwise(-1))              
#                           )
#    
#    return df_f
 
# Save dataset
def save_dataset(df, path='s3://mntz-datascience/', file_name='base_'):    
    df.repartition(10).write.mode('overwrite').parquet(path+file_name)
 
def create_list_feats(my_dict):    
    list_1 = []
    list_2 = []
    ii = -1
    for k,v in my_dict.items():
         
        list_1 += my_dict[k][:ii]
        list_2 += my_dict[k][ii*-1:]
       
    return list_1, list_2
 
def create_dict_features(list_of_feats = [],
                         drop_cols_dict = {},
                         col_names = ['pago'],
                         sort_list = True
                        ):
   
    # items to be removed    
    list_of_feats = [ele for ele in list_of_feats if ele not in drop_cols_dict]
   
    index = 0
    my_dict = {}
 
    for elmt in col_names:
       
        if sort_list == True:
            my_list = sorted([element for element in sorted(list_of_feats) if elmt in element])
            my_dict[index] = my_list
            index += 1
           
        else:            
            my_dict[index] = [element for element in list_of_feats if elmt in element]
            index += 1            
       
    return my_dict
 
def transforme_functions(col='', prefix='NEW__'):
   
    # Seleciona todas as colunas existentes mais a coluna transformada
    return [#1 polymonial
            F.when(F.col(col) == 0, None).when(F.col(col).isNull(), None).otherwise(F.pow(F.col(col), 2)).alias(f"{prefix}polynomial_dg2_{col}"),
            F.when(F.col(col) == 0, None).when(F.col(col).isNull(), None).otherwise(F.pow(F.col(col), 3)).alias(f"{prefix}polynomial_dg3_{col}"),
            F.when(F.col(col) == 0, None).when(F.col(col).isNull(), None).otherwise(F.pow(F.col(col), 4)).alias(f"{prefix}polynomial_dg4_{col}"),
           
            #2 log
            F.log(F.col(col)).alias(f"{prefix}log_value_{col}"),
           
            #3 ln
            F.log1p(F.col(col)).alias(f"{prefix}ln_value_{col}"),
           
            #4 Cosine
            F.cos(F.col(col)).alias(f"{prefix}cosine_value_{col}"),
           
            #5 Sine
            F.sin(F.col(col)).alias(f"{prefix}sine_value_{col}"),
           
            #6 Multiple of 10
            F.when((F.col(col) % 10 == 0), 1).otherwise(0).alias(f"{prefix}multiple_of_10_{col}"),  
           
            #7 Exponential
            F.exp(F.col(col)).alias(f"{prefix}exp_{col}"),
           
            #8 Inverse
            (1 / F.col(col)).alias(f"{prefix}inverse_{col}"),
           
            #9 Null or Non-null
            F.when(F.col(col).isNull(), 1).otherwise(0).alias(f"{prefix}is_null_{col}"),
           
            #10 Even or Non-even
            F.when((F.col(col) % 2 == 0) & (~F.col(col).isNull()), 1).otherwise(F.when(F.col(col).isNull(), None).otherwise(0)).alias(f"{prefix}is_even_{col}"),
           
            #11 Odd or Non-odd
            F.when((F.col(col) % 2 != 0) & (~F.col(col).isNull()), 1).otherwise(F.when(F.col(col).isNull(), None).otherwise(0)).alias(f"{prefix}is_odd_{col}"),
           
            #12 Square root                    
            F.sqrt(F.col(col).cast('float')).alias(f"{prefix}sqrt_{col}"),
           
            #13 Tangente
            F.tan(F.col(col)).alias(f"{prefix}tan_value_{col}"),
           
            #14 Arco seno
            F.asin(F.col(col)).alias(f"{prefix}arc_sin_value_{col}"),
           
            #15 Arco cosseno
            F.acos(F.col(col)).alias(f"{prefix}arc_cos_value_{col}"),
           
            #16 Arco tangente
            F.atan(F.col(col)).alias(f"{prefix}arc_tan_value_{col}"),
           
            #17 Coseno hiperbólico
            F.cosh(F.col(col)).alias(f"{prefix}hiperbolic_cos_{col}"),
           
            #18 Seno hiperbólico
            F.sinh(F.col(col)).alias(f"{prefix}hiperbolic_sin_{col}")                    
    ]
 
def delta_sum_mean_stddev_trend_functions(col1="", col2='', prefix= 'NEW__'):    
    return [
        # Sum
        (F.col(col2) + F.col(col1)).alias(f"{prefix}sum_{col2}_vs_{col1}"),
       
        # Mean
        ((F.col(col2) + F.col(col1)) / 2).alias(f"{prefix}mean_{col2}_vs_{col1}"),
       
        # Intermediate mean column for delta calculations
        ((F.col(col2) + F.col(col1)) / 2).alias(f'REMOVER__media_{col2}_vs_{col1}'),
       
        # Delta and sqrt of delta
        (F.col(col1) - F.col(f'REMOVER__media_{col2}_vs_{col1}')).alias(f"{prefix}delta_{col1}_vs_mean_{col1}_and_{col2}"),
        (F.col(col2) - F.col(f'REMOVER__media_{col2}_vs_{col1}')).alias(f"{prefix}delta_{col2}_vs_mean_{col1}_and_{col2}"),
        (F.pow(F.col(f"{prefix}delta_{col1}_vs_mean_{col1}_and_{col2}"), 2)).alias(f"{prefix}sqrt_of_delta_{col1}_vs_mean_{col1}_and_{col2}"),
        (F.pow(F.col(f"{prefix}delta_{col2}_vs_mean_{col1}_and_{col2}"), 2)).alias(f"{prefix}sqrt_of_delta_{col2}_vs_mean_{col1}_and_{col2}"),
       
        # Sum and standard deviation
        (F.col(f"{prefix}sqrt_of_delta_{col1}_vs_mean_{col1}_and_{col2}") + F.col(f"{prefix}sqrt_of_delta_{col2}_vs_mean_{col1}_and_{col2}")).alias(f'REMOVER__sum_{col2}_vs_{col1}'),
        (F.col(f'REMOVER__sum_{col2}_vs_{col1}') / 1).alias(f'REMOVER__var_{col2}_vs_{col1}'),
        (F.pow(F.col(f'REMOVER__var_{col2}_vs_{col1}'), 0.5)).alias(f"{prefix}stddev_{col2}_vs_{col1}"),
       
        # Delta calculations
        (F.col(col1) - F.col(col2)).alias(f"{prefix}delta_{col1}_vs_{col2}"),
        (F.col(col2) - F.col(col1)).alias(f"{prefix}delta_{col2}_vs_{col1}"),
       
        # Trend ratios
        F.when(F.col(col1) == 0, None).otherwise(F.col(col2) / F.col(col1)).alias(f"{prefix}trend_ratio_{col2}_vs_{col1}"),
        F.when(F.col(col2) == 0, None).otherwise(F.col(col1) / F.col(col2)).alias(f"{prefix}trend_ratio_{col1}_vs_{col2}"),
       
        # Trend clusters
        F.when(F.col(f"{prefix}trend_ratio_{col2}_vs_{col1}").isNull(), None)
        .when(F.col(f"{prefix}trend_ratio_{col2}_vs_{col1}") == 1, "Manteve")
        .when(F.col(f"{prefix}trend_ratio_{col2}_vs_{col1}") > 1, "Aumentou")
        .when(F.col(f"{prefix}trend_ratio_{col2}_vs_{col1}") < 1, "Diminuiu").alias(f"{prefix}trend_cluster_{col2}_vs_{col1}"),
       
        F.when(F.col(f"{prefix}trend_ratio_{col1}_vs_{col2}").isNull(), None)
        .when(F.col(f"{prefix}trend_ratio_{col1}_vs_{col2}") == 1, "Manteve")
        .when(F.col(f"{prefix}trend_ratio_{col1}_vs_{col2}") > 1, "Aumentou")
        .when(F.col(f"{prefix}trend_ratio_{col1}_vs_{col2}") < 1, "Diminuiu").alias(f"{prefix}trend_cluster_{col1}_vs_{col2}")
    ]
 
def calculate_skewness_kurtosis_max_min_mean_median_sttdev_sum(df, cols = [], prefix='NEW__', col_feat=''):
   
    df_temp = {}
    df_temp[col_feat] = (df.select('INDEX',
                          F.array([F.col(c) for c in cols]).alias(f"array_{col_feat}"),
                          )
                       .select('INDEX',
                               F.explode(F.col(f"array_{col_feat}")).alias('exploded_col')  # Renaming the exploded column
                          )
                       .groupBy('INDEX')
                       .agg(F.skewness(F.col('exploded_col')).alias(f"{prefix}skewness_{col_feat}"),
                            F.kurtosis(F.col('exploded_col')).alias(f"{prefix}kurtosis_{col_feat}"),
                            F.sum(F.col('exploded_col')).alias(f"{prefix}sum_{col_feat}"),
                            F.median(F.col('exploded_col')).alias(f"{prefix}median_{col_feat}"),
                            F.mean(F.col('exploded_col')).alias(f"{prefix}mean_{col_feat}"),
                            F.max(F.col('exploded_col')).alias(f"{prefix}max_{col_feat}"),                          
                            F.min(F.col('exploded_col')).alias(f"{prefix}min_{col_feat}"),
                            F.stddev(F.col('exploded_col')).alias(f"{prefix}stddev_{col_feat}"),
                           )
 
                        .withColumn(f"{prefix}max_{col_feat}", F.when(F.col(f"{prefix}max_{col_feat}").isNull(), F.lit(None))
                                .otherwise(F.col(f"{prefix}max_{col_feat}"))
                                    )
                        )                
    return df_temp
 
 
def create_new_features_part_i(df, feat_list = []):    
    transform_operations = (
        transforme_functions(col=column, prefix= 'NEW__')
        for column in feat_list
    )
    select_operations = list(chain.from_iterable(transform_operations))
   
    return df.select("INDEX",*select_operations)
 
 
def create_new_features_part_ii(df, list1=[], list2=[]):
    # Gerar todas as operações de transformação para cada par de colunas
    transform_operations = (
        delta_sum_mean_stddev_trend_functions(col1=col1, col2=col2, prefix='NEW__')
        for col1, col2 in zip(list1, list2)
    )
   
    # Usar chain.from_iterable para combinar todas as transformações em uma única lista
    select_operations = list(chain.from_iterable(transform_operations))
   
    # Aplicar as operações de seleção ao DataFrame
    return df.select("INDEX",*select_operations).drop(*[col for col in df.columns if 'REMOVER__' in col])
 
 
def create_new_features_part_iii(df, contain_list = [], list_1=[], list_2=[]):  
    df_temp = {}
    for string in contain_list:
        feat_lst = [i for i in sorted(list(set(list_1 + list_2))) if string in i and 'NEW__' not in i]        
        df_f = calculate_skewness_kurtosis_max_min_mean_median_sttdev_sum(df, cols = feat_lst, prefix='NEW__', col_feat=string)  
        df_temp.update(df_f)
 
    for key_name in df_temp.keys():
        df = df.join(df_temp[key_name], on=['INDEX'], how='left')
        df = df.drop(*[i for i in df.columns if 'array_' in i])
       
    return df
 
def add_new_features_materialidade_90(df):
   
    # materialidade_90
    my_dict_mat_90 = create_dict_features(list_of_feats = columns_mat_90,
                                          drop_cols_dict = {'cnpj_radical', 'data_ref', 'date_ref', 'paga_cpf_cnpj',
                                                            'secao_cnae', 'divisao_cnae', 'grupo_cnae', 'classe_cnae',
                                                            'subclasse_cnae', 'nm_porte_empr', 'flag_socio_pj', 'flag_socio_pf',
                                                            'flag_socio_estr', 'flag_radical_tem_filial', 'flag_filial'},
                                          col_names = ['count', 'perc_good', 'atraso_medio', 'atraso_minimo', 'pg_em_dia', 'vl_total'],
                                          sort_list = True)
   
    list_1_mat_90, list_2_mat_90 = create_list_feats(my_dict_mat_90)
   
    # Adding new features
    df_1 = create_new_features_part_i(df.select(* sorted(list(set(list_1_mat_90+list_2_mat_90)))+['INDEX']), feat_list = sorted(list(set(list_1_mat_90+list_2_mat_90))))
    df_2 = create_new_features_part_ii(df.select(* sorted(list(set(list_1_mat_90+list_2_mat_90)))+['INDEX']), list1 = list_1_mat_90, list2 = list_2_mat_90)
    df_3 = create_new_features_part_iii(df.select(*sorted(list(set(list_1_mat_90+list_2_mat_90)))+['INDEX']),
                                      contain_list = ['atraso_medio', 'atraso_minimo', 'pg_em_dia', 'vl_total', 'count', 'perc_good',],
                                      list_1=list_1_mat_90, list_2=list_2_mat_90
                                     )
    df_f = (df_1.select(*[i for i in df_1.columns if 'NEW__' in i]+['INDEX'])
            .join(df_2.select(*[i for i in df_2.columns if 'NEW__' in i]+['INDEX']), on=['INDEX'], how = 'left')
            .join(df_3.select(*[i for i in df_3.columns if 'NEW__' in i]+['INDEX']), on=['INDEX'], how = 'left')
           )
 
   
    return df_f
   
 
def add_new_features_hefindahl(df):
   
    # hefindahl
    my_dict_hefindahl = create_dict_features(list_of_feats = sorted(columns_hefindahl[:], key=lambda x: (x.split('_')[-2], int(x.split('_')[-1][:-1]))),
                                             drop_cols_dict = {'cnpj_radical', 'data_ref', 'date_ref', 'paga_cpf_cnpj',
                                                            'secao_cnae', 'divisao_cnae', 'grupo_cnae', 'classe_cnae',
                                                            'subclasse_cnae', 'nm_porte_empr', 'flag_socio_pj', 'flag_socio_pf',
                                                            'flag_socio_estr', 'flag_radical_tem_filial', 'flag_filial'},                                              
                                             col_names = ['ind_Hefindahl_H_paga', 'ind_Hefindahl_H_bene',],
                                             sort_list = False)
    list_1_hefindahl, list_2_hefindahl = create_list_feats(my_dict_hefindahl)
    list_2_hefindahl = sorted(list_2_hefindahl, key=lambda x: (x.split('_')[-2] == 'bene', x))
    list_2_hefindahl = sorted(list_2_hefindahl, key=lambda x: (not x.startswith('ind_Hefindahl_H_bene'), int(x.split('_')[-1][:-1])))    
   
    # Adding new features
   
    df_1 = create_new_features_part_i(df.select(*sorted(list(set(list_1_hefindahl+list_2_hefindahl)))+['INDEX']), feat_list = sorted(list(set(list_1_hefindahl+list_2_hefindahl))))    
    df_2 = create_new_features_part_ii(df.select(*sorted(list(set(list_1_hefindahl+list_2_hefindahl)))+['INDEX']), list1 = list_1_hefindahl, list2 = list_2_hefindahl)    
    df_3 = create_new_features_part_iii(df.select(*sorted(list(set(list_1_hefindahl+list_2_hefindahl)))+['INDEX']), contain_list = ['ind_Hefindahl_H_paga', 'ind_Hefindahl_H_bene',], list_1=list_1_hefindahl, list_2=list_2_hefindahl)
     
    df_f = (df_1.select(*[i for i in df_1.columns if 'NEW__' in i]+['INDEX'])
            .join(df_2.select(*[i for i in df_2.columns if 'NEW__' in i]+['INDEX']), on=['INDEX'], how = 'left')
            .join(df_3.select(*[i for i in df_3.columns if 'NEW__' in i]+['INDEX']), on=['INDEX'], how = 'left')
           )
   
    return df_f
 
def add_new_features_indice_liquidez(df):
   
    # indice_liquidez
    my_dict_id_liq = create_dict_features(list_of_feats = columns_indice_liquidez,
                                          drop_cols_dict = {'cnpj_radical', 'data_ref', 'date_ref', 'paga_cpf_cnpj',
                                                            'secao_cnae', 'divisao_cnae', 'grupo_cnae', 'classe_cnae',
                                                            'subclasse_cnae', 'nm_porte_empr', 'flag_socio_pj', 'flag_socio_pf',
                                                            'flag_socio_estr', 'flag_radical_tem_filial', 'flag_filial'},
                                          col_names = ['cedente_indice_liquidez', 'sacado_indice_liquidez',],
                                          sort_list = False)
    list_1_id_liq, list_2_id_liq = create_list_feats(my_dict_id_liq)  
   
    # Adding new features
    df_1 = create_new_features_part_i(df.select(* sorted(list(set(list_1_id_liq+list_2_id_liq)))+['INDEX']), feat_list = sorted(list(set(list_1_id_liq+list_2_id_liq))))
    df_2 = create_new_features_part_ii(df.select(* sorted(list(set(list_1_id_liq+list_2_id_liq)))+['INDEX']), list1 = list_1_id_liq, list2 = list_2_id_liq)
    df_3 = create_new_features_part_iii(df.select(*sorted(list(set(list_1_id_liq+list_2_id_liq)))+['INDEX']), contain_list = ['cedente_indice_liquidez', 'sacado_indice_liquidez',], list_1=list_1_id_liq, list_2=list_2_id_liq)
   
    df_f = (df_1.select(*[i for i in df_1.columns if 'NEW__' in i]+['INDEX'])
            .join(df_2.select(*[i for i in df_2.columns if 'NEW__' in i]+['INDEX']), on=['INDEX'], how = 'left')
            .join(df_3.select(*[i for i in df_3.columns if 'NEW__' in i]+['INDEX']), on=['INDEX'], how = 'left')
           )
 
    return df_f
 
def add_new_features_ranking_pcr(df):
 
    # ranking_pcr
    my_dict_rk_pcr = create_dict_features(list_of_feats = columns_ranking_pcr,
                                          drop_cols_dict = {'cnpj_radical', 'data_ref', 'date_ref', 'paga_cpf_cnpj',
                                                            'secao_cnae', 'divisao_cnae', 'grupo_cnae', 'classe_cnae',
                                                            'subclasse_cnae', 'nm_porte_empr', 'flag_socio_pj', 'flag_socio_pf',
                                                            'flag_socio_estr', 'flag_radical_tem_filial', 'flag_filial'},
                                          col_names = ['vl_pago_boletos', 'vl_recebido_boletos', ],
                                          sort_list = False)
   
    list_1_rk_pcr, list_2_rk_pcr = create_list_feats(my_dict_rk_pcr)    
   
    # Adding new features
    df_1 = create_new_features_part_i(df.select(* sorted(list(set(list_1_rk_pcr+list_2_rk_pcr)))+['INDEX']), feat_list = sorted(list(set(list_1_rk_pcr+list_2_rk_pcr))))
    df_2 = create_new_features_part_ii(df.select(* sorted(list(set(list_1_rk_pcr+list_2_rk_pcr)))+['INDEX']), list1 = list_1_rk_pcr, list2 = list_2_rk_pcr)
    df_3 = create_new_features_part_iii(df.select(*sorted(list(set(list_1_rk_pcr+list_2_rk_pcr)))+['INDEX']),
                                        contain_list = ['vl_pago_boletos', 'vl_recebido_boletos',],
                                        list_1=list_1_rk_pcr, list_2=list_2_rk_pcr
                                     )
 
    df_f = (df_1.select(*[i for i in df_1.columns if 'NEW__' in i]+['INDEX'])
            .join(df_2.select(*[i for i in df_2.columns if 'NEW__' in i]+['INDEX']), on=['INDEX'], how = 'left')
            .join(df_3.select(*[i for i in df_3.columns if 'NEW__' in i]+['INDEX']), on=['INDEX'], how = 'left')
           )
 
    return df_f
 
def add_new_features_trend(df):
   
    # ranking_pcr
    my_dict_rk_pcr = create_dict_features(list_of_feats = columns_trend,
                                          drop_cols_dict = {'cnpj_radical', 'data_ref', 'date_ref', 'paga_cpf_cnpj',
                                                            'secao_cnae', 'divisao_cnae', 'grupo_cnae', 'classe_cnae',
                                                            'subclasse_cnae', 'nm_porte_empr', 'flag_socio_pj', 'flag_socio_pf',
                                                            'flag_socio_estr', 'flag_radical_tem_filial', 'flag_filial'},
                                          col_names = ['vl_recebido'],
                                          sort_list = False)
   
    list_1_trend, list_2_trend = create_list_feats(my_dict_rk_pcr)
    list_trend_final = (list(set(list_1_trend+list_2_trend)))
   
    # Adding new features
    df_1 = create_new_features_part_i(df.select(* sorted(list(set(list_trend_final)))+['INDEX']), feat_list = sorted(list(set(list_trend_final))))
    df_f = (df_1.select(*[i for i in df_1.columns if 'NEW__' in i]+['INDEX']))
 
    return df_f
 
 
def create_more_features_into_dataset(df, select_enrich_bases=['materialidade']):
   
    new_feats_bases = {}
    for bs in select_enrich_bases:
        if bs  == 'materialidade_90':
            new_feats_bases[bs] = add_new_features_materialidade_90(df)
 
        elif bs == 'hefindahl':
            new_feats_bases[bs] = add_new_features_hefindahl(df)
 
        elif bs == 'indice_liquidez':
            new_feats_bases[bs] = add_new_features_indice_liquidez(df)
 
        elif bs == 'ranking_pcr':
            new_feats_bases[bs] = add_new_features_ranking_pcr(df)
 
        elif bs == 'trend':
            new_feats_bases[bs] = add_new_features_trend(df)
       
    for key_name in new_feats_bases.keys():
        # Join each DataFrame with base_cliente
        df = df.join(new_feats_bases[key_name], on=['INDEX'], how='left')
 
    #df = df.localCheckpoint()
   
    return df  
 
def sampling_process_by_target(df, date_column = 'date', target_column = 'tgt', sample_size_stratify=0.4, seed=10):
    """
    Perform a stratified split in PySpark DataFrame.
 
    Parameters:
    - df (DataFrame): Input PySpark DataFrame.
    - stratify_cols (list): List of column names to stratify by.
    - test_size (float): Proportion of data to include in the test set (default 0.4).
    - seed (int): Random seed for reproducibility (default 10).
 
    Returns:
    - DataFrame: Test set with stratified sampling.
    """
    # Step 1: Add a composite column for stratification
    df = df.withColumn('target_stratify', F.col(target_column).cast('string')).withColumn('date_ref_stratify', F.col(date_column).cast('string'))
    df = df.withColumn("composite_stratify", F.concat_ws("_", F.col('date_ref_stratify'), F.col('target_stratify')))
 
    # Step 2: Add a row number within each stratified group
    window_spec = Window.partitionBy("composite_stratify").orderBy(F.rand(seed=10))  # Seed for reproducibility
    df = df.withColumn("row_num_stratify", F.row_number().over(window_spec))
 
    # Step 3: Calculate the number of rows for the test set per stratified group
    group_counts = (df
                    .groupBy("composite_stratify")
                    .agg(F.count("*").alias("total_count_stratify"))
                    .withColumn("test_count_stratify", (F.col("total_count_stratify") * sample_size_stratify).cast("int")))
 
    # Step 4: Join to determine which rows go to the test set
    df = df.join(group_counts, on="composite_stratify", how="inner")
 
    # Step 5: Assign test or train based on row_num and test_count
    df = df.withColumn("split_stratify", F.when(F.col("row_num_stratify") <= F.col("test_count_stratify"), "test_stratify").otherwise("train_stratify"))
 
    # Step 6: Filter only test data (similar to df_resampled)
    final_sample_df = df.filter(F.col("split_stratify") == "test_stratify").drop("date_ref_stratify",
                                                                                 "target_stratify",
                                                                                 "row_num_stratify",
                                                                                 "total_count_stratify",
                                                                                 "test_count_stratify",
                                                                                 "composite_stratify",
                                                                                 "split_stratify")
 
    return final_sample_df
 
def run_enrichment_process(spark_session_name = 'customizacao', # add o nome que quiser
                           customer_data_path = 's3://processado-ciencia/', # declarar o path da base cliente no S3
                           customer_sep_or_delimiter= ",", # declarar se a base do cliente possui algum delimiter (Ex: ';')
                           nuclea_lake_path = 's3://mntz-datascience/', # declarar o path no S3 onde as bases pre-processadas estarão salvas
                           enriched_dataset_stage_1_path = '', # declarar path do dataset enriquecido com a Fase 1
                           cnpj_column = 'id', # alterar conforme a base de dados do cliente
                           cnpj_type = 'radical', # alterar conforme a base de dados do cliente
                           date_column = 'dt', # alterar conforme a base de dados do cliente
                           date_format='YYYmm',
                           target_column = 'tg', # alterar conforme a base de dados do cliente
                           target_0 = "",
                           target_1 = "",
                           random_sample=False, # amostragem randômica
                           fraction = 1, # fração/porcentagem da amostragem randômica
                           seed = 47,
                           sample_by_target=False, # ativar amostragem estratificada
                           fraction_stratify = 0.4, # definir porcentagem da base estratificada (proporção da target binária por cada safra                          
                           n_recent_months_oot = 1, # alterar conforme decisão da Núclea
                           select_enrich_bases = ['materialidade_90', 'hefindahl',], # nomes padrão das bases (Sugestão)
                           activate_feature_selection=False,
                           final_feature_columns = ['vlr', 'data'],
                           save_into_s3 = False, # salvar no S3 caso seja TRUE
                           save_into_mlflow = False, # salvar no MLFLOW caso seja TRUE
                           experiment_name = '', # nome da pasta no MLFlow  
                           run_name = '', # nome do experimento
                               ):
   
    #MLFLOW
    experiment_id = get_or_create_experiment(experiment_name=experiment_name, artifact_location="s3://nuclea-mntz-ia-customizacao/ia_customizacao_dev")
   
   
 
 
    with mlflow.start_run(experiment_id=experiment_id, run_name=run_name):
 
        if not enriched_dataset_stage_1_path or enriched_dataset_stage_1_path == "None":
 
            if select_enrich_bases is None or select_enrich_bases == []:
                raise ValueError("Please provide a list of desired datasets to enrich the customer's database.")            
                                   
            print('Processing the addition of new variables into the customer database...please wait...', flush=True)        
            print()
 
 
            # Customer Dataset (le base do cliente e deixa as colunas originais em caixa alta)
            base_cliente, cols_, nrows_ = read_dataset_with_index_columns(file_path=customer_data_path,
                                                                        random_sample=random_sample,
                                                                        session_name = spark_session_name,
                                                                        sep_or_delimiter=customer_sep_or_delimiter,
                                                                        fraction=fraction,
                                                                        seed=seed,
                                                                       
                                                                        )
            #print(base_cliente.count())
            base_cliente = remove_duplicates(df=base_cliente, col_id = cnpj_column, date_col=date_column)
            print(f'{nrows_ - base_cliente.count()} duplicated row(s) has(ve) been removed!')                      
            print()
            print('Customer database has been uploaded')
            print()        
 
            # Data Treatments (tratamento necessarios)
            base_cliente = filter_binary_target(df=base_cliente, target_col=target_column, positive_label=target_0, negative_label=target_1)
            base_cliente = add_cnpj_type(df=base_cliente, id_customer_col=cnpj_column)
            base_cliente = remove_caracteres_zeros_esquerda(df=base_cliente, id_customer_col=cnpj_column, cnpj_type_str=cnpj_type)
            base_cliente = convert_year_month_col(df=base_cliente, date_column_col=date_column,date_format=date_format)              
                       
            if sample_by_target:
             
                base_cliente = sampling_process_by_target(df=base_cliente,
                                                          date_column = date_column,
                                                          target_column = target_column,
                                                          sample_size_stratify=fraction_stratify,
                                                          seed=seed)
                print(f'Number of sample rows: {base_cliente.count()}')
       
            dataset: SparkDataset =  mlflow.data.from_spark(base_cliente, path=customer_data_path)
            mlflow.log_input(dataset, context="raw_sample")
       
            # All Year-Month list
            safras = (base_cliente.select('data_ref1').distinct().orderBy('data_ref1').rdd.flatMap(list).collect())
            safras_for_shift = [(ref_date - relativedelta(months=1)).strftime("%Y-%m-01") for ref_date in safras]
            safras_for_shift = sorted(safras_for_shift)
           
            # nunique()
            #cnpj_type_n = (base_cliente.select("tipo_cnpj").distinct().count())
           
            # unique()
            cnpj_type_str = cnpj_type.lower()
            #cnpj_type_str = base_cliente.select("tipo_cnpj").distinct().collect()[0][0]
            #print(cnpj_type_str)
            #print()
            #print('Base cliente')
            #print(base_cliente.printSchema())
 
            #print()
           
            # Part: RADICAL
            if cnpj_type_str == 'radical':
            #if cnpj_type_n == 1 and cnpj_type_str == 'radical':
                   
               
                #cnpjs_unicos = 0#(base_cliente.select('cnpj_radical').distinct().rdd.map(lambda row: row.cnpj_radical).collect())
               
                enrich_bases = {}
               
                for bs in select_enrich_bases:  
                   
                    if bs  == 'materialidade_90':
                       
                        #for dt in year:
                        enrich_bases[bs] = materialidade_90_dias_radical(path = 's3://mntz-scores/current/layer=silver/abt/cnpj_type=completo/score=materialidade_90_dias_evolucao/version=01/02/',
                                                                        features = columns_mat_90,
                                                                        safras=safras,
                                                                        )                
                    elif bs == 'hefindahl':
                        enrich_bases[bs] = get_feature_table_with_shift(entity_column='cnpj_radical',
                                                                        path = 's3://mntz-datascience/ft-store-v1/concentracoes_pag_rec_bol_radical.parquet/',
                                                                        features=columns_hefindahl,
                                                                        dates=safras_for_shift,
                                                                        feature_reference_date_column='mes',
                                                                        feature_reference_date_format='yyyy-MM-dd',
                                                                        )
                       
                    elif bs == 'indice_liquidez':
                        enrich_bases[bs] = get_feature_table_with_shift(entity_column='cnpj_radical',
                                                                        path = 's3://mntz-datascience/ft-store-v1/indice_liquidez_btg/cnpj_radical/',
                                                                        features = columns_indice_liquidez,
                                                                        dates=safras_for_shift,
                                                                        feature_reference_date_column='mes',
                                                                        feature_reference_date_format='yyyy-MM-dd',
                                                                        )
 
                    elif bs == 'ranking_pcr':
                        enrich_bases[bs] = get_feature_table_with_shift(entity_column='cnpj_radical',
                                                                        path = 's3://mntz-datascience/ft-store-v1/ranking_features_pcr_pjradical/',
                                                                        features = columns_ranking_pcr,
                                                                        dates=safras_for_shift,
                                                                        feature_reference_date_column='mes',
                                                                        feature_reference_date_format='yyyy-MM-dd',
                                                                        )
 
                    elif bs == 'trend':
                        enrich_bases[bs] = get_feature_table_with_shift(entity_column='cnpj_radical',
                                                                        path = 's3://mntz-datascience/ft-store-v1/trend_features_slc_pjradical/',
                                                                        features = columns_trend,
                                                                        dates=safras_for_shift,
                                                                        feature_reference_date_column='mes',
                                                                        feature_reference_date_format='yyyy-MM-dd',
                                                                        )
                    # If you need to add more dataset to enrich just include one more ELIF + def function as following
                    #elif bs == 'xxx':
                        #enrich_bases[bs] = dataset_name_radical(path = 's3://mntz-datascience/XXXXXXXX/')                
                   
 
                # FINAL JOIN
 
                # Loop through each DataFrame in the dictionary and join with base_cliente
                for key_name in enrich_bases.keys():
                   
                    # rename columns                    
                    enrich_bases[key_name] = (enrich_bases[key_name]
                                                .withColumn(f'FLAG_COBERTURA_DADOS_{key_name}', F.lit(1))
                                                .withColumnRenamed('cnpj_radical', 'radical_cnpj')  # Rename column
                                                .withColumn('radical_cnpj', F.col('radical_cnpj').cast(StringType()))  # Cast to StringType
                                                .withColumn('data_ref1', F.col('data_ref1').cast(StringType()))
                                                )
 
                    # Join each DataFrame with base_cliente
 
                    base_cliente = (base_cliente
                                    #.withColumn(cnpj_column, F.col(cnpj_column).cast(StringType()))
                                    .withColumn('radical_cnpj', F.col('radical_cnpj').cast(StringType()))
                                    .withColumn('data_ref1', F.col('data_ref1').cast(StringType()))
                                    )
                    base_cliente = (base_cliente
                                    .join(enrich_bases[key_name], on=['radical_cnpj', 'data_ref1'], how='left')
                                    .na.fill(value=0,subset=[f"FLAG_COBERTURA_DADOS_{key_name}"])
                                    )
                   
               
                # Additional transformations
                if 'materialidade_90' in select_enrich_bases:
                    base_cliente = (base_cliente
                                    #.withColumn('target_radical_temp', F.col(target_column))
                                    .withColumn('flag_cartao_3m', F.when(F.col('total_cartao_recebido_3M') > 0, 1).otherwise(0))
                                    .withColumn('flag_cartao_6m', F.when(F.col('total_cartao_recebido_6M') > 0, 1).otherwise(0))
                                    .withColumn('flag_cartao_12m', F.when(F.col('total_cartao_recebido_12M') > 0, 1).otherwise(0))                                    
                                    )
                           
               
            # Part: COMPLETO
            elif cnpj_type_str == 'completo':
            #elif cnpj_type_n == 1 and cnpj_type_str == 'completo':        
           
                enrich_bases = {}
                for bs in select_enrich_bases:  
                   
                    if bs  == 'materialidade_90':
                       
                        enrich_bases[bs] = materialidade_90_dias_completo(path = 's3://mntz-scores/current/layer=silver/abt/cnpj_type=completo/score=materialidade_90_dias_evolucao/version=01/02/',                                                                  
                                                                        features = columns_mat_90,
                                                                        safras=safras)
                       
                    elif bs == 'hefindahl':
                        enrich_bases[bs] = get_feature_table_with_shift(entity_column='cnpj',
                                                                        path = 's3://mntz-datascience/ft-store-v1/concentracoes_pag_rec_bol_cnpj.parquet/',
                                                                        features=columns_hefindahl,
                                                                        dates=safras_for_shift,
                                                                        feature_reference_date_column='mes',
                                                                        feature_reference_date_format='yyyy-MM-dd',
                                                                        )
                       
                    elif bs == 'indice_liquidez':
                        enrich_bases[bs] = get_feature_table_with_shift(entity_column='cnpj',
                                                                        path = 's3://mntz-datascience/ft-store-v1/indice_liquidez_btg/cnpj_completo/',
                                                                        features = columns_indice_liquidez,
                                                                        dates=safras_for_shift,
                                                                        feature_reference_date_column='mes',
                                                                        feature_reference_date_format='yyyy-MM-dd',
                                                                        )
 
 
 
                    elif bs == 'ranking_pcr':
                        enrich_bases[bs] = get_feature_table_with_shift(entity_column='cnpj',
                                                                        path = 's3://mntz-datascience/ft-store-v1/ranking_features_pcr_pj/',
                                                                        features = columns_ranking_pcr,
                                                                        dates=safras,
                                                                        feature_reference_date_column='mes',
                                                                        feature_reference_date_format='yyyy-MM-dd',
                                                                        )
 
                    elif bs == 'trend':
                        enrich_bases[bs] = get_feature_table_with_shift(entity_column='cnpj',
                                                                        path = 's3://mntz-datascience/ft-store-v1/trend_features_slc_pj/',
                                                                        features = columns_trend,
                                                                        dates=safras_for_shift,
                                                                        feature_reference_date_column='mes',
                                                                        feature_reference_date_format='yyyy-MM-dd',
                                                                        )
 
                # If you need to add more dataset to enrich just include one more ELIF + def function as following
                    #elif bs == 'xxx':
                        #enrich_bases[bs] = dataset_name_completo(path = 's3://mntz-datascience/XXXXXXXX/')
                   
               
                print('Adding features from FeatureStore into the customer dataframe...please wait...')
                # FINAL JOIN    
                # Loop through each DataFrame in the dictionary and join with base_cliente
                for key_name in enrich_bases.keys():
 
                    enrich_bases[key_name] = (enrich_bases[key_name]
                                                .withColumn(f'FLAG_COBERTURA_DADOS_{key_name}', F.lit(1))
                                                .withColumnRenamed('cnpj', 'completo_cnpj')
                                                .withColumn('completo_cnpj', F.col('completo_cnpj').cast(StringType()))  # Cast to StringType
                                                .withColumn('data_ref1', F.col('data_ref1').cast(StringType()))
                                            )
 
                    # Perform the join using the conditions
 
                    base_cliente = (base_cliente
                                    .withColumn('completo_cnpj', F.col(cnpj_column).cast(StringType()))
                                    .withColumn('data_ref1', F.col('data_ref1').cast(StringType()))
                                    )
                    base_cliente = (base_cliente
                                    .join(enrich_bases[key_name], on=['completo_cnpj', 'data_ref1'], how='left')
                                    .na.fill(value=0,subset=[f"FLAG_COBERTURA_DADOS_{key_name}"])
                                    )                    
                               
                # Additional transformations        
                if 'materialidade_90' in select_enrich_bases:
                    base_cliente = (base_cliente
                            #.withColumn('target_completo_temp', F.col(target_column))
                            .withColumn('flag_cartao_3m', F.when(F.col('total_cartao_recebido_3M') > 0, 1).otherwise(0))
                            .withColumn('flag_cartao_6m', F.when(F.col('total_cartao_recebido_6M') > 0, 1).otherwise(0))
                            .withColumn('flag_cartao_12m', F.when(F.col('total_cartao_recebido_12M') > 0, 1).otherwise(0))
                            )                        
               
            else:
                raise Exception('Base do cliente possui 2 tipos de CNPJ: radical e completo')                
            print()
            print('Features from FeatureStore has been added!')
            # Add cadastral features
            print()
            #print('Read cadastral dataset')
            #df_register = spark.read.table("mntztaks.tb_taks750t_estabelecimentos")
            #df_register = df_register.select(*['cd_cnpj_basi', 'cd_cep_logr', 'nm_uf_logr', 'nm_muni', 'cd_cnpj'],
            #                                    (F.date_format(F.to_date("dt_situ_cada", "yyyyMMdd"), "yyyy-MM-dd")).alias("dt_situ_cada_new"),
            #             )
            #df_register = remove_duplicates(df=df_register, col_id = 'cd_cnpj_basi', date_col='dt_situ_cada_new')            
           
            #if cnpj_type_str == 'radical':
            #
            #    df_register = df_register.withColumn('cd_cnpj', F.col('cd_cnpj').cast(StringType()))
            #    #df_register = df_register.withColumn('cd_cnpj', F.col('cd_cnpj').cast(LongType()))
            #    #df_register = df_register.withColumn('cd_cnpj', F.col('cd_cnpj').cast(DecimalType(38, 0)))
            #    #df_register = df_register.withColumn('cd_cnpj', F.col('cd_cnpj').cast(IntegerType()))
            #    
            #    base_cliente = (base_cliente
            #                    .join(
            #                        df_register.withColumnRenamed('cd_cnpj', 'radical_cnpj'),            #                        
            #                        on = 'radical_cnpj',
            #                        how = 'left')                        
            #                    .select('*',
            #                            F.substring(F.col('cd_cep_logr'), 1, 2).alias('NEW__cep_2_digitos'),
            #                            F.substring(F.col('cd_cep_logr'), 1, 3).alias('NEW__cep_3_digitos'),
            #                            F.substring(F.col('cd_cep_logr'), 1, 4).alias('NEW__cep_4_digitos'),
            #                            F.substring(F.col('cd_cep_logr'), 1, 5).alias('NEW__cep_5_digitos'),                            
            #                        )
            #                )  
            #elif cnpj_type_str == 'completo':
            #    df_register = df_register.withColumn('cd_cnpj', F.col('cd_cnpj').cast(StringType()))
            #    #df_register = df_register.withColumn('cd_cnpj', F.col('cd_cnpj').cast(LongType()))
            #    #df_register = df_register.withColumn('cd_cnpj', F.col('cd_cnpj').cast(DecimalType(38, 0)))
            #    #df_register = df_register.withColumn('cd_cnpj', F.col('cd_cnpj').cast(IntegerType()))
               
            #    base_cliente = (base_cliente
            #                    .join(df_register.withColumnRenamed('cd_cnpj', 'completo_cnpj'),
            #                        #df_register.select(*['cd_cep_logr', 'nm_uf_logr', 'nm_muni',], F.col('cd_cnpj').alias('completo_cnpj')),
            #                        on = 'completo_cnpj',
            #                        how = 'left')                        
            #                    .select('*',
            #                            F.substring(F.col('cd_cep_logr'), 1, 2).alias('NEW__cep_2_digitos'),
            #                            F.substring(F.col('cd_cep_logr'), 1, 3).alias('NEW__cep_3_digitos'),
            #                            F.substring(F.col('cd_cep_logr'), 1, 4).alias('NEW__cep_4_digitos'),
            #                            F.substring(F.col('cd_cep_logr'), 1, 5).alias('NEW__cep_5_digitos'),                            
            #                        )
            #                )  
           
            # Turn into float format
            columns_to_cast = ["valor_recebido_boletos_3M", "valor_recebido_boletos_6M", "valor_recebido_boletos_12M",
                            "total_cartao_recebido_3M", "total_cartao_recebido_6M", "total_cartao_recebido_12M",
                            "total_ted_recebido_3M", "total_ted_recebido_6M", "total_ted_recebido_12M",
                            "total_recebido_3M", "total_recebido_6M", "total_recebido_12M"]
 
            # Apply the casting using a single select statement
            base_cliente = base_cliente.select(*[F.col(column).cast(FloatType()).alias(column) if column in columns_to_cast else F.col(column) for column in base_cliente.columns])
 
            # Drop variáveis temporárias
            #cols_to_drop_before_cache = ['target_radical_temp', 'target_completo_temp',]
            #base_cliente = base_cliente.drop(*[col for col in cols_to_drop_before_cache if col in base_cliente.columns])                      
            save_dataset(base_cliente, path=nuclea_lake_path, file_name='base_cliente_enriquecida_fase_1')
            print('Base_cliente_enriquecida_fase_1 has been saved into S3')            
            print()
            base_cliente.unpersist()
 
            enriched_path = nuclea_lake_path + 'base_cliente_enriquecida_fase_1'
 
        else:
            enriched_path = enriched_dataset_stage_1_path
       
        base_cliente = spark.read.parquet(enriched_path)
        safras_for_split = (base_cliente.select('data_ref1').distinct().orderBy('data_ref1').rdd.flatMap(list).collect())
 
        #print()
        print('Applying cache function to the customer dataframe...it may take a few minutes...please wait...')
        print()
        base_cliente.cache()
        base_cliente.count() # obrigatório para "setar" o cache()
        #base_cliente = base_cliente.localCheckpoint()        
   
        print('Cache has been applied!')
        print()        
        print('Starting Data Splitting Process...please wait...')
        # Samples for modeling process
        df_treino, df_test, df_val, df_oot = split_abt_into_samples(abt_df=base_cliente,
                                                                    safras=safras_for_split,
                                                                    n_recent_months=n_recent_months_oot,
                                                                    col_data='data_ref1',
                                                                    col_id = cnpj_column,)
        print()
        print('Splitting Process is done!')
        print()
       
        # add more new features
        print('Adding more new features into the dataframe...please wait...')
        #base_cliente = create_more_features_into_dataset(df=base_cliente, select_enrich_bases=select_enrich_bases)
        df_treino = create_more_features_into_dataset(df=df_treino, select_enrich_bases=select_enrich_bases)
        df_test = create_more_features_into_dataset(df=df_test, select_enrich_bases=select_enrich_bases)
        df_val = create_more_features_into_dataset(df=df_val, select_enrich_bases=select_enrich_bases)
        df_oot = create_more_features_into_dataset(df=df_oot, select_enrich_bases=select_enrich_bases)
 
        # drop duplicates
        #base_cliente = base_cliente.dropDuplicates(['INDEX'])
        df_treino = df_treino.dropDuplicates(['INDEX'])
        df_test = df_test.dropDuplicates(['INDEX'])
        df_val = df_val.dropDuplicates(['INDEX'])
        df_oot = df_oot.dropDuplicates(['INDEX'])
 
 
        # Drop variáveis
        cols_to_drop_before_save_into_s3 = ['INDEX', 'num_digitos_cnpj', 'tipo_cnpj', 'radical_cnpj', 'completo_cnpj', 'target_radical_temp', 'target_completo_temp',]        
        #base_cliente = base_cliente.drop(*[col for col in cols_to_drop_before_save_into_s3 if col in base_cliente.columns])
        df_treino = df_treino.drop(*[col for col in cols_to_drop_before_save_into_s3 if col in df_treino.columns])
        df_test = df_test.drop(*[col for col in cols_to_drop_before_save_into_s3 if col in df_test.columns])
        df_val = df_val.drop(*[col for col in cols_to_drop_before_save_into_s3 if col in df_val.columns])
        df_oot = df_oot.drop(*[col for col in cols_to_drop_before_save_into_s3 if col in df_oot.columns])
 
        if activate_feature_selection:
 
            # Definindo as colunas a serem selecionadas
            selected_columns = [cnpj_column, date_column, 'data_ref1', target_column] + final_feature_columns
           
            # Aplicando a seleção de colunas para cada DataFrame
            #base_cliente = base_cliente.select(*selected_columns)
            df_treino = df_treino.select(*selected_columns)
            df_test = df_test.select(*selected_columns)
            df_val = df_val.select(*selected_columns)
            df_oot = df_oot.select(*selected_columns)
 
        print()
        print('New features has been added into the dataframe!')
        print()
        # Save enriched datasets into cloud
        if save_into_s3:
            print()
            print('Uploading into S3')
            #print()
            #print('......starting with customer database...')
            #save_dataset(df=base_cliente, path=nuclea_lake_path, file_name='base_cliente_enriquecida')                
            #base_cliente.unpersist()
            #print()
            #print('......customer database is done!')
            print()
            #print('......next Training Set...')
            print('......starting with Training Set...')
            save_dataset(df=df_treino, path=nuclea_lake_path, file_name='base_treino')
            df_treino.unpersist()
            print()
            print('......Training Set is done!')
            print()
            print('......next Test Set...')
            save_dataset(df=df_test, path=nuclea_lake_path, file_name='base_teste')
            df_test.unpersist()
            print()
            print('......Test Set is done!')
            print()
            print('......next Validation Set...')
            save_dataset(df=df_val, path=nuclea_lake_path, file_name='base_validacao')
            df_val.unpersist()
            print()
            print('......Validation Set is done!')
            print()
            print('......next OOT Set...')
            save_dataset(df=df_oot, path=nuclea_lake_path, file_name='base_oot')
            df_oot.unpersist()
            print()
            print('......OOT Set is done!')
            print()
       
        if save_into_mlflow:
            print("Warning: MLflow has a schema size limitation, which restricts logging large datasets.")
            print('Uploading into MLFlow only a sample and the TOP500 first features...please wait...')
           
           
            # mlflow.exceptions.RestException: INVALID_PARAMETER_VALUE: Dataset schema exceeds the maximum length of 65535
 
            #MLFLOW
            #dataset: SparkDataset =  mlflow.data.from_spark(base_cliente, path=nuclea_lake_path+'base_cliente_enriquecida')
            #mlflow.log_input(dataset, context="enriched_sample")      
           
            #MLFLOW            
            dataset: SparkDataset =  mlflow.data.from_spark(df_treino
                                                            .select(*df_treino.columns[:500])
                                                            .sample(fraction=0.25, seed=seed),
                                                            path=nuclea_lake_path+'base_treino')
            mlflow.log_input(dataset, context="training")          
           
            #MLFLOW
            dataset: SparkDataset =  mlflow.data.from_spark(df_test
                                                            .select(*df_test.columns[:500])
                                                            .sample(fraction=0.25, seed=seed),
                                                            path=nuclea_lake_path+'base_teste')
            mlflow.log_input(dataset, context="testing")          
           
            #MLFLOW
            dataset: SparkDataset =  mlflow.data.from_spark(df_val
                                                            .select(*df_val.columns[:500])
                                                            .sample(fraction=0.25, seed=seed),
                                                            path=nuclea_lake_path+'base_validacao')
            mlflow.log_input(dataset, context="validation")            
             
            #MLFLOW
            dataset: SparkDataset =  mlflow.data.from_spark(df_oot
                                                            .select(*df_oot.columns[:500])
                                                            .sample(fraction=0.25, seed=seed),
                                                            path=nuclea_lake_path+'base_oot')
            mlflow.log_input(dataset, context="out-of-time")
            print('...the datasets have been saved into MLFlow!')            
       
        print()
        print('DONE!')
 
        try:
            return df_treino, df_test, df_val, df_oot, safras
       
        except: # caso a opção enriched_dataset_stage_1_path seja ativada
            return df_treino, df_test, df_val, df_oot, safras_for_split
        #return base_cliente, df_treino, df_test, df_val, df_oot, safras
 
args = parser.parse_args()
 
# Parte do USER
(
#base_enriquecida_cliente,
 base_treino,
 base_teste,
 base_validacao,
 base_oot,
 safras) = run_enrichment_process(spark_session_name = 'ia_de_customizacao',
                                  customer_data_path = args.customer_data_path,
                                  customer_sep_or_delimiter = args.customer_sep_or_delimiter,
                                  nuclea_lake_path = args.nuclea_lake_path,
                                  enriched_dataset_stage_1_path=args.enriched_dataset_stage_1_path,
                                  cnpj_column = args.cnpj_column,
                                  cnpj_type= args.cnpj_type,
                                  date_column = args.date_column,
                                  date_format = args.date_format,
                                  target_column = args.target_column,
                                  target_0 =  args.target_0,
                                  target_1 =  args.target_1,
                                  random_sample = args.random_sample,
                                  fraction = args.fraction,
                                  sample_by_target = args.sample_by_target,
                                  fraction_stratify = args.fraction_stratify,                                  
                                  seed = args.seed,
                                  n_recent_months_oot = args.n_recent_months_oot,
                                  select_enrich_bases = args.select_enrich_bases,
                                  activate_feature_selection=args.activate_feature_selection,
                                  final_feature_columns=args.final_feature_columns,
                                  save_into_s3 = args.save_into_s3,
                                  save_into_mlflow = args.save_into_mlflow,
                                  experiment_name = args.experiment_name,
                                  run_name = args.run_name
                                 )
 