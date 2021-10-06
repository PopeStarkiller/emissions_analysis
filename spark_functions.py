import os
import config
import platform
v1, v2, v3 = config.spark_env(platform.node())
os.environ['SPARK_VERSION'] = v1
os.environ['JAVA_HOME'] = v2
os.environ['SPARK_HOME'] = v3
import findspark
import pyspark
from pyspark.sql import SparkSession
findspark.init()
spark = SparkSession.builder.appName("emissionsdataframe").getOrCreate()
from sqlalchemy import create_engine, insert
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import MetaData, update, Table
from sqlalchemy.orm import Session
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import SQLAlchemyError
import pandas as pd
from pyspark.sql.types import IntegerType,BooleanType,DateType
from pyspark.sql.functions import col
from pyspark.sql import Row
from pyspark.sql.functions import array, col, explode, lit, struct, log
from pyspark.sql import DataFrame
from typing import Iterable
import numpy as np
import spark_functions
import tensorflow as tf
import keras.metrics
import pyspark.sql.functions as F
import math
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
import seaborn as sns
rds_string = config.rds_string
engine = create_engine(f'postgresql://{rds_string}')
conn = engine.connect()
metadata = MetaData(engine)



def sql_to_spark_gdp(country):
    from pyspark.sql import Row
    from sqlalchemy.orm import Session, sessionmaker
    from sqlalchemy import MetaData, Table
    metadata = MetaData(engine)
    table_var = Table("gdp_data", metadata, autoload=True, autoload_with=engine)
    Session = sessionmaker()
    Session.configure(bind=engine)
    session = Session()
    query = session.query(table_var).filter(table_var.c.Country == country).all()
    steve = table_var.metadata.tables["gdp_data"].columns.keys()
    query_list = []
    for i in query:
        q_len = len(i)
        temp_dict = {}
        for j in range(q_len):
            key = steve[j]
            value = i[j]
            if value == None:
                value = float(0)
            temp_dict[key] = value
            if j == (q_len - 1):
                query_list.append(temp_dict)
    df = spark.createDataFrame(Row(**x) for x in query_list)
    gdp_col = df.columns
    gdp_col2 = gdp_col[2:]
    gdp_static_cols = gdp_col[:2]
    df = melt(df, id_vars=gdp_static_cols, value_vars=gdp_col2)
    df = df.withColumn("variable",df.variable.cast('int'))
    df = df.withColumnRenamed("variable","year")
    df = df.withColumnRenamed("value","gdp")
    df = df.withColumnRenamed("Country","country")
    df = df.select(['country','year','gdp'])
    return df

def sql_to_spark_emissions(country):
    from pyspark.sql import Row
    from sqlalchemy.orm import Session, sessionmaker
    from sqlalchemy import MetaData, Table
    metadata = MetaData(engine)
    table_var = Table("global_emissions", metadata, autoload=True, autoload_with=engine)
    Session = sessionmaker()
    Session.configure(bind=engine)
    session = Session()
    query = session.query(table_var).filter(table_var.c.Entity == country).all()
    steve = table_var.metadata.tables["global_emissions"].columns.keys()
    query_list = []
    for i in query:
        q_len = len(i)
        temp_dict = {}
        for j in range(q_len):
            key = steve[j]
            value = i[j]
            if value == None:
                value = float(0)
            temp_dict[key] = value
            if j == (q_len - 1):
                query_list.append(temp_dict)
    df = spark.createDataFrame(Row(**x) for x in query_list)
    df = df.withColumnRenamed("Entity","country")
    df = df.withColumnRenamed("Year","year")
    return df

def sql_to_spark_trade(country):
    from sqlalchemy.orm import Session, sessionmaker
    from sqlalchemy import MetaData, Table
    metadata = MetaData(engine)
    table_var = Table("global_trade", metadata, autoload=True, autoload_with=engine)
    Session = sessionmaker()
    Session.configure(bind=engine)
    session = Session()
    query = session.query(table_var).filter(table_var.c.country_or_area == country).all()
    steve = table_var.metadata.tables["global_trade"].columns.keys()
    c_list = []
    for i in query:
        q_len = len(i)
        if i[0] == country:
            temp_dict = {}
            for j in range(q_len):
                key = steve[j]
                value = i[j]
                if value == None:
                    value = float(0)
                temp_dict[key] = value
                if j == (q_len - 1):
                    c_list.append(temp_dict)
    df = spark.createDataFrame(Row(**x) for x in c_list)
    df = df.withColumn("year",df.year.cast('int'))
    df = df.withColumnRenamed("country_or_area","country")
    df_export = df.filter(df['flow'] == "Export")
    df_import = df.filter(df['flow'] == "Import")
    return (df_export, df_import)

def melt(
        df: DataFrame, 
        id_vars: Iterable[str], value_vars: Iterable[str], 
        var_name: str="variable", value_name: str="value") -> DataFrame:
    """Convert :class:`DataFrame` from wide to long format."""

    # Create array<struct<variable: str, value: ...>>
    _vars_and_vals = array(*(
        struct(lit(c).alias(var_name), col(c).alias(value_name)) 
        for c in value_vars))

    # Add to the DataFrame and explode
    _tmp = df.withColumn("_vars_and_vals", explode(_vars_and_vals))

    cols = id_vars + [
            col("_vars_and_vals")[x].alias(x) for x in [var_name, value_name]]
    return _tmp.select(*cols)

def min_max(df,column):
    max_ = df.agg({column: 'max'})
    min_ = df.agg({column: 'min'})
    return (min_, max_)

def min_int(df,column):
    arr = np.array(df.select(column).collect())
    y_l = []
    for i in arr:
        y_l.append(int(i[0]))
    min_ = min(y_l)
    max_ = max(y_l)
    return (min_, max_)  

def merged_df(country):
    if country == "United States":
        export_df, import_df = sql_to_spark_trade("USA")
    else:
        export_df, import_df = sql_to_spark_trade(country)
    export_df_min, export_df_max = min_int(export_df, 'year')
    import_df_min, import_df_max = min_int(import_df, 'year')

    gdp_spark = sql_to_spark_gdp(country)
    gdp_spark_min, gdp_spark_max = min_int(gdp_spark, "year")

    emissions_spark = sql_to_spark_emissions(country)
    emissions_min, emissions_max = min_int(emissions_spark, 'year')

    years_df = pd.DataFrame([{"min":export_df_min, "max":export_df_max},
    {"min":import_df_min, "max":import_df_max},
    {"min":gdp_spark_min, "max":gdp_spark_max},
    {"min":emissions_min, "max":emissions_max}])

    lower_bound = years_df['min'].max()
    upper_bound = years_df['max'].min()

    gdp_filt = gdp_spark.filter(gdp_spark["year"] >= lower_bound)
    gdp_filt = gdp_filt.filter(gdp_filt["year"] <= upper_bound)
    emissions_spark_filt = emissions_spark.filter(emissions_spark["year"] >= lower_bound)
    emissions_spark_filt = emissions_spark_filt.filter(emissions_spark_filt["year"] <= upper_bound)
    import_filt = import_df.filter(import_df["year"] >= lower_bound)
    import_filt = import_filt.filter(import_filt["year"] <= upper_bound)
    export_filt = export_df.filter(export_df["year"] >= lower_bound)
    export_filt = export_filt.filter(export_filt["year"] <= upper_bound)
    merged_df = gdp_filt.join(emissions_spark_filt, gdp_filt.year == emissions_spark_filt.year, 'outer') \
    .select(gdp_filt.country ,gdp_filt.year,gdp_filt.gdp, emissions_spark_filt.annual_co2_emissions_tonnes) \
    .distinct()
    merged_df = merged_df.orderBy(merged_df.year.asc())
    import_sum = import_filt.groupBy('year').sum()
    import_sum = import_sum.orderBy(import_sum.year.asc())
    import_sum = import_sum.withColumnRenamed("sum(trade_usd)","import_trade_sum_usd")
    import_sum = import_sum.withColumnRenamed("sum(weight_kg)","import_weight_sum_kg")
    import_sum = import_sum.withColumnRenamed("sum(quantity)","import_quantity_sum")
    export_sum = export_filt.groupBy('year').sum()
    export_sum = export_sum.orderBy(export_sum.year.asc())
    export_sum = export_sum.withColumnRenamed("sum(trade_usd)","export_trade_sum_usd")
    export_sum = export_sum.withColumnRenamed("sum(weight_kg)","export_weight_sum_kg")
    export_sum = export_sum.withColumnRenamed("sum(quantity)","export_quantity_sum")
    merged_df2 = merged_df.join(import_sum, merged_df.year == import_sum.year, 'outer') \
    .select(merged_df.country ,merged_df.year,merged_df.gdp, merged_df.annual_co2_emissions_tonnes,
        import_sum['import_trade_sum_usd'],import_sum['import_weight_sum_kg'],import_sum['import_quantity_sum']) \
    .distinct()
    merged_df2 = merged_df2.orderBy(merged_df2.year.asc())
    final_merged_spark = merged_df2.join(export_sum, merged_df2.year == export_sum.year, 'outer') \
    .select(merged_df2.country ,merged_df2.year,merged_df2.gdp, merged_df.annual_co2_emissions_tonnes,
        merged_df2['import_trade_sum_usd'],merged_df2['import_weight_sum_kg'],merged_df2['import_quantity_sum'],
        export_sum['export_trade_sum_usd'],export_sum['export_weight_sum_kg'],export_sum['export_quantity_sum']
        ) \
    .distinct()
    final_merged_spark = final_merged_spark.orderBy(final_merged_spark.year.asc())
    df = final_merged_spark
    df = df.withColumn("annual_co2_emissions_tonnes_log", F.log10(col("annual_co2_emissions_tonnes")))
    df = df.withColumn("import_trade_sum_usd_log", F.log10(col("import_trade_sum_usd")))
    df = df.withColumn("import_weight_sum_kg_log", F.log10(col("import_weight_sum_kg")))
    df = df.withColumn("import_quantity_sum_log", F.log10(col("import_quantity_sum")))
    df = df.withColumn("export_trade_sum_usd_log", F.log10(col("export_trade_sum_usd")))
    df = df.withColumn("export_weight_sum_kg_log", F.log10(col("export_weight_sum_kg")))
    df = df.withColumn("export_quantity_sum_log", F.log10(col("export_quantity_sum")))
    df = df.select('country',
                    'year',
                    'gdp',
                    'annual_co2_emissions_tonnes_log',
                    'import_trade_sum_usd_log',
                    'import_weight_sum_kg_log',
                    'import_quantity_sum_log',
                    'export_trade_sum_usd_log',
                    'export_weight_sum_kg_log',
                    'export_quantity_sum_log')


    columns = df.columns
   
    return (df, columns)

def merge_sparks(import_df,export_df,gdp_spark,emissions_spark):
    export_df_min, export_df_max = min_int(export_df, 'year')
    import_df_min, import_df_max = min_int(import_df, 'year')

    gdp_spark_min, gdp_spark_max = min_int(gdp_spark, "year")

    emissions_min, emissions_max = min_int(emissions_spark, 'year')

    years_df = pd.DataFrame([{"min":export_df_min, "max":export_df_max},
    {"min":import_df_min, "max":import_df_max},
    {"min":gdp_spark_min, "max":gdp_spark_max},
    {"min":emissions_min, "max":emissions_max}])

    lower_bound = years_df['min'].max()
    upper_bound = years_df['max'].min()

    gdp_filt = gdp_spark.filter(gdp_spark["year"] >= lower_bound)
    gdp_filt = gdp_filt.filter(gdp_filt["year"] <= upper_bound)
    emissions_spark_filt = emissions_spark.filter(emissions_spark["year"] >= lower_bound)
    emissions_spark_filt = emissions_spark_filt.filter(emissions_spark_filt["year"] <= upper_bound)
    import_filt = import_df.filter(import_df["year"] >= lower_bound)
    import_filt = import_filt.filter(import_filt["year"] <= upper_bound)
    export_filt = export_df.filter(export_df["year"] >= lower_bound)
    export_filt = export_filt.filter(export_filt["year"] <= upper_bound)
    merged_df = gdp_filt.join(emissions_spark_filt, gdp_filt.year == emissions_spark_filt.year, 'outer') \
    .select(gdp_filt.country ,gdp_filt.year,gdp_filt.gdp, emissions_spark_filt.annual_co2_emissions_tonnes) \
    .distinct()
    merged_df = merged_df.orderBy(merged_df.year.asc())
    import_sum = import_filt.groupBy('year').sum()
    import_sum = import_sum.orderBy(import_sum.year.asc())
    import_sum = import_sum.withColumnRenamed("sum(trade_usd)","import_trade_sum_usd")
    import_sum = import_sum.withColumnRenamed("sum(weight_kg)","import_weight_sum_kg")
    import_sum = import_sum.withColumnRenamed("sum(quantity)","import_quantity_sum")
    export_sum = export_filt.groupBy('year').sum()
    export_sum = export_sum.orderBy(export_sum.year.asc())
    export_sum = export_sum.withColumnRenamed("sum(trade_usd)","export_trade_sum_usd")
    export_sum = export_sum.withColumnRenamed("sum(weight_kg)","export_weight_sum_kg")
    export_sum = export_sum.withColumnRenamed("sum(quantity)","export_quantity_sum")
    merged_df2 = merged_df.join(import_sum, merged_df.year == import_sum.year, 'outer') \
    .select(merged_df.country ,merged_df.year,merged_df.gdp, merged_df.annual_co2_emissions_tonnes,
        import_sum['import_trade_sum_usd'],import_sum['import_weight_sum_kg'],import_sum['import_quantity_sum']) \
    .distinct()
    merged_df2 = merged_df2.orderBy(merged_df2.year.asc())
    final_merged_spark = merged_df2.join(export_sum, merged_df2.year == export_sum.year, 'outer') \
    .select(merged_df2.country ,merged_df2.year,merged_df2.gdp, merged_df.annual_co2_emissions_tonnes,
        merged_df2['import_trade_sum_usd'],merged_df2['import_weight_sum_kg'],merged_df2['import_quantity_sum'],
        export_sum['export_trade_sum_usd'],export_sum['export_weight_sum_kg'],export_sum['export_quantity_sum']
        ) \
    .distinct()
    final_merged_spark = final_merged_spark.orderBy(final_merged_spark.year.asc())
    return final_merged_spark
