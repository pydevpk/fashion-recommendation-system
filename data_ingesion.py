import os

import pandas as pd
import sqlalchemy
from dotenv import load_dotenv

load_dotenv('env.txt')

DB_USER = os.getenv('DB_USER')
DB_PASSWORD = os.getenv('DB_PASSWORD')
DB_NAME = os.getenv('DB_NAME')
DB_HOST = os.getenv('DB_HOST')
df = pd.read_csv("ASHI_FINAL_DATA.csv")

conn = sqlalchemy.create_engine(f'mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:3306/{DB_NAME}')

df.to_sql(name='main', con=conn, if_exists='replace', index=False)
print('Done.............')