import pandas as pd
from sqlalchemy import create_engine

DATABASE_URL = "mysql+pymysql://root:@127.0.0.1:3306/mewar_erp"
# If you have a password:
# DATABASE_URL = "mysql+pymysql://root:password@127.0.0.1:3306/mewar_erp"

engine = create_engine(DATABASE_URL)

df = pd.read_sql("SELECT * FROM inventories", engine)

print(df)
