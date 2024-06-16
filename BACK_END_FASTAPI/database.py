import urllib.parse as up
import psycopg2

def connect_db_cloud():
    DATABASE_URL = "postgres://vbtvichp:3gRKcXgL5Y3fx1ptlQwcRRBushDln2hz@lallah.db.elephantsql.com/vbtvichp"
    up.uses_netloc.append("postgres")
    url = up.urlparse(DATABASE_URL)

    # print("url",url)
    conn = psycopg2.connect(database=url.path[1:],
    user=url.username,
    password=url.password,
    host=url.hostname,
    port=url.port
    )
    return conn