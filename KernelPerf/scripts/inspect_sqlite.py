import sqlite3
import sys

db = sqlite3.connect(sys.argv[1])
print("jobs", db.execute("select status, count(*) from jobs group by status").fetchall())
print("results", db.execute("select status, count(*) from results group by status").fetchall())
print("logs", db.execute("select message from job_logs order by log_id desc limit 3").fetchall())
