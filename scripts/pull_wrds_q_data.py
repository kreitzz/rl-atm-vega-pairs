"""Pull the daily data for the Q-learning experiment (JPM/BAC + AMD/NVDA).

Run from the repo root:
    pip install wrds pandas pyarrow
    python scripts/pull_wrds_q_data.py

You will be asked for your WRDS username/password (it can save a .pgpass).
Outputs go to data/wrds_q/ :
    iv.csv         date, ticker, secid, iv        (OptionMetrics 30d 50-delta call IV, in %)
    returns.csv    date, ticker, permno, retx     (CRSP daily ex-dividend returns)
    earnings.csv   ticker, rdq                    (Compustat quarterly report dates)
    secid_check.csv  candidate OptionMetrics secids per ticker, for a sanity check
"""
from pathlib import Path

import pandas as pd
import wrds

TICKERS = ["AMD", "NVDA", "JPM", "BAC"]
# Known identifiers. JPM/BAC secids are looked up below.
KNOWN_SECID = {"AMD": 101121, "NVDA": 108321}
KNOWN_PERMNO = {"AMD": 61241, "NVDA": 86580, "JPM": 47896, "BAC": 59408}
START, END = 2002, 2025
OUT = Path("data/wrds_q")
OUT.mkdir(parents=True, exist_ok=True)

db = wrds.Connection()

# ---------- 1. OptionMetrics secids ----------
tick_list = ",".join(f"'{t}'" for t in TICKERS)
cands = db.raw_sql(f"""
    select distinct secid, ticker, issuer, effect_date
    from optionm.secnmd
    where ticker in ({tick_list})
    order by ticker, effect_date
""")
# Pick, per ticker, the secid with the most 30d/50-delta surface rows in a recent year.
counts = db.raw_sql(f"""
    select secid, count(*) as n
    from optionm.vsurfd2020
    where secid in ({",".join(str(int(s)) for s in cands.secid.unique())})
      and days = 30 and delta = 50 and cp_flag = 'C'
    group by secid
""")
cands = cands.merge(counts, on="secid", how="left").fillna({"n": 0})
cands.to_csv(OUT / "secid_check.csv", index=False)

secid = dict(KNOWN_SECID)
for t in ["JPM", "BAC"]:
    c = cands[cands.ticker == t].sort_values("n", ascending=False)
    secid[t] = int(c.secid.iloc[0])
print("Using secids:", secid)

# ---------- 2. Implied vol, year by year ----------
secid_list = ",".join(str(s) for s in secid.values())
frames = []
for year in range(START, END + 1):
    table = f"optionm.vsurfd{year}"
    try:
        df = db.raw_sql(f"""
            select secid, date, impl_volatility
            from {table}
            where secid in ({secid_list})
              and days = 30 and delta = 50 and cp_flag = 'C'
        """, date_cols=["date"])
    except Exception as e:  # table for the latest year may not exist yet
        print(f"skip {table}: {e}")
        continue
    print(year, len(df))
    frames.append(df)
iv = pd.concat(frames, ignore_index=True)
inv = {v: k for k, v in secid.items()}
iv["ticker"] = iv.secid.map(inv)
iv["iv"] = iv.impl_volatility * 100
iv = iv[iv.iv > 0][["date", "ticker", "secid", "iv"]].sort_values(["ticker", "date"])
iv.to_csv(OUT / "iv.csv", index=False)

# ---------- 3. CRSP daily returns ----------
permno_list = ",".join(str(p) for p in KNOWN_PERMNO.values())
ret = db.raw_sql(f"""
    select permno, date, retx
    from crsp.dsf
    where permno in ({permno_list})
      and date between '{START}-01-01' and '{END}-12-31'
""", date_cols=["date"])
ret["ticker"] = ret.permno.map({v: k for k, v in KNOWN_PERMNO.items()})
ret = ret[["date", "ticker", "permno", "retx"]].sort_values(["ticker", "date"])
ret.to_csv(OUT / "returns.csv", index=False)

# Sanity check: the permnos really are these tickers.
names = db.raw_sql(f"""
    select permno, ticker, comnam, namedt, nameenddt
    from crsp.stocknames where permno in ({permno_list})
""")
print(names.to_string())

# ---------- 4. Earnings dates ----------
earn = db.raw_sql(f"""
    select tic as ticker, rdq
    from comp.fundq
    where tic in ({tick_list})
      and rdq is not null
      and rdq between '{START - 1}-01-01' and '{END}-12-31'
""", date_cols=["rdq"])
earn = earn.drop_duplicates().sort_values(["ticker", "rdq"])
earn.to_csv(OUT / "earnings.csv", index=False)

db.close()
print("\nRows:", {"iv": len(iv), "returns": len(ret), "earnings": len(earn)})
print("Per-ticker IV date range:")
print(iv.groupby("ticker").date.agg(["min", "max", "count"]))
print(f"Done. Files in {OUT.resolve()}")
