"""
01_discover.py  -  profile the tables we need and write ONE text report.
Run:   python 01_discover.py "path/to/synthetic_microdata_2021-2024_XXXX"
Then review discovery_report.txt to map the variables needed.
Needs: pip install pandas
"""
import sys, glob, os
import pandas as pd

DATA = sys.argv[1] if len(sys.argv) > 1 else "."
OUT = open("discovery_report.txt", "w", encoding="utf-8")

def log(*a):
    s = " ".join(str(x) for x in a)
    print(s); OUT.write(s + "\n")

def profile(path, max_levels=25, nrows=None):
    df = pd.read_csv(path, dtype=str, nrows=nrows, low_memory=False)
    df.columns = [c.upper() for c in df.columns]
    log(f"\n=== {os.path.relpath(path, DATA)} | rows={len(df):,} cols={df.shape[1]}")
    log("columns:", list(df.columns))
    for c in df.columns:
        nun = df[c].nunique(dropna=True)
        fill = df[c].notna().mean()
        if c in ("NOMER",):
            log(f"  {c}: {nun:,} unique")
        elif nun <= max_levels:
            vc = df[c].value_counts(dropna=False).head(max_levels).to_dict()
            log(f"  {c}: filled={fill:.2f} levels={nun} {vc}")
        else:
            log(f"  {c}: filled={fill:.2f} unique={nun:,} e.g. {df[c].dropna().iloc[:4].tolist()}")
    return df

# 1) folder layout
log("MANIFEST (forms we need):")
man = pd.read_csv(f"{DATA}/MANIFEST.csv")
log(man.columns.tolist())
log(man[man.form.isin(["d002", "d004", "d006", "d008"])].to_string(max_rows=80))

for f in ("d002", "d006", "d008"):
    log(f"\n{f} files:", sorted(os.path.relpath(p, DATA) for p in glob.glob(f"{DATA}/{f}/*/*.csv")))
log("\nd004 folders 2024:", sorted(os.listdir(f"{DATA}/d004/2024")))

# 2) every d004 table of ONE quarter (2024 Q1) - the 12 tables
for p in sorted(glob.glob(f"{DATA}/d004/2024/1kv/*.csv")):
    profile(p)

# 3) the other forms, 2024
for p in sorted(glob.glob(f"{DATA}/d002/2024/*.csv")):
    profile(p)
profile(f"{DATA}/d008/2024/kontr_k.csv")
d6 = pd.read_csv(f"{DATA}/d006/2024/rio_2024.csv", dtype=str, nrows=5)
log("\n=== d006 columns:", [c.upper() for c in d6.columns])

# 4) do the same table names exist in 2021 (schema drift check)?
for y in (2021, 2022, 2023):
    names = sorted(os.path.basename(p) for p in glob.glob(f"{DATA}/d004/{y}/1kv/*.csv"))
    log(f"\nd004 {y} 1kv tables:", names)

# 5) documentation: d004 / d002 code books
log("\ndocumentation files for d004/d002/d006/d008:")
for p in sorted(glob.glob(f"{DATA}/documentation/**/*", recursive=True)):
    if os.path.isfile(p) and any(k in p.lower() for k in ("d004", "d002", "d006", "d008")):
        log("  ", os.path.relpath(p, DATA))
idx = f"{DATA}/documentation/INDEX.csv"
if os.path.exists(idx):
    i = pd.read_csv(idx)
    log("\nINDEX.csv columns:", i.columns.tolist())
    log(i[i.astype(str).apply(lambda r: r.str.contains("d004|d002", case=False).any(), axis=1)].to_string(max_rows=80, max_colwidth=90))

# 6) 1-T annual (enterprise labour) - HUGE, so only a sample of rows
log("\n\n##### 1t_god (sample only) #####")
for p in sorted(glob.glob(f"{DATA}/1t_god/*/*.csv")):
    log("file:", os.path.relpath(p, DATA), f"{os.path.getsize(p)/1e6:.0f} MB")
p24 = sorted(glob.glob(f"{DATA}/1t_god/2024/*.csv"))
if p24:
    s = pd.read_csv(p24[0], dtype=str, nrows=300000, low_memory=False)
    s.columns = [c.upper() for c in s.columns]
    log("columns:", list(s.columns))
    for c in ("KATO", "TE", "OKED", "OKED1", "KRP", "SP"):
        if c in s.columns:
            log(f"  {c}: {s[c].nunique()} levels", s[c].value_counts().head(25).to_dict())
    want = ["252101", "251401", "251414", "252203", "251409", "251411", "251412",
            "25140403", "25140405", "25230104", "252301", "252302", "252204", "25220303"]
    if "KCP" in s.columns:
        log("  wanted KCP codes present in sample:", {k: int((s.KCP == k).sum()) for k in want})
        log("  top KCP codes:", s.KCP.value_counts().head(15).to_dict())
    log(s.head(5).to_string())

OUT.close()
print("\nDone -> discovery_report.txt")
