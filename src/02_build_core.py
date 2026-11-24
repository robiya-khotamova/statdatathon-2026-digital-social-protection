"""
02_build_core.py  -  build ONE row per household per year from d008 + d006 + d004.

Run:  python 02_build_core.py "C:\\datathon\\synthetic_microdata_2021-2024_XXXX"  [outdir]
Out:  <outdir>/hh_panel.csv            main analysis table (household x year)
      <outdir>/comm_by_code.csv        communication spending per household-year per item code
      <outdir>/check_*.csv             small tables used to VERIFY assumptions (used to verify assumptions)
Needs: pip install pandas numpy
"""
import sys, os, glob
import numpy as np
import pandas as pd

DATA = sys.argv[1]
OUT = sys.argv[2] if len(sys.argv) > 2 else "work"
os.makedirs(OUT, exist_ok=True)
YEARS, QUARTERS = [2021, 2022, 2023, 2024], [1, 2, 3, 4]

# --- regions: 3 oblasts were created in 2022 (10 Abai, 33 Zhetysu, 62 Ulytau). To compare 2021-2024 we fold
#     them back into their parents. (ASSUMPTION - check_regions.csv lets us verify it.)
REG_FOLD = {"10": "63", "33": "19", "62": "35"}
REG_NAME = {"11": "Акмолинская", "15": "Актюбинская", "19": "Алматинская", "23": "Атырауская",
            "27": "ЗКО", "31": "Жамбылская", "35": "Карагандинская", "39": "Костанайская",
            "43": "Кызылординская", "47": "Мангистауская", "55": "Павлодарская", "59": "СКО",
            "61": "Туркестанская", "63": "ВКО", "71": "Астана", "75": "Алматы (г.)", "79": "Шымкент",
            "10": "Абай", "33": "Жетісу", "62": "Ұлытау"}

# --- vopr11 (income table, one row per person): GR1..GR23 follow the ORDER OF THE FORM, section 3.
#     (ASSUMPTION from the form order - check_income_nonzero.csv shows if the sparsity pattern is plausible.)
INCOME_MAP = {1: "inc_wage", 2: "inc_selfemp", 3: "inc_agri_sales_of_selfemp", 4: "inc_pension",
              5: "inc_scholarship", 6: "inc_asp", 7: "inc_housing_aid", 8: "inc_child_benefit",
              9: "inc_state_social", 10: "inc_special_state", 11: "inc_other_state",
              12: "inc_inkind_social", 13: "inc_cash_aid", 14: "inc_inkind_aid", 15: "inc_other_transfers",
              16: "inc_alimony", 17: "inc_sale_realestate", 18: "inc_sale_property", 19: "inc_property",
              20: "inc_rent_housing", 21: "inc_rent_land", 22: "inc_other_cash", 23: "inc_subsidies"}
SOCIAL_TRANSFERS = ["inc_asp", "inc_housing_aid", "inc_child_benefit", "inc_state_social",
                    "inc_special_state", "inc_other_state", "inc_inkind_social"]
# income used for income groups: everything except one-off asset sales and the "of which" line GR3
INCOME_EXCLUDE = ["inc_agri_sales_of_selfemp", "inc_sale_realestate", "inc_sale_property"]

# --- vopr12 (loans) purposes, form question 2.1: 3 food, 7 health, 8 utility arrears = current consumption
CURRENT_PURPOSE = {"3", "7", "8"}
LOAN_CODES = {"59": "loan_credit_new", "2.1.1.4.1": "loan_credit_repaid", "60": "loan_savings_used",
              "61": "loan_debt_borrowed", "2.1.1.4.2": "loan_debt_repaid", "62": "loan_debt_returned"}  # code book
HIGHCOST_SOURCE = {"1", "8"}          # 1 microfinance, 8 pawnshop (form question 2.2)


def rd(path, cols=None):
    cols = {c.upper() for c in cols} if cols else None
    df = pd.read_csv(path, dtype=str, low_memory=False,
                     usecols=(lambda c: c.upper() in cols) if cols else None)
    df.columns = [c.upper() for c in df.columns]
    return df


def num(s):
    return pd.to_numeric(s, errors="coerce")


def getc(df, c):
    return df[c] if c in df.columns else pd.Series(np.nan, index=df.index)


# ------------------------------------------------------------------ d008 roster -> household frame
def build_roster(y):
    r = rd(f"{DATA}/d008/{y}/kontr_k.csv", ["NOMER", "TE", "K", "NOMP", "RODSTVO", "GOD_ROJD", "POL", "UROV"])
    r["age"] = y - num(getc(r, "GOD_ROJD"))
    r["child"] = (r.age < 18).astype(float).where(r.age.notna())
    r["old"] = (r.age >= 65).astype(float).where(r.age.notna())
    r["adult"] = ((r.age >= 18) & (r.age < 65)).astype(float).where(r.age.notna())
    g = r.groupby("NOMER")
    hh = g.agg(TE=("TE", "first"), K=("K", "first"), size=("NOMP", "size"),
               n_child=("child", "sum"), n_old=("old", "sum"), n_adult=("adult", "sum")).reset_index()
    has_age = g["age"].apply(lambda s: s.notna().all()).reset_index(name="has_age")
    hh = hh.merge(has_age, on="NOMER")
    for c in ["n_child", "n_old", "n_adult"]:
        hh.loc[~hh.has_age, c] = np.nan
    hh["dep_ratio"] = (hh.n_child + hh.n_old) / hh.n_adult.clip(lower=1)
    hh["year"] = y
    return hh


# ------------------------------------------------------------------ d006 dwelling / digital assets
def build_dwelling(y):
    p = sorted(glob.glob(f"{DATA}/d006/{y}/*.csv"))[0]
    d = rd(p, ["NOMER", "TIP_J", "VLAD1", "OB_PL", "KOL_K", "U1", "U3", "U14", "U24",
               "U33", "U34", "U35"])
    out = pd.DataFrame({"NOMER": d.NOMER, "area": num(d.OB_PL), "rooms": num(d.KOL_K), "tip_j": d.TIP_J,
                        "fixed_phone": (d.U33 == "1").astype(int), "fixed_inet": (d.U34 == "1").astype(int),
                        "mobile_inet": (d.U35 == "1").astype(int),
                        "tap_water": (d.U14 == "1").astype(int), "central_sewer": (d.U24 == "1").astype(int),
                        "net_gas": (d.U3 == "1").astype(int)})
    out["year"] = y
    return out


# ------------------------------------------------------------------ d004 quarterly budget
def tab(y, q, n, cols):
    p = f"{DATA}/d004/{y}/{q}kv/kv_vopr{n}.csv"
    return rd(p, cols) if os.path.exists(p) else None


def profile_loans(y):
    """dump columns + small value counts of kv_vopr12 (q1) so the layout of every year can be read by eye"""
    p = f"{DATA}/d004/{y}/1kv/kv_vopr12.csv"
    if not os.path.exists(p):
        return
    df = pd.read_csv(p, dtype=str, low_memory=False)
    with open(f"{OUT}/check_loan_schema_{y}.txt", "w", encoding="utf-8") as f:
        f.write(f"{p}  rows={len(df)}\ncolumns: {list(df.columns)}\n")
        for c in df.columns:
            n = df[c].nunique()
            if n <= 30:
                f.write(f"  {c}: {df[c].value_counts(dropna=False).to_dict()}\n")
            else:
                f.write(f"  {c}: {n} unique, e.g. {df[c].dropna().iloc[:5].tolist()}\n")
        f.write(df.head(12).to_string() + "\n")


def build_d004(y):
    spend, comm, phone, income, loans = [], [], [], [], []
    comm_code, particip = [], []
    for q in QUARTERS:
        t0 = tab(y, q, 0, ["NOMER", "REZ"])
        if t0 is not None:
            particip.append(t0.loc[t0.REZ == "1", ["NOMER"]].assign(q=q))
        for n in (1, 2, 3, 4, 5, 6, 7):                       # every expenditure table
            t = tab(y, q, n, ["NOMER", "KODNU", "STOIMK"])
            if t is None:
                continue
            t["v"] = num(t.STOIMK)
            spend.append(t.groupby("NOMER").v.sum().rename(f"sp_v{n}").reset_index())
            if n == 3:                                          # communication (COICOP 08.3 + 08.1 post)
                comm.append(t.groupby("NOMER").v.sum().rename("comm").reset_index())
                comm_code.append(t.groupby(["NOMER", "KODNU"]).v.sum().reset_index())
            if n == 1:                                          # phone equipment bought (08.2)
                tp = t[t.KODNU.str.startswith("082", na=False)]
                phone.append(tp.groupby("NOMER").v.sum().rename("phone_equip").reset_index())
        t = tab(y, q, 11, ["NOMER"] + [f"GR{i}" for i in INCOME_MAP])
        if t is not None:
            for i, nm in INCOME_MAP.items():
                t[nm] = num(getc(t, f"GR{i}"))
            income.append(t.groupby("NOMER")[list(INCOME_MAP.values())].sum().reset_index())
        t = tab(y, q, 12, ["NOMER", "KODNU", "NOMVNEW", "KOD", "STOIMK"])
        if t is not None:
            t["q"] = q
            loans.append(t)
        if q == 1:                                   # always save the raw layout of the loan table, per year
            profile_loans(y)

    def tot(parts):
        if not parts:
            return pd.DataFrame(columns=["NOMER"])
        return pd.concat(parts).groupby("NOMER").sum(min_count=1).reset_index()

    d = tot(spend)
    for part in (comm, phone, income):
        x = tot(part)
        d = d.merge(x, on="NOMER", how="outer")
    d["n_quarters"] = pd.concat(particip).groupby("NOMER").size().reindex(d.NOMER).values if particip else np.nan

    # loans: purpose rows (NOMVOPR 1211) and source rows (1212) carry the answer in KOD; amount rows are NOMVOPR 200
    L = pd.concat(loans) if loans else None
    if L is not None and "NOMVNEW" not in L.columns:
        print(f"  WARNING {y}: kv_vopr12 has a different layout (no NOMVNEW) - loan fields skipped for {y}; "
              f"see check_loan_schema_{y}.txt", flush=True)
        L = None
    if L is not None:
        pur = L[L.NOMVNEW == "21"]
        src = L[L.NOMVNEW == "22"]
        f = pd.DataFrame({"NOMER": pd.concat([pur.NOMER, src.NOMER]).unique()})
        f["debt_any"] = 1
        cur = pur[pur.KOD.isin(CURRENT_PURPOSE)].NOMER.unique()
        hi = src[src.KOD.isin(HIGHCOST_SOURCE)].NOMER.unique()
        f["debt_current"] = f.NOMER.isin(cur).astype(int)
        f["debt_highcost_src"] = f.NOMER.isin(hi).astype(int)
        amt = L[L.NOMVNEW == "20"].copy()
        amt["v"] = num(amt.STOIMK)
        w = amt.pivot_table(index="NOMER", columns="KODNU", values="v", aggfunc="sum")
        w.columns = [LOAN_CODES.get(str(c).strip(), "loan_other_" + str(c).replace(".", "_")) for c in w.columns]
        d = d.merge(f, on="NOMER", how="left").merge(w.reset_index(), on="NOMER", how="left")
        L.groupby(["NOMVNEW", "KOD"]).size().rename("n").reset_index().assign(year=y).to_csv(
            f"{OUT}/check_loan_kod_{y}.csv", index=False)
        amt.groupby("KODNU").v.agg(["size", "median", "mean"]).reset_index().assign(year=y).to_csv(
            f"{OUT}/check_loan_amount_codes_{y}.csv", index=False)
    for c in ("debt_any", "debt_current", "debt_highcost_src"):
        if c not in d.columns:
            d[c] = 0
        d[c] = d[c].fillna(0).astype(int)
    d["year"] = y
    cc = pd.concat(comm_code).groupby(["NOMER", "KODNU"]).v.sum().reset_index().assign(year=y) if comm_code else None
    return d, cc


# ------------------------------------------------------------------ main
panels, ccodes, inc_chk, reg_chk = [], [], [], []
for y in YEARS:
    print("year", y, flush=True)
    hh = build_roster(y)
    dw = build_dwelling(y)
    d4, cc = build_d004(y)
    p = hh.merge(dw, on=["NOMER", "year"], how="left").merge(d4, on=["NOMER", "year"], how="left")
    p["has_dwelling"] = p.area.notna().astype(int)
    p["has_budget"] = p.n_quarters.notna().astype(int)
    panels.append(p)
    if cc is not None:
        ccodes.append(cc)
    reg_chk.append(hh.groupby("TE").size().rename(y))
    inc_chk.append(pd.DataFrame({"year": y, "col": list(INCOME_MAP.values()),
                                 "share_nonzero": [(p[c].fillna(0) > 0).mean() if c in p else np.nan
                                                   for c in INCOME_MAP.values()],
                                 "mean_nonzero": [p.loc[p[c] > 0, c].mean() if c in p else np.nan
                                                  for c in INCOME_MAP.values()]}))

P = pd.concat(panels, ignore_index=True)
for c in list(INCOME_MAP.values()) + ["comm", "phone_equip"] + [c for c in P.columns if c.startswith("sp_v")]:
    if c in P:
        P[c] = P[c].fillna(0) * P.has_budget          # no budget record -> leave 0 but flagged by has_budget
P["reg"] = P.TE.replace(REG_FOLD)
P["reg_name"] = P.reg.map(REG_NAME)
P["urban"] = (P.K == "1").astype(int)
P["spend_nonfood"] = P[[c for c in P.columns if c.startswith("sp_v")]].sum(axis=1)
P["inc_social"] = P[[c for c in SOCIAL_TRANSFERS if c in P]].sum(axis=1)
P["inc_total"] = P[[c for c in INCOME_MAP.values() if c in P and c not in INCOME_EXCLUDE]].sum(axis=1)
P["recv_social"] = (P.inc_social > 0).astype(int)
P["recv_pension"] = (P.inc_pension > 0).astype(int)
P["inc_pc_q"] = P.inc_total / P["size"] / P.n_quarters.clip(lower=1)     # per person, per quarter, tenge
P.to_csv(f"{OUT}/hh_panel.csv", index=False)
if ccodes:
    pd.concat(ccodes).to_csv(f"{OUT}/comm_by_code.csv", index=False)
pd.concat(inc_chk).to_csv(f"{OUT}/check_income_nonzero.csv", index=False)
pd.concat(reg_chk, axis=1).fillna(0).astype(int).rename_axis("TE").reset_index().to_csv(f"{OUT}/check_regions.csv", index=False)

print("\nrows:", len(P), " households-year with budget:", P.has_budget.sum(), " with dwelling:", P.has_dwelling.sum())
print(P.groupby("year")[["has_budget", "has_dwelling", "recv_social", "recv_pension", "debt_any", "debt_current",
                          "fixed_inet", "mobile_inet"]].mean().round(3).to_string())
print("\nmedian communication spend per household-year (tenge):")
print(P[P.has_budget == 1].groupby(["year", "urban"]).comm.median().unstack().round(0).to_string())
print("\nsaved to", OUT)
