"""
03_digital_index.py - add d002 (quality-of-life survey) variables to the household panel and build the
digital barrier index.   Run AFTER 02_build_core.py.

Run:  python 03_digital_index.py "C:\\...\\synthetic_microdata_2021-2024_XXXX"  [workdir]      (workdir default: work)
Out:  <workdir>/hh_panel2.csv            household x year table + d002 variables + digital barrier index
      <workdir>/check_d002_values.csv    value counts of every d002 column used (to verify the year-by-year mapping)
      <workdir>/check_vopr11_by_age.csv  share of persons with each income line, by age group (verifies GR mapping)
"""
import sys, os
import numpy as np
import pandas as pd

DATA = sys.argv[1]
WORK = sys.argv[2] if len(sys.argv) > 2 else "work"
YEARS = [2021, 2022, 2023, 2024]

# Column names differ by year (form changed in 2022: a heating question and item 1.4 were inserted).
#   2021 : part 2 + 3 live in 'ocenka'; internet questions are GR312..GR315, reason is ONE column (GR315 codes)
#   2022 : part 2 in 'ocenka' (only GR1..GR11 described), part 3 numbered as in the 2022 form
#   2023-24: part 2 moved to 'subject' (GR21...), part 3 in 'ocenka'
MAP = {
    2021: dict(net_use="GR312", home_access="GR314", unexp="GR34", arrears=["GR31_1", "GR31_2", "GR31_3"],
               reason_single="GR315", reason_flags=None, wealth=("ocenka", "GR1")),
    2022: dict(net_use="GR313", home_access="GR315", unexp="GR35", arrears=["GR31_1", "GR31_2", "GR31_3", "GR31_4"],
               reason_single=None, reason_flags=[f"GR316_{i}" for i in range(1, 7)], wealth=("ocenka", "GR1")),
    2023: dict(net_use="GR313", home_access="GR315", unexp="GR35", arrears=["GR31_1", "GR31_2", "GR31_3", "GR31_4"],
               reason_single=None, reason_flags=[f"GR316_{i}" for i in range(1, 7)], wealth=("subject", "GR21")),
    2024: dict(net_use="GR313", home_access="GR315", unexp="GR35", arrears=["GR31_1", "GR31_2", "GR31_3", "GR31_4"],
               reason_single=None, reason_flags=[f"GR316_{i}" for i in range(1, 7)], wealth=("subject", "GR21")),
}


def rd(path, cols=None):
    cols = {c.upper() for c in cols} if cols else None
    df = pd.read_csv(path, dtype=str, low_memory=False,
                     usecols=(lambda c: c.upper() in cols) if cols else None)
    df.columns = [c.upper() for c in df.columns]
    return df


def build_d002(y):
    m = MAP[y]
    oc = rd(f"{DATA}/d002/{y}/ocenka.csv")
    keep = ["NOMER", "POL", "VOZ", m["net_use"], m["home_access"], m["unexp"]] + m["arrears"]
    keep += [m["reason_single"]] if m["reason_single"] else m["reason_flags"]
    if m["wealth"][0] == "ocenka":
        keep.append(m["wealth"][1])
    missing = [c for c in keep if c not in oc.columns]
    if missing:
        print(f"  WARNING {y}: columns missing in ocenka: {missing}")
    out = pd.DataFrame({"NOMER": oc.NOMER, "resp_age": pd.to_numeric(oc.VOZ, errors="coerce"), "resp_sex": oc.POL})
    g = lambda c: oc[c] if c in oc.columns else pd.Series(np.nan, index=oc.index)
    out["net_use"] = g(m["net_use"])
    out["home_access"] = g(m["home_access"])
    out["f_nouse"] = (out.net_use == "2").astype(float).where(out.net_use.isin(["1", "2"]))
    out["f_nohome"] = (out.home_access == "2").astype(float).where(out.home_access.isin(["1", "2"]))
    out["cannot_cover_unexp"] = (g(m["unexp"]) == "2").astype(float).where(g(m["unexp"]).isin(["1", "2"]))
    ar = pd.concat([g(c) for c in m["arrears"]], axis=1)
    out["arrears_any"] = ar.isin(["1", "2"]).any(axis=1).astype(float)
    out["arrears_utilities"] = (g(m["arrears"][1]).isin(["1", "2"])).astype(float)
    out["arrears_credit"] = (g(m["arrears"][2]).isin(["1", "2"])).astype(float)
    # reasons for having no home internet (only asked of those without access)
    if m["reason_single"]:                      # 2021: 1 no provider, 2 too expensive, 3 unaware, 4 mobile instead, 5 other
        r = g(m["reason_single"])
        out["r_provider"] = (r == "1").astype(float); out["r_expensive"] = (r == "2").astype(float)
        out["r_unaware"] = (r == "3").astype(float); out["r_mobile"] = (r == "4").astype(float)
        out["r_devices"] = np.nan
    else:                                       # 2022+: flags 1..6 (5 = not enough devices, 6 = other)
        f = lambda i: (g(f"GR316_{i}") == "1").astype(float)
        out["r_provider"], out["r_expensive"], out["r_unaware"] = f(1), f(2), f(3)
        out["r_mobile"], out["r_devices"] = f(4), f(5)
    for c in ("r_provider", "r_expensive", "r_unaware", "r_mobile", "r_devices"):
        out.loc[out.f_nohome != 1, c] = np.nan   # defined only among households without home internet
    # self-rated wealth 1 (lowest) .. 6 (highest)
    if m["wealth"][0] == "subject":
        su = rd(f"{DATA}/d002/{y}/subject.csv", ["NOMER", m["wealth"][1]]).drop_duplicates("NOMER")
        out = out.merge(su.rename(columns={m["wealth"][1]: "wealth_self"}), on="NOMER", how="left")
    else:
        out["wealth_self"] = g(m["wealth"][1])
    out["wealth_self"] = pd.to_numeric(out.wealth_self, errors="coerce")
    out["year"] = y
    out = out.drop_duplicates("NOMER")
    # value counts for verification
    chk = []
    for c in keep:
        if c in oc.columns:
            vc = oc[c].value_counts(dropna=False)
            chk.append(pd.DataFrame({"year": y, "col": c, "value": vc.index.astype(str), "n": vc.values}))
    return out, pd.concat(chk)


def vopr11_by_age():
    rows = []
    for y in YEARS:
        parts = []
        for q in (1, 2, 3, 4):
            p = f"{DATA}/d004/{y}/{q}kv/kv_vopr11.csv"
            if os.path.exists(p):
                parts.append(rd(p, ["VOZ"] + [f"GR{i}" for i in range(1, 24)]))
        if not parts:
            continue
        t = pd.concat(parts)
        if "VOZ" not in t.columns:
            print(f"  note: no VOZ in vopr11 {y}"); continue
        age = pd.to_numeric(t.VOZ, errors="coerce")
        grp = pd.cut(age, [-1, 17, 24, 44, 59, 64, 120], labels=["<18", "18-24", "25-44", "45-59", "60-64", "65+"])
        for i in range(1, 24):
            nz = pd.to_numeric(t[f"GR{i}"], errors="coerce").fillna(0) > 0
            s = nz.groupby(grp, observed=True).mean()
            for k, v in s.items():
                rows.append(dict(year=y, GR=i, age_group=k, share_nonzero=round(v, 4)))
    return pd.DataFrame(rows)


P = pd.read_csv(f"{WORK}/hh_panel.csv", dtype={"NOMER": str, "TE": str, "K": str, "reg": str})
d2, chks = [], []
for y in YEARS:
    print("d002", y, flush=True)
    a, b = build_d002(y)
    d2.append(a); chks.append(b)
D = pd.concat(d2, ignore_index=True)
P = P.merge(D, on=["NOMER", "year"], how="left")
P["has_d002"] = P.f_nohome.notna().astype(int)

# infrastructure flags from d006 (only where a dwelling record exists)
for c, nm in (("fixed_inet", "f_nofixed"), ("mobile_inet", "f_nomobile")):
    P[nm] = (P[c] == 0).astype(float).where(P.has_dwelling == 1)
flags = ["f_nohome", "f_nouse", "f_nofixed", "f_nomobile"]
P["barrier_n"] = P[flags].notna().sum(axis=1)
P["barrier_idx"] = P[flags].mean(axis=1).where(P.barrier_n >= 3)                  # 0 = no barrier .. 1 = all four
P["digital_excl"] = ((P.f_nohome == 1) & (P.f_nomobile == 1)).astype(float).where(
    P.f_nohome.notna() & P.f_nomobile.notna())                                       # no home internet AND no mobile internet at dwelling

P.to_csv(f"{WORK}/hh_panel2.csv", index=False)
pd.concat(chks).to_csv(f"{WORK}/check_d002_values.csv", index=False)
vopr11_by_age().to_csv(f"{WORK}/check_vopr11_by_age.csv", index=False)

print("\nhouseholds-year:", len(P), " with d002 answers:", int(P.has_d002.sum()))
print(P.groupby("year")[["f_nohome", "f_nouse", "f_nofixed", "f_nomobile", "digital_excl", "barrier_idx",
                          "cannot_cover_unexp", "arrears_any"]].mean().round(3).to_string())
print("\nno home internet, by urban (1) / rural (0):")
print(P.groupby(["year", "urban"]).f_nohome.mean().unstack().round(3).to_string())
print("\nsaved", f"{WORK}/hh_panel2.csv")
