"""
04_analysis.py - all analysis for Tasks 1-6 / Hypotheses 1-6.
Run:  python 04_analysis.py [workdir] [resultsdir]       (defaults: work  results)
Needs: pip install pandas numpy scipy statsmodels scikit-learn matplotlib
Reads: <workdir>/hh_panel2.csv, <workdir>/comm_by_code.csv
Writes: <resultsdir>/tables/*.csv, <resultsdir>/figures/*.png, <resultsdir>/findings.txt
"""
import sys, os, json, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
WORK = sys.argv[1] if len(sys.argv) > 1 else "work"
RES = sys.argv[2] if len(sys.argv) > 2 else "results"
os.makedirs(f"{RES}/tables", exist_ok=True); os.makedirs(f"{RES}/figures", exist_ok=True)
NAVY, TEAL, AMBER, RED, GREY, LIGHT = "#1F3A5F", "#2A9D8F", "#E9A23B", "#C8553D", "#8D99AE", "#D9E2EC"
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.titleweight": "bold", "axes.titlesize": 13, "figure.dpi": 150})
F = []                                   # findings log
def say(s=""):
    print(s); F.append(str(s))
def save(name):
    plt.tight_layout(); plt.savefig(f"{RES}/figures/{name}.png", dpi=170, bbox_inches="tight"); plt.close()
def tab(df, name, **kw):
    df.to_csv(f"{RES}/tables/{name}.csv", **kw)

# ============================================================ data prep
P = pd.read_csv(f"{WORK}/hh_panel2.csv", dtype={"NOMER": str, "TE": str, "K": str, "reg": str})
P = P[P.has_budget == 1].copy()
P["nq"] = P.n_quarters.clip(lower=1)
P["rural"] = 1 - P.urban
P["comm_pc_q"] = P.comm / P["size"] / P.nq                           # tenge per person per quarter
P["inc_nosoc_pc_q"] = (P.inc_total - P.inc_social) / P["size"] / P.nq    # income WITHOUT social transfers
P["svc"] = P[["loan_credit_repaid", "loan_debt_repaid"]].sum(axis=1)   # debt service (tenge, year)
P["svc_inc"] = P.svc / P.inc_total.where(P.inc_total > 0)
qq = lambda s: pd.qcut(s.rank(method="first"), 5, labels=False) + 1
P["q5"] = P.groupby("year").inc_pc_q.transform(qq)
P["q5_pre"] = P.groupby("year").inc_nosoc_pc_q.transform(qq)
P["lsize"] = np.log(P["size"])
QLAB = {1: "Q1\n(беднейшие)", 2: "Q2", 3: "Q3", 4: "Q4", 5: "Q5\n(богатейшие)"}
YEARS = [2021, 2022, 2023, 2024]
say(f"Households-years analysed: {len(P):,}  (years {YEARS}); regions: {P.reg_name.nunique()}")

def logit(formula, data, label):
    try:
        m = smf.logit(formula, data=data).fit(disp=0, maxiter=200)
        return m
    except Exception as e:
        say(f"  model failed ({label}): {e}")
        return None
def orrow(m, term, label):
    b, ci = m.params[term], m.conf_int().loc[term]
    return dict(model=label, term=term, OR=np.exp(b), lo=np.exp(ci[0]), hi=np.exp(ci[1]), p=m.pvalues[term], n=int(m.nobs))

# ============================================================ TASK 1 / H1
say("\n" + "=" * 70 + "\nTASK 1 / H1  Communication spending by income and urban/rural\n" + "=" * 70)
g = (P.groupby(["year", "q5", "urban"])
       .apply(lambda d: pd.Series(dict(n=len(d), comm_pc_median=d.comm_pc_q.median(),
                                       share_of_income=d.comm.sum() / d.inc_total.sum(),
                                       share_of_nonfood=d.comm.sum() / d.spend_nonfood.sum())))
       .reset_index())
tab(g, "T1_comm_by_quintile_urban", index=False)
d24 = P[P.year == 2024]
q1, q5 = d24[d24.q5 == 1], d24[d24.q5 == 5]
say(f"2024 median communication spend per person per quarter: Q1 = {q1.comm_pc_q.median():,.0f} tg, Q5 = {q5.comm_pc_q.median():,.0f} tg "
    f"(ratio {q5.comm_pc_q.median() / q1.comm_pc_q.median():.1f}x)")
sh = P.groupby(["year", "q5"]).apply(lambda d: d.comm.sum() / d.inc_total.sum()).unstack()
say("Share of household income spent on communication (ratio of sums), by year x quintile:\n" + (sh * 100).round(2).to_string())
u = P.groupby(["year", "urban"]).comm_pc_q.median().unstack()
say("Median communication spend per person per quarter, rural (0) vs urban (1):\n" + u.round(0).to_string())
pos = P[(P.comm_pc_q > 0) & (P.inc_pc_q > 0)].copy()
pos["lc"], pos["li"] = np.log(pos.comm_pc_q), np.log(pos.inc_pc_q)
ols = smf.ols("lc ~ li + rural + lsize + C(year) + C(reg)", data=pos).fit(cov_type="HC1")
el, ci = ols.params["li"], ols.conf_int().loc["li"]
say(f"Income elasticity of communication spending = {el:.2f} [{ci[0]:.2f}; {ci[1]:.2f}]  (<1 => poorer households spend a LARGER share of income)")
say(f"Rural vs urban at equal income: {100 * (np.exp(ols.params['rural']) - 1):+.1f}% communication spend per person (p={ols.pvalues['rural']:.3f})")
mw = stats.mannwhitneyu(P[P.q5 == 1].comm_pc_q, P[P.q5 == 5].comm_pc_q)
say(f"Mann-Whitney Q1 vs Q5 (all years): p={mw.pvalue:.2e}")
H1 = dict(elasticity=el, ratio_q5_q1_2024=q5.comm_pc_q.median() / q1.comm_pc_q.median())

# fig 1
fig, ax = plt.subplots(1, 2, figsize=(12, 4.8))
w = 0.38; x = np.arange(1, 6)
for k, (lab, col) in enumerate([("Село", AMBER), ("Город", NAVY)]):
    v = d24[d24.urban == k].groupby("q5").comm_pc_q.median().reindex(x)
    ax[0].bar(x + (k - .5) * w, v / 1000, w, label=lab, color=col)
ax[0].set_xticks(x); ax[0].set_xticklabels([QLAB[i] for i in x]); ax[0].set_ylabel("тыс. тенге на человека в квартал")
ax[0].set_title("Расходы на связь и интернет, 2024"); ax[0].legend(frameon=False)
for yr, col in zip(YEARS, [GREY, LIGHT, TEAL, NAVY]):
    ax[1].plot(x, sh.loc[yr].values * 100, marker="o", color=col, lw=2.2, label=str(yr))
ax[1].set_xticks(x); ax[1].set_xticklabels([QLAB[i] for i in x]); ax[1].set_ylabel("% дохода домохозяйства")
ax[1].set_title("Доля дохода, уходящая на связь"); ax[1].legend(frameon=False, ncol=2)
save("F1_comm_spending")

# composition
cc = pd.read_csv(f"{WORK}/comm_by_code.csv", dtype={"NOMER": str, "KODNU": str})
GRP = {"08300014": "Мобильный интернет", "08300008": "Сотовая связь", "08300004": "Домашний (фиксированный) интернет",
       "08300006": "ТВ (кабельное/спутниковое)", "08300011": "ТВ (кабельное/спутниковое)", "08300001": "Стационарный телефон",
       "08300002": "Стационарный телефон", "08300013": "Стационарный телефон", "08300012": "Стационарный телефон",
       "081000": "Почта и курьеры", "08100003": "Почта и курьеры"}
cc["grp"] = cc.KODNU.map(GRP).fillna("Прочее")
cc = cc.merge(P[["NOMER", "year", "q5", "urban"]], on=["NOMER", "year"])
comp = cc.groupby(["year", "q5", "urban", "grp"]).v.sum().reset_index()
comp["share"] = comp.v / comp.groupby(["year", "q5", "urban"]).v.transform("sum")
tab(comp, "T1_comm_composition", index=False)
fixed_pay = (cc[cc.KODNU == "08300004"].groupby(["year", "q5", "urban"]).NOMER.nunique() /
             P.groupby(["year", "q5", "urban"]).size()).rename("share_paying_fixed_internet").reset_index()
tab(fixed_pay, "T1_fixed_internet_payers", index=False)
fp = fixed_pay[fixed_pay.year == 2024].pivot(index="q5", columns="urban", values="share_paying_fixed_internet")
say("2024 share of households paying for FIXED home internet (rural 0 / urban 1):\n" + (fp * 100).round(1).to_string())
order = ["Мобильный интернет", "Сотовая связь", "Домашний (фиксированный) интернет", "ТВ (кабельное/спутниковое)",
         "Стационарный телефон", "Почта и курьеры"]
cols = [NAVY, TEAL, AMBER, RED, GREY, LIGHT]
fig, ax = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
for k, ttl in enumerate(["Село", "Город"]):
    c = comp[(comp.year == 2024) & (comp.urban == k)].pivot(index="q5", columns="grp", values="share").reindex(columns=order).fillna(0)
    bot = np.zeros(5)
    for o, col in zip(order, cols):
        ax[k].bar(range(1, 6), c[o].values * 100, bottom=bot, color=col, label=o); bot += c[o].values * 100
    ax[k].set_title(f"{ttl}: структура расходов на связь, 2024"); ax[k].set_xticks(range(1, 6))
    ax[k].set_xticklabels([QLAB[i] for i in range(1, 6)])
ax[0].set_ylabel("% расходов на связь"); ax[1].legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(.5, -.18), ncol=3)
save("F2_comm_structure")

# ============================================================ TASK 2 / H3, H4
say("\n" + "=" * 70 + "\nTASK 2 / H3-H4  Social transfers and debt\n" + "=" * 70)
rows = []
for yr in YEARS:
    d = P[P.year == yr]
    for r, lab in [(1, "recipient"), (0, "non-recipient")]:
        s = d[d.recv_social == r]
        rows.append(dict(year=yr, group=lab, n=len(s), debt_any=s.debt_any.mean(), debt_current=s.debt_current.mean(),
                         debt_highcost=s.debt_highcost_src.mean(), arrears_any=s.arrears_any.mean(),
                         cannot_cover=s.cannot_cover_unexp.mean(), mean_debt_service_share=s.svc_inc.replace(np.inf, np.nan).mean()))
T2 = pd.DataFrame(rows); tab(T2, "T2_debt_by_recipient", index=False)
say("Debt incidence by receipt of social transfers (GR6-GR12; pensions excluded):\n" + T2.round(3).to_string(index=False))
fm = "{y} ~ recv_social + C(q5_pre) + lsize + rural + C(year) + C(reg)"
res = []
for y, lab in [("debt_current", "Долг на текущее потребление"), ("debt_any", "Любой кредит/долг"), ("debt_highcost_src", "Займ в МФО/ломбарде")]:
    m = logit(fm.format(y=y), P, y)
    if m is not None:
        r = orrow(m, "recv_social", lab); res.append(r)
        say(f"Adjusted OR for {y}: {r['OR']:.2f} [{r['lo']:.2f}; {r['hi']:.2f}] p={r['p']:.3f}")
sub = P[P.year >= 2022]
for y, lab in [("arrears_any", "Просрочки платежей")]:
    m = logit(fm.format(y=y), sub.dropna(subset=[y]), y)
    if m is not None:
        r = orrow(m, "recv_social", lab); res.append(r)
        say(f"Adjusted OR for {y} (2022-24): {r['OR']:.2f} [{r['lo']:.2f}; {r['hi']:.2f}] p={r['p']:.3f}")
OR2 = pd.DataFrame(res); tab(OR2, "T2_adjusted_odds_ratios", index=False)
# by transfer line
for line, lab in [("inc_child_benefit", "детские пособия"), ("inc_state_social", "гос. соц. пособие"), ("inc_special_state", "спец. гос. пособие")]:
    P["flag"] = (P[line] > 0).astype(int)
    m = logit("debt_current ~ flag + C(q5_pre) + lsize + rural + C(year) + C(reg)", P, line)
    if m is not None:
        r = orrow(m, "flag", lab); say(f"  debt_current OR for {lab}: {r['OR']:.2f} [{r['lo']:.2f}; {r['hi']:.2f}] p={r['p']:.3f}")

# H4: debt service vs transfers
R = P[(P.recv_social == 1) & (P.inc_social > 0)].copy()
R["svc_tr"] = R.svc / R.inc_social
R["leak"] = np.minimum(R.svc, R.inc_social)
by = R.groupby("q5_pre").apply(lambda d: pd.Series(dict(
    n=len(d), share_servicing=(d.svc > 0).mean(),
    median_svc_to_transfer_if_servicing=d[d.svc > 0].svc_tr.median(),
    share_svc_ge_transfer=(d.svc >= d.inc_social).mean(),
    aggregate_offset=d.leak.sum() / d.inc_social.sum()))).reset_index()
tab(by, "T2_debt_service_vs_transfers", index=False)
say("\nH4 - debt service relative to social transfers among recipients, by pre-transfer income quintile:\n" + by.round(3).to_string(index=False))
say(f"Overall: {(R.svc > 0).mean():.1%} of recipient households pay on credit/debts; for them the median annual debt service equals "
    f"{R[R.svc > 0].svc_tr.median():.0%} of the transfers received; capped aggregate offset = {R.leak.sum() / R.inc_social.sum():.1%} of all transfer money.")
H4 = dict(share_servicing=(R.svc > 0).mean(), median_ratio=R[R.svc > 0].svc_tr.median(), offset=R.leak.sum() / R.inc_social.sum())
fig, ax = plt.subplots(1, 2, figsize=(12, 4.8))
x = np.arange(len(YEARS)); w = .38
for k, (lab, col) in enumerate([("не получают пособий", GREY), ("получают пособия", TEAL)]):
    v = T2[T2.group == ["non-recipient", "recipient"][k]].debt_current.values * 100
    ax[0].bar(x + (k - .5) * w, v, w, color=col, label=lab)
    for xi, vi in zip(x + (k - .5) * w, v): ax[0].text(xi, vi + .05, f"{vi:.1f}", ha="center", fontsize=9)
ax[0].set_xticks(x); ax[0].set_xticklabels(YEARS); ax[0].set_ylabel("% домохозяйств")
ax[0].set_title("Заём на текущее потребление\n(еда, лекарства, коммунальные долги)"); ax[0].legend(frameon=False)
ax[1].bar(by.q5_pre, by.aggregate_offset * 100, color=[RED, AMBER, AMBER, TEAL, TEAL], width=.6)
for xi, vi in zip(by.q5_pre, by.aggregate_offset * 100): ax[1].text(xi, vi + .3, f"{vi:.0f}%", ha="center")
ax[1].set_xticks(range(1, 6)); ax[1].set_xticklabels([QLAB[i] for i in range(1, 6)])
ax[1].set_title("Какая часть пособий «уходит» на обслуживание долгов"); ax[1].set_ylabel("% суммы пособий (потолок = размер пособия)")
save("F3_transfers_debt")

# ============================================================ TASK 3 / H2
say("\n" + "=" * 70 + "\nTASK 3 / H2  Regional and infrastructure differences in digital access\n" + "=" * 70)
reg = P.groupby(["reg_name", "year"]).agg(n=("NOMER", "size"), barrier_idx=("barrier_idx", "mean"), no_home_internet=("f_nohome", "mean"),
                                           no_mobile_signal=("f_nomobile", "mean"), no_fixed=("f_nofixed", "mean"), digital_excl=("digital_excl", "mean")).reset_index()
tab(reg, "T3_regions", index=False)
r24 = reg[reg.year == 2024].sort_values("barrier_idx", ascending=False)
say("2024 digital barrier index (0-1) - five highest regions:\n" + r24.head(5)[["reg_name", "barrier_idx", "no_home_internet", "no_mobile_signal"]].round(3).to_string(index=False))
say("five lowest:\n" + r24.tail(5)[["reg_name", "barrier_idx", "no_home_internet", "no_mobile_signal"]].round(3).to_string(index=False))
hm = reg.pivot(index="reg_name", columns="year", values="barrier_idx"); hm = hm.sort_values(2024)
fig, ax = plt.subplots(figsize=(7.5, 7))
im = ax.imshow(hm.values, cmap="YlOrRd", aspect="auto"); ax.set_xticks(range(4)); ax.set_xticklabels(hm.columns)
ax.set_yticks(range(len(hm))); ax.set_yticklabels(hm.index)
for i in range(hm.shape[0]):
    for j in range(hm.shape[1]): ax.text(j, i, f"{hm.values[i, j]:.2f}", ha="center", va="center", fontsize=8)
ax.set_title("Индекс цифрового барьера по регионам (0–1)"); plt.colorbar(im, shrink=.7); ax.spines[:].set_visible(False)
save("F5_region_heatmap")
d = P[P.year >= 2022]
ur = d.groupby(["q5", "urban"]).agg(no_home=("f_nohome", "mean"), idx=("barrier_idx", "mean"), no_mob=("f_nomobile", "mean")).reset_index()
tab(ur, "T3_urban_rural_by_quintile", index=False)
say("No home internet, rural vs urban by income quintile (2022-24):\n" + ur.pivot(index="q5", columns="urban", values="no_home").round(3).to_string())
for dep in ["f_nohome", "barrier_idx"]:
    for fe in ["", " + C(reg)"]:
        if dep == "f_nohome":
            m = logit(f"f_nohome ~ rural + C(q5) + lsize + C(year){fe}", P.dropna(subset=["f_nohome"]), dep)
            if m is not None:
                r = orrow(m, "rural", dep); H2 = r; say(f"  adjusted OR rural vs urban, {dep}{' (region FE)' if fe else ''}: {r['OR']:.2f} [{r['lo']:.2f}; {r['hi']:.2f}] p={r['p']:.4f}")
        else:
            m = smf.ols(f"barrier_idx ~ rural + C(q5) + lsize + C(year){fe}", data=P.dropna(subset=["barrier_idx"])).fit(cov_type="HC1")
            say(f"  adjusted gap rural-urban in barrier index{' (region FE)' if fe else ''}: {m.params['rural']:+.3f} (p={m.pvalues['rural']:.4f})")
fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
for k, (lab, col) in enumerate([("Село", AMBER), ("Город", NAVY)]):
    v = ur[ur.urban == k].set_index("q5").no_home
    ax[0].bar(np.arange(1, 6) + (k - .5) * .38, v.values * 100, .38, color=col, label=lab)
ax[0].set_xticks(range(1, 6)); ax[0].set_xticklabels([QLAB[i] for i in range(1, 6)]); ax[0].set_ylabel("% домохозяйств без домашнего интернета")
ax[0].set_title("Нет домашнего интернета: село vs город\nпри одинаковом уровне дохода (2022–24)"); ax[0].legend(frameon=False)
rs = d[d.f_nohome == 1].groupby("urban")[["r_provider", "r_expensive", "r_unaware", "r_mobile", "r_devices"]].mean()
labs = ["нет провайдера", "слишком дорого", "не знают об этом", "хватает мобильного", "не хватает устройств"]
y0 = np.arange(5)
for k, (lab, col) in enumerate([("Село", AMBER), ("Город", NAVY)]):
    ax[1].barh(y0 + (k - .5) * .38, rs.loc[k].values * 100, .38, color=col, label=lab)
ax[1].set_yticks(y0); ax[1].set_yticklabels(labs); ax[1].invert_yaxis(); ax[1].set_xlabel("% среди домохозяйств без домашнего интернета")
ax[1].set_title("Почему нет домашнего интернета"); ax[1].legend(frameon=False)
save("F6_urban_rural_digital")
tab(rs.reset_index(), "T3_reasons_no_home_internet", index=False)
say("Reasons for no home internet (% of households without it, 2022-24; rural 0 / urban 1):\n" + (rs * 100).round(1).T.to_string())

# ============================================================ TASK 4
say("\n" + "=" * 70 + "\nTASK 4  Take-up gap (eligible-looking households without any social transfer)\n" + "=" * 70)
P["poor_line"] = P.groupby("year").inc_nosoc_pc_q.transform("median") * 0.6
P["low_inc"] = (P.inc_nosoc_pc_q < P.poor_line).astype(int)
P["barrier_lvl"] = pd.cut(P.barrier_idx, [-.01, .2499, .4999, 1.0], labels=["низкий", "средний", "высокий"])
L = P[P.low_inc == 1]
say(f"Low-income households (pre-transfer income per person < 60% of the year median): {len(L):,} ({len(L) / len(P):.1%})")
t4 = L.groupby(["year", "barrier_lvl"]).agg(n=("NOMER", "size"), no_social=("recv_social", lambda s: 1 - s.mean())).reset_index()
tab(t4, "T4_takeup_gap_by_barrier", index=False)
say("Share of low-income households receiving NO social transfer, by digital barrier level:\n" + t4.pivot(index="year", columns="barrier_lvl", values="no_social").round(3).to_string())
L2 = L.dropna(subset=["barrier_idx"]).copy(); L2["no_social"] = 1 - L2.recv_social; L2["high_b"] = (L2.barrier_idx >= .5).astype(int)
m = logit("no_social ~ high_b + lsize + rural + C(year) + C(reg)", L2, "gap")
if m is not None:
    r = orrow(m, "high_b", "gap"); say(f"Adjusted OR of receiving nothing, high vs low/mid barrier: {r['OR']:.2f} [{r['lo']:.2f}; {r['hi']:.2f}] p={r['p']:.3f}")
    H4b = r
LC = P[(P.year >= 2022) & (P.low_inc == 1) & (P.n_child > 0)].copy(); LC["no_child_benefit"] = (LC.inc_child_benefit <= 0).astype(int)
tc = LC.groupby("barrier_lvl").agg(n=("NOMER", "size"), gap=("no_child_benefit", "mean")).reset_index()
tab(tc, "T4_child_benefit_gap", index=False)
say("Low-income households WITH CHILDREN and no child benefit (2022-24):\n" + tc.round(3).to_string(index=False))
rg = L.groupby("reg_name").agg(n=("NOMER", "size"), gap=("recv_social", lambda s: 1 - s.mean())).reset_index().sort_values("gap", ascending=False)
tab(rg, "T4_gap_by_region", index=False)
fig, ax = plt.subplots(1, 2, figsize=(12, 4.8), gridspec_kw={"width_ratios": [1, 1.15]})
pv = t4.pivot(index="year", columns="barrier_lvl", values="no_social") * 100
for c, col in zip(["низкий", "средний", "высокий"], [TEAL, AMBER, RED]):
    ax[0].plot(pv.index, pv[c], marker="o", lw=2.4, color=col, label=f"барьер: {c}")
ax[0].set_xticks(YEARS); ax[0].set_ylabel("% малообеспеченных без соц. выплат"); ax[0].legend(frameon=False)
ax[0].set_title("«Недополучение» помощи (take-up gap)")
rg2 = rg.sort_values("gap"); ax[1].barh(rg2.reg_name, rg2.gap * 100, color=[RED if v >= rg.gap.quantile(.75) else LIGHT for v in rg2.gap])
ax[1].set_title("Доля малообеспеченных без выплат, по регионам"); ax[1].set_xlabel("%")
save("F8_takeup_gap")

# ============================================================ TASK 5 / H5
say("\n" + "=" * 70 + "\nTASK 5 / H5  Double vulnerability (debt stress + digital exclusion)\n" + "=" * 70)
P["debt_stress"] = ((P.svc_inc >= .20) | (P.arrears_credit == 1) | (P.debt_current == 1)).astype(int)
P["high_barrier"] = (P.barrier_idx >= .5).astype(float).where(P.barrier_idx.notna())
P["double"] = ((P.debt_stress == 1) & (P.high_barrier == 1)).astype(float).where(P.high_barrier.notna())
pr = P.groupby("year").agg(high_barrier=("high_barrier", "mean"), debt_stress=("debt_stress", "mean"), double=("double", "mean"), n_double=("double", "sum")).reset_index()
tab(pr, "T5_prevalence", index=False)
say("Prevalence by year:\n" + pr.round(4).to_string(index=False))
a = P.groupby("q5").double.mean()
say("Share in double vulnerability by income quintile (pooled):\n" + (a * 100).round(2).to_string())
exp = P.high_barrier.mean() * P.debt_stress.mean()
say(f"Observed co-occurrence {P.double.mean():.3%} vs expected if independent {exp:.3%} (ratio {P.double.mean() / exp:.2f})")
chi = stats.chi2_contingency(pd.crosstab(P.high_barrier, P.debt_stress)); say(f"chi-square independence test p={chi[1]:.3g}")
# H5: zone of digital invisibility
Q = P[P.year >= 2022].copy()
Q["dep_hi"] = (Q.dep_ratio >= Q.dep_ratio.median()).astype(int)
Q["invisible"] = ((Q.q5_pre <= 2) & (Q.dep_hi == 1) & (Q.barrier_idx >= .5)).astype(float).where(Q.barrier_idx.notna())
zi = Q.groupby("year").invisible.mean()
say("H5 'zone of digital invisibility' (bottom-2 pre-transfer quintiles + above-median dependency + barrier index >= 0.5), share of households:\n" + (zi * 100).round(2).to_string())
zr = Q.groupby("reg_name").invisible.mean().sort_values(ascending=False); tab(zr.reset_index(), "T5_invisibility_by_region", index=False)
say("Top-5 regions by 'invisibility' share:\n" + (zr.head(5) * 100).round(2).to_string())
# clustering
D = Q[Q.double == 1].copy()
feat = pd.DataFrame({"доход_на_чел_(log)": np.log(D.inc_pc_q.clip(lower=1)), "размер": D["size"],
                     "доля_детей": D.n_child / D["size"], "доля_пожилых": D.n_old / D["size"], "село": D.rural,
                     "получают_пособия": D.recv_social, "долг_к_доходу": D.svc_inc.clip(upper=1), "индекс_барьера": D.barrier_idx})
ok = feat.notna().all(axis=1); D, feat = D[ok], feat[ok]
Z = StandardScaler().fit_transform(feat)
K = 4 if len(D) >= 200 else 3
km = KMeans(K, n_init=20, random_state=0).fit(Z); D["cluster"] = km.labels_
prof = feat.assign(cluster=km.labels_).groupby("cluster").mean()
prof["n"] = D.groupby("cluster").size(); prof["доля_%"] = prof.n / prof.n.sum() * 100
extra = D.groupby("cluster").agg(top_region=("reg_name", lambda s: s.value_counts().index[0]), q1_q2_share=("q5", lambda s: (s <= 2).mean()),
                                  cannot_cover=("cannot_cover_unexp", "mean"), no_mobile=("f_nomobile", "mean"), no_home=("f_nohome", "mean"))
prof = prof.join(extra); tab(prof.reset_index(), "T5_cluster_profiles", index=False)
say(f"\nDouble-vulnerable households 2022-24 used for profiling: {len(D)}; clusters = {K}\n" + prof.round(2).to_string())
zprof = pd.DataFrame(Z, columns=feat.columns).assign(cluster=km.labels_).groupby("cluster").mean()
fig, ax = plt.subplots(1, 2, figsize=(13, 4.8), gridspec_kw={"width_ratios": [1, 1.25]})
ax[0].plot(pr.year, pr.high_barrier * 100, marker="o", lw=2.4, color=AMBER, label="высокий цифровой барьер")
ax[0].plot(pr.year, pr.debt_stress * 100, marker="o", lw=2.4, color=GREY, label="долговая нагрузка")
ax[0].plot(pr.year, pr.double * 100, marker="o", lw=2.8, color=RED, label="двойная уязвимость")
ax[0].set_xticks(YEARS); ax[0].set_ylabel("% домохозяйств"); ax[0].legend(frameon=False); ax[0].set_title("Масштаб уязвимости")
im = ax[1].imshow(zprof.values, cmap="RdBu_r", vmin=-2, vmax=2, aspect="auto"); ax[1].set_xticks(range(zprof.shape[1]))
ax[1].set_xticklabels(zprof.columns, rotation=40, ha="right", fontsize=9)
ax[1].set_yticks(range(K)); ax[1].set_yticklabels([f"Профиль {i + 1} (n={int(prof.n[i])})" for i in range(K)])
for i in range(K):
    for j in range(zprof.shape[1]): ax[1].text(j, i, f"{zprof.values[i, j]:+.1f}", ha="center", va="center", fontsize=8)
ax[1].set_title("Профили зоны двойной уязвимости\n(отклонение от среднего, в σ)"); ax[1].spines[:].set_visible(False)
save("F9_double_vulnerability")

# ============================================================ TASK 6 / H6
say("\n" + "=" * 70 + "\nTASK 6 / H6  Channel sizing (scenario, not a causal test)\n" + "=" * 70)
def tier(r):
    if pd.isna(r.f_nohome) or pd.isna(r.f_nomobile): return np.nan
    if r.f_nohome == 0 and r.f_nouse != 1: return "A: цифровой канал"
    if r.f_nomobile == 0 and r.f_nouse != 1: return "B: SMS + бесплатный мобильный (zero-rating)"
    return "C: офлайн / с помощником"
P["tier"] = P.apply(tier, axis=1)
grp = {"Все домохозяйства": P, "Малообеспеченные": P[P.low_inc == 1], "Село": P[P.rural == 1],
       "Долговая нагрузка": P[P.debt_stress == 1], "Двойная уязвимость": P[P.double == 1]}
T6 = pd.DataFrame({k: v.tier.value_counts(normalize=True) for k, v in grp.items()}).T.fillna(0)
tab(T6.reset_index(), "T6_channel_tiers", index=False)
say("Share of households by best available channel tier (pooled 2021-24):\n" + (T6 * 100).round(1).to_string())
dv = P[P.double == 1]
say(f"Among double-vulnerable households: {1 - dv.f_nomobile.mean():.1%} have mobile internet coverage at the dwelling (SMS/zero-rating reachable); "
    f"{dv.r_expensive.mean():.1%} (of those without home internet) say home internet is too expensive.")
H6 = dict(sms_reach_double=1 - dv.f_nomobile.mean(), tierC_all=T6.loc["Все домохозяйства"].get("C: офлайн / с помощником", np.nan))
fig, ax = plt.subplots(figsize=(10.5, 4.4)); left = np.zeros(len(T6))
for t, col in zip(sorted(T6.columns), [TEAL, AMBER, RED]):
    ax.barh(T6.index, T6[t] * 100, left=left, color=col, label=t)
    for i, v in enumerate(T6[t] * 100):
        if v > 4: ax.text(left[i] + v / 2, i, f"{v:.0f}%", ha="center", va="center", color="white", fontweight="bold")
    left += T6[t].values * 100
ax.invert_yaxis(); ax.set_xlabel("% домохозяйств"); ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(.5, -.15), ncol=3, fontsize=9)
ax.set_title("Какой канал доставки соцпомощи доступен домохозяйству"); ax.set_xlim(0, 100)
save("F11_channel_tiers")

# ============================================================ verdicts
say("\n" + "=" * 70 + "\nHYPOTHESIS SCORECARD\n" + "=" * 70)
def orv(label):
    x = OR2[OR2.model == label]
    return x.iloc[0] if len(x) else None
def verdict(ok, partial=False): return "SUPPORTED" if ok else ("PARTLY" if partial else "NOT SUPPORTED")
sc = []
sc.append(("H1 poorest spend less on communication", verdict(H1["ratio_q5_q1_2024"] > 1.2),
           f"Q5/Q1 median spend {H1['ratio_q5_q1_2024']:.1f}x; income elasticity {H1['elasticity']:.2f} (<1 = poor spend a larger SHARE of income)"))
sc.append(("H2 rural households face higher digital barrier", verdict(H2["lo"] > 1),
           f"adjusted OR no home internet, rural vs urban = {H2['OR']:.2f} [{H2['lo']:.2f}; {H2['hi']:.2f}]"))
o = orv("Долг на текущее потребление")
if o is not None:
    sc.append(("H3 transfer recipients borrow more for current needs", verdict(o.lo > 1, o.OR > 1),
               f"adjusted OR = {o.OR:.2f} [{o.lo:.2f}; {o.hi:.2f}], p={o.p:.3f}"))
sc.append(("H4 debt service erodes transfers", verdict(H4["offset"] >= .10, H4["offset"] >= .03),
           f"{H4['share_servicing']:.0%} of recipients service debt; capped offset = {H4['offset']:.1%} of transfer money (weak test: repayment vs transfer, same year)"))
sc.append(("H5 low income + dependency + no connectivity overlap", verdict(P.double.mean() / exp > 1.2, P.double.mean() / exp > 1),
           f"double-vulnerable {P.double.mean():.1%}; co-occurrence ratio vs independence {P.double.mean() / exp:.2f}; invisibility zone {zi.iloc[-1]:.1%} (2024)"))
sc.append(("H6 SMS / zero-rating can reach the vulnerable", verdict(H6["sms_reach_double"] >= .5, H6["sms_reach_double"] >= .3),
           f"{H6['sms_reach_double']:.0%} of double-vulnerable households have mobile coverage (scenario estimate, not a causal test)"))
for h, v, t in sc: say(f"{v:>14} | {h}\n{'':>14}   {t}")
tab(pd.DataFrame(sc, columns=["hypothesis", "verdict", "evidence"]), "scorecard", index=False)
say("\nCAVEAT: synthetic microdata keeps roughly half of real-world associations -> treat effect sizes as LOWER BOUNDS.")
json.dump({k: (float(v) if np.isscalar(v) else str(v)) for k, v in {**H1, **H4, **H6}.items()}, open(f"{RES}/key_numbers.json", "w"), indent=1)
P.to_csv(f"{RES}/tables/analysis_panel.csv", index=False)
open(f"{RES}/findings.txt", "w", encoding="utf-8").write("\n".join(F))
print("\nDone. See", RES)
