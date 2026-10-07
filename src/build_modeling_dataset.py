"""
Build the modelling dataset matching the group concept note:

  Target:   At risk (qualified/adverse/disclaimer/outstanding) vs
            Not at risk (unqualified, with or without findings)
            for year t+1
  Features: financial ratios for year t, plus audit outcome up to t
  Horizon:  one audit cycle (predict t+1 using only t)

HOW TO RUN: save this file as  src/build_modeling_dataset.py  in your project,
then from Command Prompt:
    cd C:\\Users\\chaba\\Documents\\AdvDip\\IDA117V\\IDA117V-Audit-Risk-Project
    python src\\build_modeling_dataset.py

This version finds data/raw and data/processed relative to WHERE THIS FILE LIVES,
not relative to your current folder — so it works the same whether you run it
from Command Prompt, from a notebook, or from any other working directory.

NOTE: uses "Audited Actual" figures for year t, which is NOT leakage here,
since year t's audit is already finished by the time year t+1 is being predicted.
"""

from pathlib import Path
import pandas as pd
import requests

# ---- Paths anchored to this file's own location, not the current working directory ----
SCRIPT_DIR = Path(__file__).resolve().parent      # .../project/src
PROJECT_ROOT = SCRIPT_DIR.parent                  # .../project
RAW = PROJECT_ROOT / "data" / "raw"
OUT = PROJECT_ROOT / "data" / "processed"
RAW.mkdir(parents=True, exist_ok=True)
OUT.mkdir(parents=True, exist_ok=True)

AUDIT_FILE = RAW / "audit_opinion_facts.csv"

BASE = "https://municipaldata.treasury.gov.za/api/cubes"
YEARS = range(2018, 2026)
IDX = ["demarcation.code", "financial_year_end.year"]
DRILL = "demarcation.code|demarcation.label|financial_year_end.year|item.code|item.label"
# grants_v2 names its item dimension "grant", not "item"
DRILL_GRANTS = "demarcation.code|demarcation.label|financial_year_end.year|grant.code|grant.label"

NOT_AT_RISK = {"unqualified", "unqualified_emphasis_of_matter"}
AT_RISK = {"qualified", "adverse", "disclaimer", "outstanding"}


def fetch_aggregate(cube, aggregates, cut, pagesize=10000, drilldown=DRILL):
    rows, page = [], 0
    while True:
        params = {"drilldown": drilldown, "aggregates": aggregates,
                  "cut": cut, "pagesize": pagesize, "page": page}
        r = requests.get(f"{BASE}/{cube}/aggregate", params=params, timeout=120)
        if r.status_code != 200:
            print(f"    ERROR {r.status_code} on {cube} (cut={cut}): {r.text[:300]}")
            break
        data = r.json()
        rows.extend(data["cells"])
        if len(rows) >= data["total_cell_count"] or not data["cells"]:
            break
        page += 1
    return pd.DataFrame(rows)


def fetch_cube_all_years(cube, aggregates, amount_type="Audited Actual", drilldown=DRILL):
    frames = []
    for y in YEARS:
        cut = f'financial_year_end.year:{y}'
        if amount_type:
            cut += f'|amount_type.label:"{amount_type}"'
        frames.append(fetch_aggregate(cube, aggregates, cut, drilldown=drilldown))
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return df


def sum_by_code_range(df, lo, hi, reference_index=None):
    """
    Sum amount.sum for rows whose item.code (numeric) falls in [lo, hi).
    Used instead of a single precomputed 'Total' row, since that row isn't
    consistently backfilled for older years, while the component line items are.
    """
    if df.empty:
        return pd.Series(0.0, index=reference_index) if reference_index is not None else pd.Series(dtype=float)
    codes = pd.to_numeric(df["item.code"], errors="coerce")
    mask = codes.notna() & (codes >= lo) & (codes < hi)
    result = df[mask].groupby(IDX)["amount.sum"].sum()
    if reference_index is not None:
        result = result.reindex(reference_index, fill_value=0.0)
    return result


def find_item_total(df, keywords, code_prefix=None, reference_index=None):
    """
    Sum amount.sum for rows whose item.label contains ALL given keywords
    (case-insensitive), grouped by municipality-year.
    Missing items return 0 (aligned to reference_index) instead of crashing
    later arithmetic.
    """
    def zeros():
        if reference_index is not None:
            return pd.Series(0.0, index=reference_index)
        return pd.Series(dtype=float)

    if df.empty or "item.label" not in df.columns:
        print(f"    keywords {keywords}: cube empty or has no item.label column, using 0")
        return zeros()

    mask = df["item.label"].str.lower().apply(lambda s: all(k.lower() in s for k in keywords))
    if code_prefix:
        mask &= df["item.code"].astype(str).str.startswith(code_prefix)
    matched_labels = df.loc[mask, "item.label"].unique()
    print(f"    keywords {keywords} matched items: {list(matched_labels)[:5]}"
          f"{' ...' if len(matched_labels) > 5 else ''}")
    if not mask.any():
        print(f"    WARNING: no items matched {keywords}, using 0")
        return zeros()

    value_col = "amount.sum" if "amount.sum" in df.columns else \
        [c for c in df.columns if c.endswith(".sum")][0]
    result = df[mask].groupby(IDX)[value_col].sum()
    if reference_index is not None:
        result = result.reindex(reference_index, fill_value=0.0)
    return result


def main():
    print("Project root:", PROJECT_ROOT)
    print("Looking for audit file at:", AUDIT_FILE)
    if not AUDIT_FILE.exists():
        raise FileNotFoundError(
            f"Could not find {AUDIT_FILE}. Make sure audit_opinion_facts.csv "
            f"is inside the data/raw folder next to this script's project root."
        )

    print("\nFetching cubes (Audited Actual, all years 2018-2025)...\n")

    incexp = fetch_cube_all_years("incexp_v2", "amount.sum")
    print(f"incexp_v2: {incexp.shape}")
    cflow = fetch_cube_all_years("cflow_v2", "amount.sum")
    print(f"cflow_v2: {cflow.shape}")
    finpos = fetch_cube_all_years("financial_position_v2", "amount.sum")
    print(f"financial_position_v2: {finpos.shape}")
    creditor = fetch_cube_all_years("aged_creditor_v2", "total_amount.sum")
    print(f"aged_creditor_v2: {creditor.shape}")
    debtor = fetch_cube_all_years("aged_debtor_v2", "total_amount.sum")
    print(f"aged_debtor_v2: {debtor.shape}")
    # grants_v2 has no amount_type dimension at all, so no filter is passed
    grants = fetch_cube_all_years("grants_v2", "amount.sum", amount_type=None, drilldown=DRILL_GRANTS)
    if not grants.empty:
        grants = grants.rename(columns={"grant.code": "item.code", "grant.label": "item.label"})
    print(f"grants_v2: {grants.shape}")
    repmaint = fetch_cube_all_years("repmaint_v2", "amount.sum")
    print(f"repmaint_v2: {repmaint.shape}")
    uifwexp = fetch_cube_all_years("uifwexp", "amount.sum", amount_type=None)
    print(f"uifwexp: {uifwexp.shape}\n")

    for name, df in [("incexp", incexp), ("cflow", cflow), ("finpos", finpos),
                      ("creditor", creditor), ("debtor", debtor), ("grants", grants),
                      ("repmaint", repmaint), ("uifwexp", uifwexp)]:
        df.to_csv(RAW / f"{name}_audited_all.csv", index=False)

    print("Engineering ratios...\n")

    # mSCOA item code ranges (from the standard chart of accounts):
    # revenue items are coded below 2900 (2900 itself is the precomputed Total
    # Revenue row); expenditure items are coded 3000-4399 (4400 is the
    # precomputed Total Expenditure row). We sum the COMPONENTS rather than
    # trust the precomputed Total rows, since those aren't consistently
    # backfilled for older years while the components are.
    total_revenue = sum_by_code_range(incexp, 0, 2900)
    ref_idx = total_revenue.index

    total_expenditure = sum_by_code_range(incexp, 3000, 4400, reference_index=ref_idx)
    operating_surplus = total_revenue - total_expenditure

    cash = find_item_total(finpos, ["cash", "cash equivalents"], reference_index=ref_idx)
    monthly_expenditure = total_expenditure / 12
    cash_coverage_ratio = cash / monthly_expenditure.replace(0, pd.NA)

    billed_revenue = find_item_total(incexp, ["property rates"], reference_index=ref_idx) + \
        find_item_total(incexp, ["service charges"], reference_index=ref_idx)
    bad_debts = find_item_total(incexp, ["debt impairment"], reference_index=ref_idx)
    collection_rate = (billed_revenue - bad_debts) / billed_revenue.replace(0, pd.NA)

    total_debtors = find_item_total(debtor, ["total by income source"], reference_index=ref_idx)
    debtor_days = total_debtors / (total_revenue.replace(0, pd.NA) / 365)

    # Every row in repmaint_v2 IS repairs & maintenance spend (broken down by cost
    # type, e.g. "Contracted Services", "Employee Related Costs"), so sum it all —
    # no keyword filter needed here, unlike the other cubes.
    if not repmaint.empty:
        rm_spend = repmaint.groupby(IDX)["amount.sum"].sum().reindex(ref_idx, fill_value=0.0)
    else:
        rm_spend = pd.Series(0.0, index=ref_idx)
    repairs_maintenance_ratio = rm_spend / total_expenditure.replace(0, pd.NA)

    irregular_total = find_item_total(uifwexp, ["irregular"], reference_index=ref_idx)
    irregular_expenditure_ratio = irregular_total / total_expenditure.replace(0, pd.NA)

    # grants_v2 lists individual named grants (no single "Total" row), so sum
    # every grant received, same approach as repmaint above.
    if not grants.empty:
        grant_total = grants.groupby(IDX)["amount.sum"].sum().reindex(ref_idx, fill_value=0.0)
    else:
        grant_total = pd.Series(0.0, index=ref_idx)
    grant_dependency = grant_total / total_revenue.replace(0, pd.NA)

    features = pd.DataFrame({
        "total_revenue": total_revenue,
        "total_expenditure": total_expenditure,
        "operating_surplus": operating_surplus,
        "cash_coverage_ratio": cash_coverage_ratio,
        "collection_rate": collection_rate,
        "debtor_days": debtor_days,
        "repairs_maintenance_ratio": repairs_maintenance_ratio,
        "irregular_expenditure_ratio": irregular_expenditure_ratio,
        "grant_dependency": grant_dependency,
    }).reset_index()

    print("\nFeature table shape:", features.shape)

    # ---------------- Build the lagged target ----------------
    audit = pd.read_csv(AUDIT_FILE).rename(
        columns={"Demarcation Code": "demarcation.code", "Year End": "financial_year_end.year"}
    )
    audit["risk_label"] = audit["Code"].apply(
        lambda c: "at_risk" if c in AT_RISK else ("not_at_risk" if c in NOT_AT_RISK else None)
    )
    audit = audit.dropna(subset=["risk_label"])

    audit_prior = audit[["demarcation.code", "financial_year_end.year", "risk_label", "Code"]].rename(
        columns={"risk_label": "prior_audit_outcome", "Code": "prior_audit_opinion_raw"}
    )

    audit_target = audit[["demarcation.code", "Name", "financial_year_end.year", "risk_label", "Code"]].copy()
    audit_target["financial_year_end.year"] -= 1
    audit_target = audit_target.rename(
        columns={"risk_label": "target_next_year", "Code": "target_opinion_raw"}
    )
    # Name is reference-only (not a model feature); it's attached via the t+1
    # audit row since that's where we have it, but describes the same municipality

    # ---- Diagnostics: why might years disappear in the merges below? ----
    print("\n--- DIAGNOSTICS before merging ---")
    print("features years:", sorted(features["financial_year_end.year"].unique()),
          "dtype:", features["financial_year_end.year"].dtype)
    print("audit_prior years:", sorted(audit_prior["financial_year_end.year"].unique()),
          "dtype:", audit_prior["financial_year_end.year"].dtype)
    print("audit_target years (shifted, i.e. t):", sorted(audit_target["financial_year_end.year"].unique()),
          "dtype:", audit_target["financial_year_end.year"].dtype)
    print("features demarcation.code sample:", features["demarcation.code"].head(3).tolist())
    print("audit demarcation.code sample:", audit_prior["demarcation.code"].head(3).tolist())
    print("--- end diagnostics ---\n")

    modeling = features.merge(audit_prior, on=IDX, how="left")
    print("After merging prior outcome:", modeling.shape)
    modeling = modeling.merge(audit_target, on=IDX, how="inner")
    print("After merging t+1 target (inner):", modeling.shape)

    print("Modelling dataset shape (features for year t, target for t+1):", modeling.shape)
    print(modeling["target_next_year"].value_counts())

    modeling.to_csv(OUT / "modeling_dataset.csv", index=False)
    print(f"\nSaved {OUT / 'modeling_dataset.csv'}")


if __name__ == "__main__":
    main()
