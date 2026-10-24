"""Rebuild the source panel and reproduce the current employment/productivity study.
Run python -X utf8 run_analysis.py. Raw BLS slices must already be downloaded.
"""
from pathlib import Path
import os,sys,runpy,hashlib,json
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parent
os.chdir(ROOT)
if hasattr(sys.stdout,'reconfigure'):sys.stdout.reconfigure(encoding='utf-8')
RAW=ROOT/'data/raw'
(ROOT/'data/processed').mkdir(exist_ok=True)
STATES = {
    "AL": "Alabama", "AK": "Alaska", "AZ": "Arizona", "AR": "Arkansas", "CA": "California",
    "CO": "Colorado", "CT": "Connecticut", "DE": "Delaware", "FL": "Florida", "GA": "Georgia",
    "HI": "Hawaii", "ID": "Idaho", "IL": "Illinois", "IN": "Indiana", "IA": "Iowa",
    "KS": "Kansas", "KY": "Kentucky", "LA": "Louisiana", "ME": "Maine", "MD": "Maryland",
    "MA": "Massachusetts", "MI": "Michigan", "MN": "Minnesota", "MS": "Mississippi", "MO": "Missouri",
    "MT": "Montana", "NE": "Nebraska", "NV": "Nevada", "NH": "New Hampshire", "NJ": "New Jersey",
    "NM": "New Mexico", "NY": "New York", "NC": "North Carolina", "ND": "North Dakota", "OH": "Ohio",
    "OK": "Oklahoma", "OR": "Oregon", "PA": "Pennsylvania", "RI": "Rhode Island", "SC": "South Carolina",
    "SD": "South Dakota", "TN": "Tennessee", "TX": "Texas", "UT": "Utah", "VT": "Vermont",
    "VA": "Virginia", "WA": "Washington", "WV": "West Virginia", "WI": "Wisconsin", "WY": "Wyoming",
}
NAME2ABBR = {v: k for k, v in STATES.items()}


def seds(file, msns):
    # EIA SEDS wide file -> one row per (state, year), one column per series
    d = pd.read_csv(f"{RAW}/{file}.csv", dtype={"MSN": str, "State": str})
    d = d[d.MSN.isin(msns) & d.State.isin(STATES)]
    years = [c for c in d.columns if c.isdigit()]
    long = d.melt(id_vars=["State", "MSN"], value_vars=years, var_name="year", value_name="value")
    long["year"] = long.year.astype(int)
    long["value"] = pd.to_numeric(long.value, errors="coerce")
    return long.pivot(index=["State", "year"], columns="MSN", values="value").reset_index()


def sagdp(table, linecodes):
    # BEA SAGDP all-areas file -> one row per (state, year), one column per LineCode
    d = pd.read_csv(f"{RAW}/sagdp/{table}__ALL_AREAS_1997_2025.csv", encoding="latin-1", dtype=str)
    d.columns = d.columns.str.strip()
    d["GeoName"] = d.GeoName.str.strip()
    d["LineCode"] = d.LineCode.str.strip()
    d = d[d.GeoName.isin(NAME2ABBR) & d.LineCode.isin(linecodes)].copy()
    d["State"] = d.GeoName.map(NAME2ABBR)
    years = [c for c in d.columns if c.isdigit()]
    long = d.melt(id_vars=["State", "LineCode"], value_vars=years, var_name="year", value_name="value")
    long["year"] = long.year.astype(int)
    long["value"] = pd.to_numeric(long.value, errors="coerce")  # BEA marks suppressed cells "(D)"
    return long.pivot(index=["State", "year"], columns="LineCode", values="value").reset_index()


def land_area():
    # Census 2024 Gazetteer, tab-separated; USPS is the state abbreviation
    g = pd.read_csv(f"{RAW}/2024_Gaz_state_national.txt", sep="\t", dtype={"USPS": str})
    g = g.rename(columns=str.strip)
    g = g[g.USPS.isin(STATES)].rename(columns={"USPS": "State", "ALAND_SQMI": "land_sqmi"})
    return g[["State", "land_sqmi"]]

use = seds("use_all_btu", ["CLTCB", "NGTCB", "PATCB", "NUETB", "RETCB", "TETCB", "TPOPP"])
co2 = seds("co2_all", ["TETCE"])
price = seds("pr_all", ["TETCD"])
real = sagdp("SAGDP9", ["1"]).rename(columns={"1": "rgdp"})
nominal = sagdp("SAGDP2", ["1", "6", "12"]).rename(columns={"1": "ngdp", "6": "mining", "12": "manuf"})

panel = (use.merge(co2, on=["State", "year"]).merge(price, on=["State", "year"])
            .merge(real, on=["State", "year"]).merge(nominal, on=["State", "year"])
            .merge(land_area(), on="State"))

panel["fuel_total"] = panel[["CLTCB", "NGTCB", "PATCB", "NUETB", "RETCB"]].sum(axis=1, min_count=5)
panel["coal_share"] = panel.CLTCB / panel.fuel_total
panel["lowcarbon_share"] = (panel.RETCB + panel.NUETB) / panel.fuel_total
panel["gas_share"] = panel.NGTCB / panel.fuel_total
panel["petroleum_share"] = panel.PATCB / panel.fuel_total
panel["industry_share"] = (panel.mining + panel.manuf) / panel.ngdp
panel["pop"] = panel.TPOPP * 1000
panel["co2_pc"] = panel.TETCE * 1e6 / panel["pop"]   # tonnes per person
panel["ln_co2_pc"] = np.log(panel.co2_pc)
panel["ln_price"] = np.log(panel.TETCD)
panel["ln_density"] = np.log(panel["pop"] / panel.land_sqmi)
panel["rgdp_pc"] = panel.rgdp * 1e6 / panel["pop"]


assert not panel.duplicated(['State','year']).any()
assert len(panel)==1400 and panel.State.nunique()==50
assert np.allclose(panel.coal_share+panel.gas_share+panel.petroleum_share+panel.lowcarbon_share,1)
panel.to_csv(ROOT/'data/processed/state_panel.csv',index=False,encoding='utf-8-sig')
runpy.run_path(str(ROOT/'scripts/analyze_productivity.py'),run_name='__main__')
out=ROOT/'data/processed/employment_carbon_productivity'
manifest={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(RAW.rglob('*')) if p.is_file() and p.suffix.lower() in ['.csv','.txt']}
(out/'raw_data_sha256.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
print('Current carbon-productivity analysis reproduced from raw inputs.')
