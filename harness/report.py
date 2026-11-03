"""Readable summary of a run: uv run python harness/report.py exp_demo [--top 15]"""
import argparse, json
from pathlib import Path
import duckdb

ROOT = Path(__file__).resolve().parent.parent
ap = argparse.ArgumentParser(); ap.add_argument("exp"); ap.add_argument("--top", type=int, default=15)
a = ap.parse_args()
d = ROOT / "runs" / a.exp
st = json.loads((d / "state.json").read_text()) if (d / "state.json").exists() else {}
print(f"# {a.exp}  n={st.get('n')}  errors={st.get('n_err')}  confirmed={st.get('confirmed')}")
for f in st.get("filons", {}).values():
    print(f"  {f['id']} {f['status']:9s} trigger={f['trigger']:.4g} best={f['best']:.4g} "
          f"rep={f.get('rep_mean', float('nan')):.4g} t={f.get('t_stat', float('nan')):.3g} exploit={f['n_exploit']}")
con = duckdb.connect()
q = f"""select kind, count(*) n, round(avg(metrics.score),4) mean, round(max(metrics.score),4) max
        from read_json_auto('{d}/results.jsonl', format='newline_delimited', ignore_errors=true)
        where error is null group by kind order by n desc"""
print(con.sql(q))
print(con.sql(f"""select kind, round(metrics.score,4) score, params::varchar params, seed
   from read_json_auto('{d}/results.jsonl', format='newline_delimited', ignore_errors=true)
   where error is null order by metrics.score desc limit {a.top}"""))
