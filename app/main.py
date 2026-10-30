"""Web app: fixtures with model probabilities, ratings, and an honest accuracy check."""
import math
import os
from datetime import datetime, timedelta, timezone

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from . import data
from .model import DixonColes

app = FastAPI(title="Football Model")
STATIC = os.path.join(os.path.dirname(__file__), "static")
app.mount("/static", StaticFiles(directory=STATIC), name="static")

_models = {}      # key -> (model, n_matches)
_accuracy = {}    # key -> (n_matches, result)
LOW_DATA = 8      # fewer weighted matches than this = low confidence


def _check(key):
    if key not in data.COMPETITIONS:
        raise HTTPException(404, f"Unknown competition {key}")


def get_model(key):
    matches = data.training_matches(key)
    if len(matches) < 30:
        raise HTTPException(409, "Not enough results yet. Press Sync to download history.")
    cached = _models.get(key)
    if cached and cached[1] == len(matches):
        return cached[0]
    model = DixonColes().fit(matches, datetime.now(timezone.utc))
    _models[key] = (model, len(matches))
    return model


@app.get("/")
def index():
    return FileResponse(os.path.join(STATIC, "index.html"))


@app.get("/api/competitions")
def competitions():
    status, counts = data.sync_status()
    out = []
    for key, (name, slug, train) in data.COMPETITIONS.items():
        st = status.get(slug, {})
        out.append({"key": key, "name": name, "synced_at": st.get("synced_at"),
                    "results": sum(counts.get(s, 0) for s in train),
                    "errors": [f"{s}: {status[s]['error']}" for s in train if status.get(s, {}).get("error")]})
    return out


@app.post("/api/sync/{key}")
def sync(key: str, force: bool = False):
    _check(key)
    res = data.sync_competition(key, force)
    _models.pop(key, None)
    return res


@app.get("/api/fixtures/{key}")
def fixtures(key: str, days: int = 14):
    _check(key)
    model = get_model(key)
    out = []
    for f in data.upcoming(key, days):
        item = {"id": f["id"], "date": f["date"], "home": f["home"], "away": f["away"],
                "neutral": bool(f["neutral"])}
        if model.knows(f["home_id"]) and model.knows(f["away_id"]):
            item["prediction"] = model.predict(f["home_id"], f["away_id"], bool(f["neutral"]))
            thin = min(model.n_matches[f["home_id"]], model.n_matches[f["away_id"]])
            item["low_data"] = thin < LOW_DATA
        else:
            item["prediction"] = None
            item["low_data"] = True
        out.append(item)
    return {"fixtures": out, "home_advantage": model.home_adv}


@app.get("/api/ratings/{key}")
def ratings(key: str):
    _check(key)
    return get_model(key).ratings()


@app.get("/api/accuracy/{key}")
def accuracy(key: str, days: int = 180, refit_days: int = 14):
    """Walk-forward test on recent results: each match is predicted using only earlier data."""
    _check(key)
    all_matches = data.training_matches(key)
    cached = _accuracy.get(key)
    if cached and cached[0] == len(all_matches):
        return cached[1]
    tests = data.recent_results(key, days)
    if len(tests) < 20:
        raise HTTPException(409, "Not enough recent results to test. Sync first.")
    model, last_fit = None, None
    ll = brier = hits = 0.0
    ll_base = 0.0
    n = 0
    for m in tests:
        if model is None or (m["dt"] - last_fit).days >= refit_days:
            hist = [x for x in all_matches if x["dt"] < m["dt"]]
            if len(hist) < 100:
                continue
            model, last_fit = DixonColes().fit(hist, m["dt"]), m["dt"]
            res_hist = [0 if x["hg"] > x["ag"] else 1 if x["hg"] == x["ag"] else 2 for x in hist[-1000:]]
            base = [max(res_hist.count(k), 1) / len(res_hist) for k in range(3)]
        if not (model.knows(m["home_id"]) and model.knows(m["away_id"])):
            continue
        p = model.predict(m["home_id"], m["away_id"], bool(m["neutral"]))
        pv = [p["home"], p["draw"], p["away"]]
        res = 0 if m["hg"] > m["ag"] else 1 if m["hg"] == m["ag"] else 2
        ll += -math.log(max(pv[res], 1e-12))
        ll_base += -math.log(base[res])
        brier += sum((pv[k] - (k == res)) ** 2 for k in range(3))
        hits += pv.index(max(pv)) == res
        n += 1
    if n == 0:
        raise HTTPException(409, "Not enough history before the test window.")
    result = {"matches": n, "log_loss": ll / n, "baseline_log_loss": ll_base / n,
              "brier": brier / n, "favourite_hit_rate": hits / n,
              "skill_vs_baseline": 1 - (ll / n) / (ll_base / n)}
    _accuracy[key] = (len(all_matches), result)
    return result


@app.get("/api/health")
def health():
    return {"ok": True, "time": datetime.now(timezone.utc).isoformat()}
