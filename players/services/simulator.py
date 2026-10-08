"""Transfer ROI simulation, following the project plan:

    ROI = (predicted future value - cost) / cost * 100

Cost is the transfer fee plus salary over the holding period (capped by contract
length), as in the frontend's Simulator page. Risk-adjusted ROI scales ROI by
(1 - riskScore/100).
"""

# Multiplier applied to the model's expected gain (predicted - fee) per scenario.
SCENARIO_GAIN_MULTIPLIER = {"conservative": 0.7, "base": 1.0, "optimistic": 1.35}
BEAR_MULTIPLIER = 0.3
BULL_MULTIPLIER = 1.5


def _case(label, fee, predicted, total_cost, mult):
    future = fee + (predicted - fee) * mult
    roi = (future - total_cost) / total_cost * 100
    return {"case": label, "futureValue": round(future, 2), "roi": round(roi, 1)}


def simulate_transfer(*, fee, predicted_value, risk_score, salary=0.0, contract_years=1,
                      holding_years=1, scenario="base"):
    if fee <= 0:
        raise ValueError("fee must be positive")
    scenario = scenario.lower()
    if scenario not in SCENARIO_GAIN_MULTIPLIER:
        raise ValueError(f"scenario must be one of {sorted(SCENARIO_GAIN_MULTIPLIER)}")

    total_cost = fee + salary * min(holding_years, contract_years)
    cases = [
        _case("Bear Case", fee, predicted_value, total_cost, BEAR_MULTIPLIER),
        _case("Base Case", fee, predicted_value, total_cost, SCENARIO_GAIN_MULTIPLIER[scenario]),
        _case("Bull Case", fee, predicted_value, total_cost, BULL_MULTIPLIER),
    ]
    base = cases[1]
    risk_factor = 1 - (risk_score or 0) / 100
    return {
        "totalCost": round(total_cost, 2),
        "scenario": scenario,
        "projectedROI": base["roi"],
        "riskAdjustedROI": round(base["roi"] * risk_factor, 1),
        "cases": cases,
    }


def build_portfolio(candidates, *, budget, max_players=None):
    """Greedy portfolio: rank by risk-adjusted upside and buy while the budget allows.

    candidates: iterable of dicts with id, name, currentValue, predictedValue, riskScore.
    Fee is assumed to equal current market value until real fee data exists.
    """
    ranked = []
    for c in candidates:
        cur, pred = c["currentValue"], c["predictedValue"]
        if not cur or pred is None:
            continue
        upside = (pred - cur) / cur * 100
        ranked.append({**c, "upside": upside, "score": upside * (1 - (c["riskScore"] or 0) / 100)})
    ranked.sort(key=lambda c: c["score"], reverse=True)

    picks, spent = [], 0.0
    for c in ranked:
        if c["score"] <= 0:
            break
        if max_players and len(picks) >= max_players:
            break
        if spent + c["currentValue"] <= budget:
            picks.append(c)
            spent += c["currentValue"]

    future = sum(c["predictedValue"] for c in picks)
    return {
        "budget": budget,
        "totalCost": round(spent, 2),
        "projectedValue": round(future, 2),
        "portfolioROI": round((future - spent) / spent * 100, 1) if spent else 0.0,
        "players": [
            {
                "id": c["id"], "name": c["name"], "fee": c["currentValue"],
                "predictedValue": c["predictedValue"], "upside": round(c["upside"]),
                "riskScore": c["riskScore"], "riskAdjustedScore": round(c["score"], 1),
            }
            for c in picks
        ],
    }
