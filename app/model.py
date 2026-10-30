"""Dixon-Coles model with time decay, keyed by team id, with neutral-venue support."""
import math

import numpy as np
from scipy.optimize import minimize

XI = 0.0019          # time decay per day (half-life about one year)
RIDGE = 0.01         # pulls thin-data teams toward average
MAX_GOALS = 10
_FACT = np.array([math.factorial(k) for k in range(MAX_GOALS + 1)], dtype=float)


class DixonColes:
    def fit(self, matches, ref_dt):
        ids = sorted({m["home_id"] for m in matches} | {m["away_id"] for m in matches})
        self.idx = {t: i for i, t in enumerate(ids)}
        self.names, self.n_matches = {}, {t: 0 for t in ids}
        for m in matches:
            self.names[m["home_id"]], self.names[m["away_id"]] = m["home"], m["away"]
            self.n_matches[m["home_id"]] += 1
            self.n_matches[m["away_id"]] += 1
        n = len(ids)
        h = np.array([self.idx[m["home_id"]] for m in matches])
        a = np.array([self.idx[m["away_id"]] for m in matches])
        x = np.array([m["hg"] for m in matches], dtype=float)
        y = np.array([m["ag"] for m in matches], dtype=float)
        hv = np.array([0.0 if m.get("neutral") else 1.0 for m in matches])
        days = np.array([(ref_dt - m["dt"]).total_seconds() / 86400 for m in matches])
        w = np.exp(-XI * np.maximum(days, 0))

        m00, m01 = (x == 0) & (y == 0), (x == 0) & (y == 1)
        m10, m11 = (x == 1) & (y == 0), (x == 1) & (y == 1)

        def nll(p):
            att, dfn, home, rho = p[:n], p[n:2 * n], p[2 * n], p[2 * n + 1]
            ll_lam = home * hv + att[h] - dfn[a]
            ll_mu = att[a] - dfn[h]
            lam, mu = np.exp(ll_lam), np.exp(ll_mu)
            tau = np.ones_like(lam)
            tau[m00] = 1 - lam[m00] * mu[m00] * rho
            tau[m01] = 1 + lam[m01] * rho
            tau[m10] = 1 + mu[m10] * rho
            tau[m11] = 1 - rho
            tau = np.maximum(tau, 1e-10)
            d_lam, d_mu, d_rho = np.zeros_like(lam), np.zeros_like(lam), np.zeros_like(lam)
            d_lam[m00] = d_mu[m00] = -lam[m00] * mu[m00] * rho / tau[m00]
            d_rho[m00] = -lam[m00] * mu[m00] / tau[m00]
            d_lam[m01] = lam[m01] * rho / tau[m01]
            d_rho[m01] = lam[m01] / tau[m01]
            d_mu[m10] = mu[m10] * rho / tau[m10]
            d_rho[m10] = mu[m10] / tau[m10]
            d_rho[m11] = -1 / tau[m11]

            f = -(w * (np.log(tau) + x * ll_lam - lam + y * ll_mu - mu)).sum()
            f += RIDGE * (att @ att + dfn @ dfn)
            g_lam, g_mu = w * (x - lam + d_lam), w * (y - mu + d_mu)
            g = np.zeros_like(p)
            np.add.at(g, h, g_lam)
            np.add.at(g, n + a, -g_lam)
            np.add.at(g, a, g_mu)
            np.add.at(g, n + h, -g_mu)
            g[2 * n] = (g_lam * hv).sum()
            g[2 * n + 1] = (w * d_rho).sum()
            g = -g
            g[:2 * n] += 2 * RIDGE * p[:2 * n]
            return f, g

        p0 = np.zeros(2 * n + 2)
        p0[2 * n] = 0.25
        bounds = [(None, None)] * (2 * n + 1) + [(-0.2, 0.2)]
        p = minimize(nll, p0, jac=True, method="L-BFGS-B", bounds=bounds).x
        self.att, self.dfn = p[:n], p[n:2 * n]
        self.home_adv, self.rho = float(p[2 * n]), float(p[2 * n + 1])
        return self

    def knows(self, team_id):
        return team_id in self.idx

    def predict(self, home_id, away_id, neutral=False):
        i, j = self.idx[home_id], self.idx[away_id]
        lam = math.exp((0 if neutral else self.home_adv) + self.att[i] - self.dfn[j])
        mu = math.exp(self.att[j] - self.dfn[i])
        g = np.arange(MAX_GOALS + 1)
        M = np.outer(np.exp(-lam) * lam ** g / _FACT, np.exp(-mu) * mu ** g / _FACT)
        M[0, 0] *= 1 - lam * mu * self.rho
        M[0, 1] *= 1 + lam * self.rho
        M[1, 0] *= 1 + mu * self.rho
        M[1, 1] *= 1 - self.rho
        M /= M.sum()
        tot = np.add.outer(g, g)
        top = sorted(((float(M[r, c]), int(r), int(c)) for r in range(6) for c in range(6)), reverse=True)[:3]
        return {
            "home": float(np.tril(M, -1).sum()), "draw": float(np.trace(M)), "away": float(np.triu(M, 1).sum()),
            "over25": float(M[tot > 2.5].sum()), "btts": float(M[1:, 1:].sum()),
            "xg_home": lam, "xg_away": mu,
            "top_scores": [{"score": f"{r}-{c}", "p": p} for p, r, c in top],
        }

    def ratings(self, min_matches=5):
        rows = []
        for t, i in self.idx.items():
            if self.n_matches[t] >= min_matches:
                rows.append({"team": self.names[t], "attack": float(self.att[i]), "defence": float(self.dfn[i]),
                             "overall": float(self.att[i] + self.dfn[i]), "matches": self.n_matches[t]})
        return sorted(rows, key=lambda r: -r["overall"])
