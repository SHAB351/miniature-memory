import pandas as pd
import numpy as np
import math
import scipy.stats
from scipy.special import erfinv
import warnings
warnings.filterwarnings("ignore")

# ======================== 1. Fonctions statistiques par loi ========================

def weibull_2p_stats(alpha, beta):
    mtbf = alpha * math.gamma(1 + 1 / beta)
    median = alpha * (math.log(2))**(1 / beta)
    mode = alpha * ((beta - 1) / beta) ** (1 / beta) if beta > 1 else 0
    q75 = alpha * (-math.log(1 - 0.25)) ** (1 / beta)
    q99 = alpha * (-math.log(1 - 0.01)) ** (1 / beta)

    def hazard(t): return (beta / alpha) * (t / alpha) ** (beta - 1)
    return {
        "MTBF": f"{mtbf:.1f} ({hazard(mtbf):.5f})",
        "Mediane": f"{median:.1f} ({hazard(median):.5f})",
        "Mode": f"{mode:.1f} ({hazard(mode):.5f})",
        "Q75": f"{q75:.1f} ({hazard(q75):.5f})",
        "Q99": f"{q99:.1f} ({hazard(q99):.5f})"
    }

def weibull_3p_stats(alpha, beta, gamma_):
    mtbf = gamma_ + alpha * math.gamma(1 + 1 / beta)
    median = gamma_ + alpha * (math.log(2))**(1 / beta)
    mode = gamma_ + alpha * ((beta - 1) / beta) ** (1 / beta) if beta > 1 else gamma_
    q75 = gamma_ + alpha * (-math.log(1 - 0.25)) ** (1 / beta)
    q99 = gamma_ + alpha * (-math.log(1 - 0.01)) ** (1 / beta)

    def hazard(t):
        t_ = t - gamma_
        if t_ <= 0: return 0
        return (beta / alpha) * (t_ / alpha) ** (beta - 1)
    return {
        "MTBF": f"{mtbf:.1f} ({hazard(mtbf):.5f})",
        "Mediane": f"{median:.1f} ({hazard(median):.5f})",
        "Mode": f"{mode:.1f} ({hazard(mode):.5f})",
        "Q75": f"{q75:.1f} ({hazard(q75):.5f})",
        "Q99": f"{q99:.1f} ({hazard(q99):.5f})"
    }

def gamma_stats(k, theta):
    mtbf = k * theta
    median = scipy.stats.gamma.ppf(0.5, a=k, scale=theta)
    mode = (k - 1) * theta if k >= 1 else 0
    q75 = scipy.stats.gamma.ppf(0.75, a=k, scale=theta)
    q99 = scipy.stats.gamma.ppf(0.99, a=k, scale=theta)

    def hazard(t):
        if t <= 0: return 0
        f = scipy.stats.gamma.pdf(t, a=k, scale=theta)
        R = 1 - scipy.stats.gamma.cdf(t, a=k, scale=theta)
        return f / R if R > 0 else 0

    return {
        "MTBF": f"{mtbf:.1f} ({hazard(mtbf):.5f})",
        "Mediane": f"{median:.1f} ({hazard(median):.5f})",
        "Mode": f"{mode:.1f} ({hazard(mode):.5f})",
        "Q75": f"{q75:.1f} ({hazard(q75):.5f})",
        "Q99": f"{q99:.1f} ({hazard(q99):.5f})"
    }

def lognormale_stats(mu, sigma):
    mtbf = math.exp(mu + sigma**2 / 2)
    median = math.exp(mu)
    mode = math.exp(mu - sigma**2)
    q75 = math.exp(mu + sigma * math.sqrt(2) * erfinv(2 * 0.75 - 1))
    q99 = math.exp(mu + sigma * math.sqrt(2) * erfinv(2 * 0.99 - 1))

    def hazard(t):
        if t <= 0: return 0
        f = (1 / (t * sigma * math.sqrt(2 * math.pi))) * math.exp(-(math.log(t) - mu) ** 2 / (2 * sigma**2))
        R = 1 - scipy.stats.lognorm.cdf(t, s=sigma, scale=np.exp(mu))
        return f / R if R > 0 else 0

    return {
        "MTBF": f"{mtbf:.1f} ({hazard(mtbf):.5f})",
        "Mediane": f"{median:.1f} ({hazard(median):.5f})",
        "Mode": f"{mode:.1f} ({hazard(mode):.5f})",
        "Q75": f"{q75:.1f} ({hazard(q75):.5f})",
        "Q99": f"{q99:.1f} ({hazard(q99):.5f})"
    }

def exponentielle_stats(lmbda):
    mtbf = 1 / lmbda
    median = math.log(2) / lmbda
    mode = 0
    q75 = -math.log(1 - 0.25) / lmbda
    q99 = -math.log(1 - 0.01) / lmbda

    def hazard(_): return lmbda
    return {
        "MTBF": f"{mtbf:.1f} ({hazard(mtbf):.5f})",
        "Mediane": f"{median:.1f} ({hazard(median):.5f})",
        "Mode": f"{mode:.1f} ({hazard(mode):.5f})",
        "Q75": f"{q75:.1f} ({hazard(q75):.5f})",
        "Q99": f"{q99:.1f} ({hazard(q99):.5f})"
    }

def gumbel_stats(mu, beta):
    mtbf = mu + beta * 0.5772
    median = mu - beta * math.log(math.log(2))
    mode = mu
    q75 = scipy.stats.gumbel_r.ppf(0.75, loc=mu, scale=beta)
    q99 = scipy.stats.gumbel_r.ppf(0.99, loc=mu, scale=beta)

    def hazard(t):
        f = scipy.stats.gumbel_r.pdf(t, loc=mu, scale=beta)
        R = 1 - scipy.stats.gumbel_r.cdf(t, loc=mu, scale=beta)
        return f / R if R > 0 else 0

    return {
        "MTBF": f"{mtbf:.1f} ({hazard(mtbf):.5f})",
        "Mediane": f"{median:.1f} ({hazard(median):.5f})",
        "Mode": f"{mode:.1f} ({hazard(mode):.5f})",
        "Q75": f"{q75:.1f} ({hazard(q75):.5f})",
        "Q99": f"{q99:.1f} ({hazard(q99):.5f})"
    }

# ======================== 2. Lecture des paramètres ========================

df = pd.read_excel(r"C:\Users\COMPUTER\Validation_Lois_Fiabilite.xlsx", sheet_name="Résumé Meilleure Loi")
resultats = []

# ======================== 3. Traitement ========================

for _, row in df.iterrows():
    try:
        loi = row["Loi"]
        methode = row["Méthode"]

        if loi == "Weibull 2P":
            stats = weibull_2p_stats(float(row["alpha"]), float(row["beta"]))
        elif loi == "Weibull 3P":
            stats = weibull_3p_stats(float(row["alpha"]), float(row["beta"]), float(row["gamma"]))
        elif loi == "Gamma":
            stats = gamma_stats(float(row["k"]), float(row["theta"]))
        elif loi == "Lognormale":
            stats = lognormale_stats(float(row["mu_ln"]), float(row["sigma_ln"]))
        elif loi == "Exponentielle":
            stats = exponentielle_stats(float(row["lambda_"]))
        elif loi == "Gumbel":
            stats = gumbel_stats(float(row["mu_gumbel"]), float(row["beta_gumbel"]))
        else:
            continue

        ligne = {
            "Site": row["Site"],
            "Composant": row["Composant"],
            "Loi": loi,
            "Méthode": methode
        }
        ligne.update(stats)
        resultats.append(ligne)

    except Exception as e:
        print(f"[Erreur] {row['Site']} - {row['Composant']} - {loi} : {e}")
        continue

# ======================== 4. Export ========================

df_stats = pd.DataFrame(resultats)
df_stats.to_excel("Statistiques_Fiabilite.xlsx", index=False)
print("✅ Statistiques calculées et exportées dans 'Statistiques_Fiabilite.xlsx'")
