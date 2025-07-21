import pandas as pd
import numpy as np
import math
from typing import Callable
import os
from scipy.special import gammainc
from scipy.stats import norm

# ==============================
# 0. Paramètres globaux
# ==============================

# Plage de temps
t_min = 0
t_max = 10000
n_points = 200
temps = np.linspace(t_min, t_max, n_points)

# Variation pour la dérivée numérique
delta = 1e-4

# Chemin du fichier des meilleures lois
fichier_lois = r"C:\Users\COMPUTER\Validation_Lois_Fiabilite.xlsx"

# ==============================
# 1. Fonctions R(t)
# ==============================

def R_weibull_2p(t, alpha, beta):
    return np.exp(-(t / alpha) ** beta)

def R_weibull_3p(t, alpha, beta, gamma):
    t_adj = np.maximum(t - gamma, 0)
    return np.exp(-(t_adj / alpha) ** beta)

def R_gamma(t, k, theta):
    return 1 - gammainc(k, t / theta)

def R_lognormale(t, mu_ln, sigma_ln):
    with np.errstate(divide='ignore'):
        return 1 - norm.cdf((np.log(t) - mu_ln) / sigma_ln)

def R_gumbel(t, mu, beta):
    # Corrige beta trop petit ou nul
    beta = max(beta, 1e-6)
    z = -(t - mu) / beta
    z = np.clip(z, -700, 700)  # éviter overflow dans exp()
    return np.exp(-np.exp(z))

def R_exponentielle(t, lambda_):
    return np.exp(-lambda_ * t)

# ==============================
# 2. Lecture des données
# ==============================

df_lois = pd.read_excel(fichier_lois, sheet_name="Résumé Meilleure Loi")

# ==============================
# 3. Fiabilités des composants
# ==============================

resultats = []
fiabilites_composants = {}

for (site, composant), row in df_lois.groupby(["Site", "Composant"]):
    loi = row["Loi"].values[0]

    try:
        if loi == "Weibull 2P":
            alpha = row["alpha"].values[0]
            beta = row["beta"].values[0]
            R_t = R_weibull_2p(temps, alpha, beta)

        elif loi == "Weibull 3P":
            alpha = row["alpha"].values[0]
            beta = row["beta"].values[0]
            gamma_ = row["gamma"].values[0]
            R_t = R_weibull_3p(temps, alpha, beta, gamma_)

        elif loi == "Gamma":
            k = row["k"].values[0]
            theta = row["theta"].values[0]
            R_t = R_gamma(temps, k, theta)

        elif loi == "Lognormale":
            mu_ln = row["mu_ln"].values[0]
            sigma_ln = row["sigma_ln"].values[0]
            R_t = R_lognormale(temps, mu_ln, sigma_ln)

        elif loi == "Gumbel":
            mu = row["mu_gumbel"].values[0]
            beta = row["beta_gumbel"].values[0]
            R_t = R_gumbel(temps, mu, beta)

        elif loi == "Exponentielle":
            lambda_ = row["lambda_"].values[0]
            R_t = R_exponentielle(temps, lambda_)

        else:
            print(f"[Info] Loi non supportée : {loi}")
            continue

        label = f"{site} | {composant}"
        fiabilites_composants[label] = R_t

    except Exception as e:
        print(f"[Erreur] {site} - {composant} - {loi} : {e}")

# ==============================
# 4. Fiabilités des sites
# ==============================

fiabilites_sites = {}
sites = df_lois["Site"].unique()

for site in sites:
    composants_site = [key for key in fiabilites_composants if key.startswith(site)]
    R_site = np.ones_like(temps)
    for comp in composants_site:
        R_site *= fiabilites_composants[comp]
    fiabilites_sites[site] = R_site

# ==============================
# 5. Facteurs d’importance
# ==============================

facteurs_importance = []

for site in sites:
    composants_site = [key for key in fiabilites_composants if key.startswith(site)]
    R0 = fiabilites_sites[site]

    for comp in composants_site:
        R_i = fiabilites_composants[comp]

        # Perturbation numérique
        R_plus = {k: (v if k != comp else np.clip(v + delta, 0, 1)) for k, v in fiabilites_composants.items()}
        R_minus = {k: (v if k != comp else np.clip(v - delta, 0, 1)) for k, v in fiabilites_composants.items()}

        R_site_plus = np.ones_like(temps)
        R_site_minus = np.ones_like(temps)

        for k in composants_site:
            R_site_plus *= R_plus[k]
            R_site_minus *= R_minus[k]

        importance = (R_site_plus - R_site_minus) / (2 * delta)

        for t, val in zip(temps, importance):
            facteurs_importance.append({
                "Site": site,
                "Composant": comp.split(" | ")[1],
                "Temps": t,
                "Importance_Marginale": val
            })

# ==============================
# 6. Export Excel
# ==============================

df_fiabilite_comps = pd.DataFrame({
    "Temps": temps,
    **{k: v for k, v in fiabilites_composants.items()}
})

df_fiabilite_sites = pd.DataFrame({
    "Temps": temps,
    **{k: v for k, v in fiabilites_sites.items()}
})

df_importance = pd.DataFrame(facteurs_importance)

chemin_export = r"C:\Users\COMPUTER\Fiabilite_Sites_Composants.xlsx"

with pd.ExcelWriter(chemin_export, engine="openpyxl") as writer:
    df_fiabilite_comps.to_excel(writer, sheet_name="R_composants", index=False)
    df_fiabilite_sites.to_excel(writer, sheet_name="R_sites", index=False)
    df_importance.to_excel(writer, sheet_name="Importance", index=False)

print(f"✅ Export terminé : {chemin_export}")
