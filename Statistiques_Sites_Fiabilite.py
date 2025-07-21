import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gamma, lognorm
import os

# === 0. Chargement des paramètres ===
chemin = r"C:\Users\COMPUTER\Validation_Lois_Fiabilite.xlsx"
df = pd.read_excel(chemin, sheet_name="Résumé Meilleure Loi")

# === 1. Définir les fonctions de fiabilité ===
def R_weibull2p(t, alpha, beta):
    return np.exp(-(t / alpha) ** beta)

def R_weibull3p(t, alpha, beta, gamma_):
    t_adj = np.maximum(t - gamma_, 1e-6)
    return np.exp(-(t_adj / alpha) ** beta)

def R_gamma(t, k, theta):
    from scipy.stats import gamma
    return 1 - gamma.cdf(t, a=k, scale=theta)

def R_lognormale(t, mu, sigma):
    from scipy.stats import lognorm
    return 1 - lognorm.cdf(t, s=sigma, scale=np.exp(mu))

def R_gumbel(t, mu, beta):
    z = (t - mu) / beta
    return np.exp(-np.exp(-z))

def R_expo(t, lambd):
    return np.exp(-lambd * t)

# === 2. Paramètres de temps ===
t = np.linspace(0, 600, 100)

# === 3. Fiabilité globale par site ===
sites = df["Site"].unique()
print("sites :", sites)
courbes = {}

for site in sites:
    df_site = df[df["Site"] == site]
    R_total = np.ones_like(t)

    for _, row in df_site.iterrows():
        loi = row["Loi"]

        try:
            if loi == "Weibull 2P":
                R = R_weibull2p(t, row["alpha"], row["beta"])
            elif loi == "Weibull 3P":
                R = R_weibull3p(t, row["alpha"], row["beta"], row["gamma"])
            elif loi == "Gamma":
                R = R_gamma(t, row["k"], row["theta"])
            elif loi == "Lognormale":
                R = R_lognormale(t, row["mu_ln"], row["sigma_ln"])
            elif loi == "Gumbel":
                R = R_gumbel(t, row["mu_gumbel"], row["beta_gumbel"])
            elif loi == "Exponentielle":
                R = R_expo(t, row["lambda_"])
            else:
                continue

            R_total *= R  # Système en série
        except Exception as e:
            print(f"[Erreur] {site} - {row['Composant']} - {loi} : {e}")

    courbes[site] = R_total

# === 4. Tracer un seul graphique comparatif ===
plt.figure(figsize=(10, 6))
for site, R in courbes.items():
    plt.plot(t, R, label=site, linewidth=2)

plt.title("Comparaison des fiabilités des sites", fontsize=14)
plt.xlabel("Temps $t$ (heures)", fontsize=12)
plt.ylabel("Fiabilité $R(t)$", fontsize=12)
plt.grid(True)
plt.legend()
plt.tight_layout()

# Enregistrer la figure
plt.savefig(r"C:\Users\COMPUTER\Comparaison_Fiabilite_Sites.png")
plt.show()
