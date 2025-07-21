import pandas as pd
import numpy as np
from scipy.stats import gamma, lognorm
from scipy.special import gamma as gamma_func
from reliability.Fitters import Fit_Weibull_2P, Fit_Weibull_3P

# Charger les données
df = pd.read_excel(r"C:\Users\COMPUTER\Documents\TFC\FINALY\DONNEES TTR ET TBF 2.xlsx",sheet_name="Données TTR")
# Nettoyage des noms de colonnes (Solution 2)
df.columns = df.columns.str.strip().str.replace(r'[^a-zA-Z0-9]', '', regex=True)
# Colonnes du DataFrame final (14 colonnes sans MTBF)
colonnes_resultats = [
    "Site", "Composant", "Loi", "Méthode",
    "alpha", "beta", "gamma",        # Weibull
    "k", "theta",                    # Gamma
    "mu_ln", "sigma_ln",             # Lognormale
    "mu_gumbel", "beta_gumbel",      # Gumbel
    "lambda_"                        # Exponentielle
]

# Liste des résultats
resultats = []

# Boucle sur chaque groupe Site-Composant
for (site, composant), groupe in df.groupby(["Site", "Composant"]):
    tbf = groupe["TBF"].dropna().values
    if len(tbf) < 3:
        continue

    try:
        mean = np.mean(tbf)
        std = np.std(tbf, ddof=1)

        ### WEIBULL 2P - Moments ###
        def weibull_moments(data):
            def objective(beta):
                alpha = mean / gamma_func(1 + 1 / beta)
                return (alpha ** 2 * (gamma_func(1 + 2 / beta) - gamma_func(1 + 1 / beta) ** 2) - std ** 2) ** 2
            beta_grid = np.linspace(0.5, 10, 1000)
            beta_best = beta_grid[np.argmin([objective(b) for b in beta_grid])]
            alpha_best = mean / gamma_func(1 + 1 / beta_best)
            return alpha_best, beta_best

        alpha_mom, beta_mom = weibull_moments(tbf)
        resultats.append([site, composant, "Weibull 2P", "Moments",
                          alpha_mom, beta_mom, "", "", "", "", "", "", "", ""])

        ### WEIBULL 2P - MLE ###
        fit_mle = Fit_Weibull_2P(failures=tbf, method='MLE', show_probability_plot=False, print_results=False)
        resultats.append([site, composant, "Weibull 2P", "MLE",
                          fit_mle.alpha, fit_mle.beta, "", "", "", "", "", "", "", ""])

        ### WEIBULL 2P - Régression ###
        fit_ls = Fit_Weibull_2P(failures=tbf, method='LS', show_probability_plot=False, print_results=False)
        resultats.append([site, composant, "Weibull 2P", "Régression",
                          fit_ls.alpha, fit_ls.beta, "", "", "", "", "", "", "", ""])

        ### WEIBULL 3P - Itération ###
        fit_3p = Fit_Weibull_3P(failures=tbf, method='MLE', show_probability_plot=False, print_results=False)
        resultats.append([site, composant, "Weibull 3P", "Itération",
                          fit_3p.alpha, fit_3p.beta, fit_3p.gamma, "", "", "", "", "", ""])

        ### GAMMA ###
        k_hat = mean ** 2 / std ** 2
        theta_hat = std ** 2 / mean
        resultats.append([site, composant, "Gamma", "Moments",
                          "", "", "", k_hat, theta_hat, "", "", "", "", ""])

        ### LOGNORMALE ###
        logs = np.log(tbf)
        mu_ln = np.mean(logs)
        sigma_ln = np.std(logs, ddof=1)
        resultats.append([site, composant, "Lognormale", "Moments",
                          "", "", "", "", "", mu_ln, sigma_ln, "", "", ""])

        ### GUMBEL ###
        beta_gumbel = std * np.sqrt(6) / np.pi
        mu_gumbel = mean - 0.5772 * beta_gumbel
        resultats.append([site, composant, "Gumbel", "Moments",
                          "", "", "", "", "", "", "", mu_gumbel, beta_gumbel, ""])

        ### EXPONENTIELLE ###
        lambda_hat = 1 / mean
        resultats.append([site, composant, "Exponentielle", "MLE",
                          "", "", "", "", "", "", "", "", "", lambda_hat])

    except Exception as e:
        print(f"[Erreur] {site} - {composant} : {e}")

# Création du DataFrame
df_resultats = pd.DataFrame(resultats, columns=colonnes_resultats)

# Export vers Excel
df_resultats.to_excel("Parametres_Fiabilite_Sans_MTBF.xlsx", index=False)
print("✅ Résultats exportés vers Parametres_Fiabilite_Sans_MTBF.xlsx")
