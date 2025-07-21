import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.stats import kstest, anderson, expon, gamma, lognorm, gumbel_r, weibull_min
from reliability.Fitters import Fit_Weibull_2P, Fit_Weibull_3P

# ======================= 0. Préparation =======================

# Charger les données de fiabilité (paramètres estimés)
parametres = pd.read_excel(r"C:\Users\COMPUTER\Parametres_Fiabilite_Sans_MTBF.xlsx")

# Charger la base TBF d'origine
df_tbf = pd.read_excel(r"C:\Users\COMPUTER\Documents\TFC\FINALY\DONNEES TTR ET TBF 2.xlsx",sheet_name="Données TTR")
# Nettoyage des noms de colonnes (Solution 2)
df_tbf.columns = df_tbf.columns.str.strip().str.replace(r'[^a-zA-Z0-9]', '', regex=True)


# Créer dossier pour sauvegarder les graphes
os.makedirs("graphes_validation", exist_ok=True)

# Initialiser liste pour stocker les résultats des tests
validation_resultats = []

# ======================= 1. Boucle site/composant =======================

# Boucle sur chaque couple (site, composant)
for (site, composant), groupe in df_tbf.groupby(["Site", "Composant"]):

    tbf = groupe["TBF"].dropna().values
    if len(tbf) < 5:
        continue  # Trop peu de données pour une validation fiable

    # Extraire les lois disponibles pour ce couple dans les paramètres
    param_group = parametres[(parametres["Site"] == site) & (parametres["Composant"] == composant)]

    for _, row in param_group.iterrows():
        loi = row["Loi"]
        methode = row["Méthode"]

        try:
            # ======================= 2. Construire la distribution =======================
            if loi == "Exponentielle":
                lmbda = row["lambda_"]
                dist = expon(scale=1/lmbda)

            elif loi == "Gamma":
                k = row["k"]
                theta = row["theta"]
                dist = gamma(a=k, scale=theta)

            elif loi == "Lognormale":
                mu_ln = row["mu_ln"]
                sigma_ln = row["sigma_ln"]
                dist = lognorm(s=sigma_ln, scale=np.exp(mu_ln))

            elif loi == "Gumbel":
                mu = row["mu_gumbel"]
                beta = row["beta_gumbel"]
                dist = gumbel_r(loc=mu, scale=beta)

            elif loi == "Weibull 2P":
                alpha = row["alpha"]
                beta = row["beta"]
                dist = weibull_min(c=beta, scale=alpha)

            elif loi == "Weibull 3P":
                alpha = row["alpha"]
                beta = row["beta"]
                gamma_val = row["gamma"]
                dist = weibull_min(c=beta, scale=alpha, loc=gamma_val)

            else:
                continue  # Loi non reconnue

            # ======================= 3. Tests d'adéquation =======================

            # K-S test
            ks_stat, ks_pvalue = kstest(tbf, dist.cdf)

            # A-D test (remarque : certains types ne sont pas supportés → contournement)
            try:
                ad_test = anderson(tbf, dist.name if hasattr(dist, 'name') else 'expon')
                ad_stat = ad_test.statistic
            except:
                ad_stat = np.nan

            # ======================= 4. Graphes QQ et PP =======================

            sorted_tbf = np.sort(tbf)
            n = len(tbf)
            prob = np.arange(1, n+1) / (n + 1)

            # QQ-Plot
            theo_quantiles = dist.ppf(prob)
            plt.figure()
            plt.scatter(theo_quantiles, sorted_tbf, color='blue')
            plt.plot([min(theo_quantiles), max(theo_quantiles)],
                     [min(theo_quantiles), max(theo_quantiles)], color='red', linestyle='--')
            plt.title(f"QQ-Plot - {site} - {composant} - {loi} ({methode})")
            plt.xlabel("Quantiles théoriques")
            plt.ylabel("Quantiles empiriques")
            qq_path = f"graphes_validation/QQ_{site}_{composant}_{loi}_{methode}.png".replace(" ", "_")
            plt.savefig(qq_path)
            plt.close()

            # PP-Plot
            theo_cdf = dist.cdf(sorted_tbf)
            plt.figure()
            plt.plot(prob, theo_cdf, 'o', color='green')
            plt.plot([0, 1], [0, 1], 'r--')
            plt.title(f"PP-Plot - {site} - {composant} - {loi} ({methode})")
            plt.xlabel("Probabilités empiriques")
            plt.ylabel("Probabilités théoriques")
            pp_path = f"graphes_validation/PP_{site}_{composant}_{loi}_{methode}.png".replace(" ", "_")
            plt.savefig(pp_path)
            plt.close()

            # ======================= 5. Stockage des résultats =======================
            validation_resultats.append({
                "Site": site,
                "Composant": composant,
                "Loi": loi,
                "Méthode": methode,
                "KS_Stat": ks_stat,
                "KS_pval": ks_pvalue,
                "AD_Stat": ad_stat,
                "QQ_plot": qq_path,
                "PP_plot": pp_path
            })

        except Exception as e:
            print(f"[Erreur] {site}-{composant}-{loi}: {e}")
            continue

# ======================= 6. Sortie Excel =======================

df_validation = pd.DataFrame(validation_resultats)
# ======================= 7. Classement des lois =======================

# Calcul du score global (à minimiser)
df_validation["Score_Global"] = df_validation["KS_Stat"] + df_validation["AD_Stat"]

# Extraire les 3 meilleures lois par site/composant
top3 = (
    df_validation
    .sort_values(["Site", "Composant", "Score_Global"])
    .groupby(["Site", "Composant"])
    .head(3)
    .copy()
)

# Ajouter un rang (1er, 2e, 3e)
top3["Classement"] = top3.groupby(["Site", "Composant"])["Score_Global"].rank(method="first")

# ======================= 9. Résumé Meilleure Loi =======================

# =================== 1. Charger les paramètres depuis l'étape précédente ===================

df_parametres = pd.read_excel(r"C:\Users\COMPUTER\Parametres_Fiabilite_Sans_MTBF.xlsx")

# =================== 2. Joindre les paramètres à best_laws ===================

# On suppose que best_laws existe déjà
# Faire la jointure sur les colonnes clés
# Sélectionner la meilleure loi (rang 1) pour chaque composant/site
best_laws = top3[top3["Classement"] == 1].copy()

df_best_with_params = pd.merge(
    best_laws,
    df_parametres,
    on=["Site", "Composant", "Loi", "Méthode"],
    how="left"
)

# =================== 3. Préparer les colonnes pour le résumé ===================

colonnes_resumes = [
    "Site", "Composant", "Loi", "Méthode",
    "alpha", "beta", "gamma",
    "k", "theta",
    "mu_ln", "sigma_ln",
    "mu_gumbel", "beta_gumbel",
    "lambda_"
]

# Nettoyer les NaN → remplacer par chaîne vide pour affichage clair
df_resume = df_best_with_params[colonnes_resumes].fillna("")

# =================== 4. Export Excel avec les 3 feuilles ===================

with pd.ExcelWriter("Validation_Lois_Fiabilite.xlsx", engine="openpyxl", mode="w") as writer:
    df_validation.to_excel(writer, sheet_name="Résultats Tests", index=False)
    top3.to_excel(writer, sheet_name="Classement Top 3", index=False)
    df_resume.to_excel(writer, sheet_name="Résumé Meilleure Loi", index=False)

print("✅ Résumé Meilleure Loi mis à jour avec les paramètres complets.")
