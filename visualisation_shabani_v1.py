import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.stats import expon, gamma, lognorm, gumbel_r, weibull_min

# ===== STYLE DE VISUALISATION =====
plt.rcParams.update({
    "font.family": "Consolas",
    "text.usetex": False,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "legend.fontsize": 10
})

# ===== CHARGEMENT DES DONNÉES =====
df_best = pd.read_excel(r"C:\Users\COMPUTER\Validation_Lois_Fiabilite.xlsx", sheet_name="Résumé Meilleure Loi")
df_best = df_best.replace(r'^\s*$', np.nan, regex=True)

# ===== PRÉPARATION =====
os.makedirs("figures_individuelles", exist_ok=True)
os.makedirs("figures_par_site", exist_ok=True)
t = np.linspace(0.01, 4000, 500)

# Dictionnaire pour regrouper les figures par site
figures_sites = {}

# ===== TRAITEMENT PAR COMPOSANT =====
for (site, composant), row in df_best.groupby(["Site", "Composant"]):
    loi = row["Loi"].values[0]
    params = row.iloc[0]

    try:
        # Sélection de la distribution
        if loi == "Exponentielle":
            dist = expon(scale=1 / float(params["lambda_"]))
        elif loi == "Gamma":
            dist = gamma(a=float(params["k"]), scale=float(params["theta"]))
        elif loi == "Lognormale":
            dist = lognorm(s=float(params["sigma_ln"]), scale=np.exp(float(params["mu_ln"])))
        elif loi == "Gumbel":
            dist = gumbel_r(loc=float(params["mu_gumbel"]), scale=float(params["beta_gumbel"]))
        elif loi == "Weibull 2P":
            dist = weibull_min(c=float(params["beta"]), scale=float(params["alpha"]))
        elif loi == "Weibull 3P":
            dist = weibull_min(c=float(params["beta"]), scale=float(params["alpha"]), loc=float(params["gamma"]))
        else:
            continue

        # Fonctions
        R_t = dist.sf(t)
        f_t = dist.pdf(t)
        lambda_t = np.divide(f_t, R_t, out=np.zeros_like(f_t), where=(R_t > 0))

        # Création de la figure
        fig, axs = plt.subplots(3, 1, figsize=(8, 10), sharex=True)
        fig.suptitle(f"{site} - {composant} ({loi})", fontsize=14, weight='bold')

        axs[0].plot(t, R_t, color="blue")
        axs[0].set_ylabel(r"$R(t)$")
        axs[0].grid(True)

        axs[1].plot(t, f_t, color="green")
        axs[1].set_ylabel(r"$f(t)$")
        axs[1].grid(True)

        axs[2].plot(t, lambda_t, color="red")
        axs[2].set_ylabel(r"$\lambda(t)$")
        axs[2].set_xlabel(r"$t$ (min)")
        axs[2].grid(True)

        fig.tight_layout(rect=[0, 0, 1, 0.95])

        # Sauvegarde individuelle
        file_name = f"{site}_{composant}_{loi}_courbes.png".replace(" ", "_")
        path_fig = os.path.join("figures_individuelles", file_name)
        fig.savefig(path_fig)
        plt.close(fig)

        # Ajout pour regroupement par site
        if site not in figures_sites:
            figures_sites[site] = []
        figures_sites[site].append((composant, path_fig))

    except Exception as e:
        print(f"[Erreur pour {site} - {composant} ({loi})] : {e}")

# ===== FIGURES PAR SITE =====
for site, composants_figures in figures_sites.items():
    if not composants_figures:
        continue  # Aucun graphique pour ce site

    nb = len(composants_figures)
    cols = 2
    rows = int(np.ceil(nb / cols))
    fig, axs = plt.subplots(rows, cols, figsize=(14, 5 * rows))
    axs = axs.flatten()

    for ax, (composant, path_img) in zip(axs, composants_figures):
        img = plt.imread(path_img)
        ax.imshow(img)
        ax.axis("off")
        ax.set_title(composant)

    for i in range(len(composants_figures), len(axs)):
        axs[i].axis("off")

    # ✅ Correction du titre sans LaTeX complexe
    fig.suptitle(f"Courbes R(t), f(t), λ(t) - Site {site}", fontsize=16, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    site_file = f"{site}_grille_composants.png".replace(" ", "_")
    fig.savefig(os.path.join("figures_par_site", site_file))
    plt.close(fig)
