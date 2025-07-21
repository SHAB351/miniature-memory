# ------------------------------
# 1. Importation des bibliothèques
# ------------------------------
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams

# ------------------------------
# 2. Chargement et préparation des données
# ------------------------------
# Lire le fichier Excel (à adapter selon votre chemin local)
df = pd.read_excel(r"C:\Users\COMPUTER\Documents\FIABILITE\DONNEES TTR ET TBF 2.xlsx", sheet_name="Données TTR")

# Nettoyage des colonnes inutiles
df = df.drop(columns=[col for col in df.columns if "Unnamed" in col], errors='ignore')

# ------------------------------
# 3. Regroupement par composant et calcul du TTR total
# ------------------------------
abc_df = df.groupby("Composant")["TTR (minutes)"].sum().reset_index()
abc_df = abc_df.rename(columns={"TTR (minutes)": "TTR_total"})

# ------------------------------
# 4. Calcul du pourcentage et du pourcentage cumulé
# ------------------------------
abc_df["%"] = 100 * abc_df["TTR_total"] / abc_df["TTR_total"].sum()
abc_df = abc_df.sort_values(by="TTR_total", ascending=False).reset_index(drop=True)
abc_df["% cumulé"] = abc_df["%"].cumsum()

# ------------------------------
# 5. Classification ABC
# ------------------------------
def classer_abc(pct_cumule):
    if pct_cumule <= 80:
        return "A"
    elif pct_cumule <= 95:
        return "B"
    else:
        return "C"

abc_df["Classe ABC"] = abc_df["% cumulé"].apply(classer_abc)

# ------------------------------
# 6. Exportation des résultats
# ------------------------------
abc_df.to_excel("Analyse_ABC_TTR.xlsx", index=False)

# ------------------------------
# 7. Configuration du style graphique (Consolas + LaTeX)
# ------------------------------
rcParams.update({
    "font.family": "Consolas",
    "text.usetex": False,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.figsize": (10, 6)
})

# ------------------------------
# 8. Création du graphique de Pareto
# ------------------------------
fig, ax1 = plt.subplots()

# Barres : TTR_total
ax1.bar(abc_df["Composant"], abc_df["TTR_total"], color='skyblue', label=r"\textbf{TTR total}")
ax1.set_ylabel(r"\textbf{TTR total (minutes)}", fontsize=10)
ax1.set_xlabel(r"\textbf{Composants}", fontsize=10)
ax1.tick_params(axis='x', rotation=45)

# Courbe cumulative
ax2 = ax1.twinx()
ax2.plot(abc_df["Composant"], abc_df["% cumulé"], color='red', marker='o', label=r"\textbf{\% cumulé}")
ax2.set_ylabel(r"\textbf{\% cumulé}", fontsize=10)
ax2.set_ylim(0, 110)

# Lignes de seuils ABC
ax2.axhline(80, color='green', linestyle='--', linewidth=1)
ax2.axhline(95, color='orange', linestyle='--', linewidth=1)
ax2.text(len(abc_df) - 1, 81, r"$80\%$ seuil~A", color="green", fontsize=9, ha='right')
ax2.text(len(abc_df) - 1, 96, r"$95\%$ seuil~B", color="orange", fontsize=9, ha='right')

# Légendes et titre
fig.legend(loc="upper center", bbox_to_anchor=(0.5, 1.05), ncol=2, fontsize=9)
plt.title(r"\textbf{Analyse ABC des composants basée sur le TTR}", pad=30)
plt.tight_layout()
plt.grid(True, axis='y', linestyle='--', linewidth=0.5)
plt.show()
