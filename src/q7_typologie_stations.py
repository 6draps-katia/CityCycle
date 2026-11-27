import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from data_loader import get_prepared_dataset
from features import flux_station_heure, profil_stations


def analyse_stations():
  # Charger les données nettoyées des stations
  df_stations, df_trajets, _ = get_prepared_dataset()

  # Récupération des variables de flux avec features.py
  flux = flux_station_heure(df_trajets, df_stations)
  df_profil = profil_stations(flux, df_stations)

  # Choisir les 3 variables d'exploitation et standardisation
  cols = ["capacite", "altitude_m", "interventions_mois"]
  X = StandardScaler().fit_transform(df_stations[cols])

  # Calcul du score de silhouette et l'inertie
  for k in range(2, 6):
    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    labels = km.fit_predict(X)
    score = silhouette_score(X, labels)
    print(
        f"Pour k={k} le score de silouhette est : {score:.2f} | inertie :"
        f" {km.inertia_:.1f}"
    )

  # Regrouper les stations en 3 clusters avec K-Means
  kmeans = KMeans(n_clusters=3, n_init=15, random_state=42)
  df_stations["cluster"] = kmeans.fit_predict(X)
  df_profil["cluster"] = df_stations["cluster"].values

  # Résumer chaque cluster par moyenne et effectif
  print("Caractéristiques par cluster")
  print(
      df_profil.groupby("cluster")
      .agg(
          nb_stations=("station", "count"),
          capacite_moy=("capacite", "mean"),
          altitude_moy=("altitude_m", "mean"),
          interventions_moy=("interventions_mois", "mean"),
          dep_matin_moy=("dep_matin", "mean"),
          arr_matin_moy=("arr_matin", "mean"),
          taux_vidage_moy=("taux_vidage", "mean"),
      )
      .round(1)
  )

  print("\nRépartition par type de quartier")
  print(pd.crosstab(df_stations["cluster"], df_stations["type_quartier"]))


  couleurs = {
      0: "#2E9E6B",
      1: "#3B7EA1",
      2: "#F38B2B",
  }

  fig, ax = plt.subplots(figsize=(10, 6))

  for c in [0, 1, 2]:
    subset = df_stations[df_stations["cluster"] == c]
    x = subset["capacite"]
    y = subset["interventions_mois"]


    ellipse = Ellipse(
        xy=(x.mean(), y.mean()),
        width=max(x.std() * 3.2, 4.0),
        height=max(y.std() * 3.2, 3.0),
        edgecolor=couleurs[c],
        facecolor=couleurs[c],
        alpha=0.14,
        linestyle="--",
        linewidth=1.5,
        zorder=1,
    )
    ax.add_patch(ellipse)

    # Points du cluster
    ax.scatter(
        x,
        y,
        color=couleurs[c],
        s=65,
        label=f"Cluster {c} (n={len(subset)})",
        zorder=2,
    )

  centroides = df_stations.groupby("cluster")[
      ["capacite", "interventions_mois"]
  ].mean()
  ax.scatter(
      centroides["capacite"],
      centroides["interventions_mois"],
      marker="X",
      s=180,
      color="#1B365D",
      edgecolor="white",
      linewidth=1.2,
      label="Centroïdes",
      zorder=3,
  )

  ax.set_title(
      "Les types de stations (Capacité vs Interventions/mois)",
      fontsize=12,
      fontweight="bold",
      pad=12,
  )
  ax.set_xlabel("Capacité (nombre de bornes)", fontsize=11)
  ax.set_ylabel("Interventions par mois", fontsize=11)
  ax.grid(True, alpha=0.35)
  ax.legend(frameon=True, facecolor="white", framealpha=0.95, loc="upper right")

  plt.tight_layout()
  plt.show()

  return df_profil


def run():
  return analyse_stations()


if __name__ == "__main__":
  run()