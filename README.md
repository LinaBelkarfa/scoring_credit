# Scoring Crédit - Prédiction du Risque de Défaut

Ce projet implémente un pipeline de Machine Learning de bout en bout pour évaluer le risque de crédit et prédire la probabilité de défaut de paiement des agents économiques. L'objectif final est de déployer le modèle le plus performant et le plus sûr sous forme d'API.


## Objectif du Projet

L'enjeu principal de ce projet est la **gestion stricte du risque financier**. Accorder un crédit à un client qui ne remboursera pas (Faux Négatif) représente un coût bien plus élevé pour l'institution que le manque à gagner lié au refus d'un client solvable (Faux Positif).

Dans ce contexte, l'objectif est double :
1. Entraîner et comparer plusieurs modèles de classification (ex: Régression Logistique, Random Forest, XGBoost, LightGBM etc...).
2. Sélectionner le meilleur modèle en fonction de sa capacité à sécuriser les décisions de crédit.

---

## Métriques d'Évaluation Métier

L'**Accuracy (précision globale) est totalement proscrite** pour ce projet car les classes de défaut sont souvent très déséquilibrées (peu de défauts par rapport aux crédits remboursés) et elle ne reflète pas le coût asymétrique des erreurs.

Pour optimiser le modèle, nous nous concentrons sur :
*   **Minimiser les Faux Négatifs (FN)** : Ne surtout pas prédire "solvable" ($0$) pour un agent qui va faire défaut ($1$).
*   **Maximiser les Vrais Négatifs (VN) et Vrais Positifs (VP)** : Identifier correctement les clients sûrs et bloquer à coup sûr les profils à risque.

### Métriques clés ciblées :
*   **Le Rappel (Recall / Sensitivity)** : C'est notre métrique prioritaire. Un rappel élevé garantit que l'on capture un maximum d'agents en défaut. (TP/(TP+FN)). Sur tous les vrais défauts, combien le modèle a-t-il réussi à en bloquer ?
*   **La Precision** : C'est la seconde métrique, sur tous les clients que le modèle prédit en défaut (bloqués), combien allaient vraiment faire défaut ? Une bonne précision évite au modèle d'être trop paranoïaque et de refuser inutilement des crédits à des clients qui auraient pourtant été solvables (les Faux Positifs). (TP/(TP+FP)). 
*   **Le F2-Score** : Une variante du F1-score qui donne deux fois plus d'importance au Rappel qu'à la Précision, idéale pour notre cas de figure.
*   **L'AUC-PR (Area Under the Curve Precision-Recall)** : Contrairement à l'AUC-ROC classique qui peut être trompée par un grand nombre de clients solvables (Vrais Négatifs), l'AUC-PR se concentre uniquement sur la classe minoritaire (le défaut de paiement). Elle calcule l'aire sous la courbe qui combine la Précision et le Rappel pour tous les seuils de décision possibles. Plus ce score est proche de 1, plus le modèle est robuste pour identifier les profils à risque sans générer de faux diagnostics, ce qui en fait notre indicateur global le plus fiable face au fort déséquilibre de nos données.

## Architecture du Projet

Le projet est structuré de la manière suivante pour garantir la modularité et le suivi des sources :

```text
/scoring_credit
├── .devcontainer/       # Configuration de l'environnement Docker isolé
│   ├── devcontainer.json
│   └── Dockerfile
├── data/                # Stockage des données du projet
│   └── Data Projet.csv  # Dataset brut contenant les historiques de crédit
├── src/                 # Code source logique du pipeline IA
│   ├── preprocessing.py # Nettoyage, encodage et feature engineering
│   └── training.py      # Entraînement, tuning et évaluation des modèles
├── main.py              # Point d'entrée de l'application (API FastAPI)
├── README.md            # Documentation du projet
└── requirements.txt     # Dépendances du projet (pandas, scikit-learn...)
```

## Démarrer le Projet

Suivez ces étapes pour lancer votre environnement de développement isolé et installer l'ensemble des bibliothèques :

### 1. Ouvrir le projet dans le Dev Container
1. Assurez-vous que **Docker Desktop** est démarré sur votre machine.
2. Ouvrez le dossier `scoring_credit` avec **VS Code**.
3. Une notification devrait vous proposer de rouvrir le dossier dans un conteneur. Si ce n'est pas le cas, appuyez sur **`F1`**, tapez et sélectionnez : `Dev Containers: Reopen in Container`.
4. Attendez que VS Code termine la construction de l'image Linux isolée.

### 2. Installer les dépendances (Requirements)
Une fois connecté à l'intérieur du conteneur (le terminal affiche `root@...:/workspace#`), ouvrez un terminal dans VS Code (`Ctrl + \``) et exécutez la commande suivante pour installer l'ensemble des packages d'IA fégés (`pandas`, `scikit-learn`, `xgboost`, `lightgbm`, `catboost`, `shap`) :

```bash
pip install -r requirements.txt