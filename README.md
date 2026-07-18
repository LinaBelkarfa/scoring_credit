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
*   **Le F2-Score** : Une variante du F1-score qui donne plus d'importance au Rappel qu'à la Précision, idéale pour notre cas de figure. Formule mathématique du $F_2\text{-Score}$ : 
$$F_2 = \frac{5 \times \text{VP}}{(5 \times \text{VP}) + (4 \times \text{FN}) + \text{FP}}$$
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
```


# Dictionnaire des Données (Data Dictionary)

Le dataset contient les informations sociodémographiques et financières des clients. L'objectif est de prédire la variable cible `default`.

### Identification & Structure
*   **`ncust`** : Numéro d'identification interne du client (identifiant court).
    *   *Type :* Numérique / Identifiant.
    *   *Valeurs :* Entiers de `1919` à `4809`.
    *   *⚠️ Note de Data Cleaning :* À supprimer avant l'entraînement car ce n'est pas une variable prédictive.
*   **`customer`** : Identifiant global du client dans le système bancaire (identifiant long).
    *   *Type :* Numérique / Identifiant[cite: 1].
    *   *Valeurs :* Entiers de `10012` à `453777`[cite: 1].
    *   *⚠️ Note de Data Cleaning :* À supprimer également car ce n'est pas une variable prédictive.

*   **`branch`** : Code de l'agence bancaire ou zone géographique de rattachement du client.
    *   *Type :* Catégorielle (doit être convertie en `factor` / `category`).
    *   *Valeurs :* Entiers (ex: agence `3` à `91`).

### Profil Sociodémographique
*   **`age`** : Âge du client.
    *   *Type :* Numérique continu (Entier).
    *   *Valeurs :* De `18` à `79` ans (Médiane : `31` ans).
*   **`ed`** : Niveau d'études atteint par le client.
    *   *Type :* Catégorielle ordinale (`ordered factor`).
    *   *Valeurs :* `Niveau bac` < `Bac+2` < `Bac+3` < `Bac+4` < `Bac+5 et plus`.
*   **`employ`** : Ancienneté professionnelle (nombre d'années passées chez l'employeur actuel).
    *   *Type :* Numérique continu.
    *   *Valeurs :* De `0` à `63` ans.
*   **`address`** : Stabilité résidentielle (nombre d'années passées à l'adresse actuelle).
    *   *Type :* Numérique continu.
    *   *Valeurs :* De `0` à `34` ans.

### Profil Financier & Endettement
*   **`income`** : Revenu annuel du client (généralement exprimé en milliers d'unités, par exemple 12€ = 12000€).
    *   *Type :* Numérique continu.
    *   *Valeurs :* De `12.0` à `1079.0` (Médiane : `39.0`).
*   **`debtinc`** (*Debt-to-Income ratio*) : Taux d'endettement global (pourcentage du revenu mensuel/annuel consacré au remboursement des dettes).
    *   *Type :* Numérique continu (Pourcentage).
    *   *Valeurs :* De `0.0%` à `40.7%` (Médiane : `8.5%`).
*   **`creddebt`** (*Credit Debt*) : Encours de la dette liée aux cartes de crédit et crédits à la consommation.
    *   *Type :* Numérique continu.
    *   *Valeurs :* De `0.0` à `35.97`.
*   **`othdebt`** (*Other Debt*) : Encours des autres types de dettes privées ou bancaires (prêts auto, personnels, etc.).
    *   *Type :* Numérique continu.
    *   *Valeurs :* De `0.00` à `63.47`.

### Target (Variable Cible)
*   **`default`** : Statut de défaut de paiement du client (Variable à prédire).
    *   *Type :* Catégorielle binaire.
    *   *Valeurs :* 
        *   `Oui` (Classe Positive) : Le client a été en défaut de paiement (risque élevé).
        *   `Non` (Classe Négative) : Le client a remboursé son crédit normalement.
    *   *Distribution initiale :* ~37,3% de `Oui` pour ~62,7% de `Non`.

## Cohérence Mathématique des Variables Financières

Il existe une relation mathématique exacte entre le taux d'endettement (`debtinc`), le revenu (`income`) et les encours de dettes (`creddebt` et `othdebt`). La variable `debtinc` (*Debt-to-Income ratio*) n'est pas une donnée brute, mais le résultat direct du calcul de la charge de la dette totale par rapport au revenu.

Tous les montants financiers (`income`, `creddebt`, `othdebt`) sont exprimés **en milliers d'euros (k€)**.

#### 📝 Formule :
$$\text{debtinc} = \frac{\text{creddebt} + \text{othdebt}}{\text{income}} \times 100$$

#### 🔍 Exemple concret (Ligne 1 du dataset) :
Si l'on prend les données du premier client du jeu de données :
*   **`income`** : $44$ ($44\ 000\text{ €}$)
*   **`creddebt`** : $2.99$ ($2\ 990\text{ €}$)
*   **`othdebt`** : $4.79$ ($4\ 790\text{ €}$)

1. **Calcul de la dette totale :** 
   $$2.99 + 4.79 = 7.78\text{ k€}$$
2. **Calcul du ratio d'endettement :** 
   $$\frac{7.78}{44} \times 100 = 17.68\%$$

Ce résultat correspond (après arrondi) à la valeur stockée dans la colonne **`debtinc`** ($17.7\%$).

> 💡 **Note pour le Feature Engineering :** Cette colinéarité parfaite implique qu'il faudra évaluer la pertinence de conserver les trois variables financières conjointement lors de la phase d'entraînement de certains modèles sensibles à la multi-colinéarité (comme la régression logistique ou le SVM linéaire).