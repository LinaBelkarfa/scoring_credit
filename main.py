#!/usr/bin/env python3
"""
Pipeline principal pour le projet de scoring crédit.
Ce script orchestre le chargement des données, le preprocessing,
l'entraînement des modèles et l'évaluation.
"""

import os
import sys
import logging

# Configuration du logging pour suivre proprement ce qui se passe dans le terminal
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


def run_pipeline():
    logger.info("===== DÉMARRAGE DU PIPELINE DE SCORING CRÉDIT =====")
    
    # 0. Vérification des chemins de données
    path_data = "data/input/data_scoring_credit.csv"
    if not os.path.exists(path_data):
        logger.error(f"Le fichier de données est introuvable au chemin : {path_data}")
        logger.info("Veuillez placer votre fichier CSV dans le dossier 'data/'")
        return

    # 1. Étape de Preprocessing (Pipeline de données)
    logger.info("Étape 1 : Chargement, nettoyage des données, séparation en data_train, data_test, data_pred")
    # TODO: Importer et appeler tes fonctions de src/preprocessing.py
    # X_train, X_test, y_train, y_test = preparer_donnees(path_data)
    
    # 2. Étape d'Entraînement du modèle
    logger.info("Étape 2 : Entraînement du modèle de Machine Learning")
    # TODO: Importer et appeler tes fonctions de src/training.py
    # model = entrainer_modele(X_train, y_train)

    # 3. Étape d'Évaluation
    logger.info("Étape 3 : Évaluation des performances et sélection du meilleur modèle")
    # TODO: Calculer les métriques asymétriques liées au coût du risque de crédit

    # 4. Étape de prévision 
    logger.info("Étape 4 : Émission et évaluation des prévisions sur data_pred")
    # TODO: Faire les prédictions

    # 5. Explicabilité
    logger.info("Étape 5 : Calculer l'explicabilité des prévisions faites sur data_pred")
    # TODO: Générer les SHAP values et graphiques d'explicabilité pour data_pred
    
    logger.info("===== PIPELINE EXÉCUTÉ AVEC SUCCÈS =====")


if __name__ == "__main__":
    # Ce bloc garantit que le script ne s'exécute que s'il est lancé directement
    run_pipeline()