# Probl-meinverse
# Problème inverse en thermique transitoire

## Description
Ce projet porte sur la résolution d’un problème inverse basé sur l’équation de la chaleur 1D en régime transitoire.  
L’objectif est de reconstruire les conditions initiales d’un système à partir de mesures de température aux frontières.

## Méthodologie
- Formulation du problème direct et du problème inverse
- Discrétisation numérique (méthode des éléments finis / Euler)
- Construction d’une matrice d’observation reliant les mesures aux inconnues
- Résolution du problème inverse par moindres carrés
- Régularisation de Tikhonov pour stabiliser la solution

## Résultats
- Reconstruction des conditions initiales à partir de données bruitées
- Mise en évidence du caractère mal posé du problème inverse
- Amélioration de la stabilité grâce à la régularisation
- Analyse de l’influence du bruit sur la qualité de reconstruction

## Technologies utilisées
- MATLAB
- Méthodes numériques
- Problèmes inverses
- Régularisation (Tikhonov)

## Remarque
Les données utilisées peuvent inclure du bruit afin de tester la robustesse des méthodes d’identification.
