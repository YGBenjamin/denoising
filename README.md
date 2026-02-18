# The Denoising Project
### Benjamin Yang

Ce projet est la suite logique de mon travail sur la détection de coins de Harris. Après avoir appris à extraire des formes, je me concentre ici sur la **restauration d'image**.

## Pourquoi ce projet ?

En imagerie médicale (MedTech), le bruit est inévitable. Que ce soit en radiographie ou en IRM, la physique du capteur "pollue" toujours le signal original. Mon objectif est de comprendre comment :
1. **Modéliser le bruit** : Comprendre mathématiquement les différents types de perturbations (Gaussien, impulsionnel, etc.).
2. **Filtrer intelligemment** : Découvrir et implémenter des algorithmes capables de nettoyer une image tout en préservant les détails critiques (les bords).

## Démarche

Comme pour mes projets précédents, je privilégie une approche par étape :
- Expérimentations mathématiques en **Notebook** pour comprendre le comportement des bruits.
- Implémentation de filtres (linéaires et non-linéaires) en partant de zéro.
- Comparaison des résultats pour identifier quel filtre est le plus adapté à quel problème.

Ce projet me permet de solidifier mes bases en traitement du signal avant d'attaquer des problématiques de vision par ordinateur plus complexes.
