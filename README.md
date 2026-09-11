Projet Puls-Events

# Objectif
Créer un Chatbot intelligent capable de répondre aux utilisateurs à propos des évenements culturels à venir. Il devra s'appuyer sur un RAG (Retrieval Augmented Generation).

# Structure du projet
xxx

# Instructions de reproduction
Construction de l'index Faiss : 
Lancer le script de construction de l'index : `python -m scripts.build_index`

Lancement de l'API dans une image docker en local :
Lancer le moteur avec Docker Desktop
Lancer l'image docker : `docker compose build`
Activer le service : `docker compose up`

Lancer l'api en local : `uvicorn src.api:app --reload`
Swagger : http://127.0.0.1:8000/docs


# Optimisations
Optimisation des paramètres du modèle : le mode thinking a été désactivé pour limiter le temps de réponse. Ici il n'est pas nécessaire d'avoir un modèle de raisonnement très poussé.
Optimisation du volume de données envoyé au LLM pour la formulation de la réponse. Travail sur le compactage des dates.
On est passé à un format :
Du 12/06/2026 au 20/12/2026 ; créneaux récurrents : 13:00 - 19:00; 19:00 - 23:00

