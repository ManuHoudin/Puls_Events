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



