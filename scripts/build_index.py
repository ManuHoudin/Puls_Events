from pathlib import Path
import os

from mistralai.client import Mistral
from dotenv import load_dotenv

from src.services.EmbeddingService import EmbeddingService
from src.services.FaissRepository import FaissRepository
from src.services.OpenAgendaClient import OpenAgendaClient
from src.services.EventProcessor import EventProcessor

load_dotenv()


def main():

    mistral_client = Mistral()

    openagenda_client = OpenAgendaClient(
        api_key=os.environ["DASHSCOPE_API_KEY"]
    )

    event_processor = EventProcessor()

    embedding_service = EmbeddingService(
        client=mistral_client,
        batch_size=100,
    )

    faiss_repository = FaissRepository(
        index_path=Path(
            "data/index_faiss/evenements.faiss"
        ),
        metadata_path=Path(
            "data/index_faiss/metadata.parquet"
        ),
    )

    print("Construction de l'index FAISS terminée.")
    # récupération des événements
    # traitement
    # embeddings
    # création de l'index
    # sauvegarde


if __name__ == "__main__":
    main()