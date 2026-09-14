from types import SimpleNamespace
from unittest.mock import Mock

import pandas as pd
import pytest

from src.services.EmbeddingService import EmbeddingService


def test_embed_dataframe_ajoute_les_embeddings():
    client = Mock()

    client.embeddings.create.return_value = (
        SimpleNamespace(
            data=[
                SimpleNamespace(
                    index=0,
                    embedding=[0.1, 0.2, 0.3],
                ),
                SimpleNamespace(
                    index=1,
                    embedding=[0.4, 0.5, 0.6],
                ),
            ]
        )
    )

    service = EmbeddingService(
        client=client,
        batch_size=100,
    )

    df = pd.DataFrame(
        {
            "texte_chunk": [
                "Premier événement",
                "Deuxième événement",
            ]
        }
    )

    result = service.embed_dataframe(df)

    assert "embedding" in result.columns

    assert result["embedding"].tolist() == [
        [0.1, 0.2, 0.3],
        [0.4, 0.5, 0.6],
    ]

    assert len(result) == 2

    client.embeddings.create.assert_called_once_with(
        model="mistral-embed",
        inputs=[
            "Premier événement",
            "Deuxième événement",
        ],
    )


def test_embeddings_sont_remis_dans_le_bon_ordre():
    client = Mock()

    # Mistral retourne volontairement les résultats
    # dans le désordre.
    client.embeddings.create.return_value = (
        SimpleNamespace(
            data=[
                SimpleNamespace(
                    index=1,
                    embedding=[0.4, 0.5],
                ),
                SimpleNamespace(
                    index=0,
                    embedding=[0.1, 0.2],
                ),
            ]
        )
    )

    service = EmbeddingService(
        client=client
    )

    df = pd.DataFrame(
        {
            "texte_chunk": [
                "Premier",
                "Deuxième",
            ]
        }
    )

    result = service.embed_dataframe(df)

    assert result["embedding"].tolist() == [
        [0.1, 0.2],
        [0.4, 0.5],
    ]


def test_embed_dataframe_respecte_le_batch_size():
    client = Mock()

    client.embeddings.create.side_effect = [
        SimpleNamespace(
            data=[
                SimpleNamespace(
                    index=0,
                    embedding=[0.1],
                ),
                SimpleNamespace(
                    index=1,
                    embedding=[0.2],
                ),
            ]
        ),
        SimpleNamespace(
            data=[
                SimpleNamespace(
                    index=0,
                    embedding=[0.3],
                ),
            ]
        ),
    ]

    service = EmbeddingService(
        client=client,
        batch_size=2,
    )

    df = pd.DataFrame(
        {
            "texte_chunk": [
                "Premier",
                "Deuxième",
                "Troisième",
            ]
        }
    )

    result = service.embed_dataframe(df)

    assert result["embedding"].tolist() == [
        [0.1],
        [0.2],
        [0.3],
    ]

    assert client.embeddings.create.call_count == 2

    appels = client.embeddings.create.call_args_list

    assert appels[0].kwargs["inputs"] == [
        "Premier",
        "Deuxième",
    ]

    assert appels[1].kwargs["inputs"] == [
        "Troisième",
    ]


def test_embed_dataframe_vide():
    client = Mock()

    service = EmbeddingService(
        client=client
    )

    df = pd.DataFrame(
        {
            "texte_chunk": []
        }
    )

    result = service.embed_dataframe(df)

    assert len(result) == 0
    assert "embedding" in result.columns

    client.embeddings.create.assert_not_called()


def test_embed_dataframe_nombre_embeddings_incorrect():
    client = Mock()

    client.embeddings.create.return_value = (
        SimpleNamespace(
            data=[
                SimpleNamespace(
                    index=0,
                    embedding=[0.1, 0.2],
                ),
            ]
        )
    )

    service = EmbeddingService(
        client=client
    )

    df = pd.DataFrame(
        {
            "texte_chunk": [
                "Premier",
                "Deuxième",
            ]
        }
    )

    with pytest.raises(
        ValueError,
        match="Le nombre d'embeddings ne correspond",
    ):
        service.embed_dataframe(df)