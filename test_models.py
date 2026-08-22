from src.models import (
    FEATURES,
    build_models,
)


def test_models():

    models = build_models()

    assert len(models) >= 2

    for bundle in models:

        assert bundle.name
        assert bundle.model is not None

    assert len(FEATURES) > 0
