from sentence_transformers import SentenceTransformer

_model_instance = None

def get_embedding_model():
    global _model_instance
    if _model_instance is None:
        _model_instance = SentenceTransformer("sentence-transformers/LaBSE", device="cpu")
    return _model_instance

class LazyEmbeddingModel:
    def encode(self, *args, **kwargs):
        return get_embedding_model().encode(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(get_embedding_model(), name)

embedding_model = LazyEmbeddingModel()