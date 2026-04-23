"""
Model loader for the inference API.
Loads the champion model and scaler from disk.
"""
import json
import os
import joblib
import numpy as np


MODEL_REGISTRY = os.environ.get("MODEL_REGISTRY", "/tmp/fraud_model_registry")
LOCAL_MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results", "models")


class ModelLoader:
    """Singleton model loader with lazy initialisation."""

    def __init__(self):
        self._model = None
        self._metadata = None
        self._model_type = "xgboost"
        self._selected_indices = None
        self._loaded = False

    def load(self) -> bool:
        """Try to load champion model from registry, then fall back to results/models/."""
        # Try champion registry first
        champion_path = os.path.join(MODEL_REGISTRY, "champion", "model.joblib")
        meta_path     = os.path.join(MODEL_REGISTRY, "champion", "metadata.json")

        if not os.path.exists(champion_path):
            # Fall back to latest model in results/models/
            for name in ["xgboost_model.joblib", "lightgbm_model.joblib", "rf_hybrid_model.joblib"]:
                fallback = os.path.join(LOCAL_MODEL_DIR, name)
                if os.path.exists(fallback):
                    champion_path = fallback
                    break

        if not os.path.exists(champion_path):
            print("[model_loader] No model found – API will return dummy predictions")
            return False

        self._model = joblib.load(champion_path)
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                self._metadata = json.load(f)
            self._model_type = self._metadata.get("model_type", "xgboost")

        if isinstance(self._model, dict) and self._model.get("type") == "rf_hybrid":
            self._model_type = "rf_hybrid"
            self._selected_indices = self._model.get("selected_indices")

        self._loaded = True
        print(f"[model_loader] Loaded model: {self._model_type} from {champion_path}")
        return True

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def model_type(self) -> str:
        return self._model_type

    @property
    def metadata(self) -> dict:
        return self._metadata or {}

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return fraud probability for each row in X."""
        if not self._loaded:
            # Return dummy probability
            return np.full(len(X), 0.1)

        if self._model_type == "rf_hybrid":
            if self._selected_indices is not None:
                n_features = len(self._selected_indices)
                if X.shape[1] >= n_features:
                    X = X[:, self._selected_indices]
                else:
                    # Pad if needed
                    X = np.pad(X, ((0, 0), (0, n_features - X.shape[1])))
            return self._model["xgb_model"].predict_proba(X)[:, 1]
        else:
            return self._model.predict_proba(X)[:, 1]


# Module-level singleton
loader = ModelLoader()
