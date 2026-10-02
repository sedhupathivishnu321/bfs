import numpy as np
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor


def flat_feats(X):
    """X (N,L,F) window-relative inputs -> compact tabular features for classical ML."""
    N = len(X)
    last = X[:, -8:, :].reshape(N, -1)
    stat = np.concatenate([X.mean(1), X.std(1), X[:, -1] - X[:, -12], X[:, -1] - X[:, -24]], 1)
    return np.concatenate([last, stat], 1)


class Persistence:
    def fit(self, X, Y): self.d = Y.shape[1]; return self
    def predict(self, X): return np.zeros((len(X), self.d), np.float32)


class AR:
    """Ridge-regularised linear autoregression on the full window (own+ctx)."""
    def fit(self, X, Y): self.m = Ridge(alpha=10.0).fit(X.reshape(len(X), -1), Y); return self
    def predict(self, X): return self.m.predict(X.reshape(len(X), -1))


class GBM:
    def __init__(self, seed=0): self.seed = seed
    def fit(self, X, Y):
        F = flat_feats(X)
        self.ms = [HistGradientBoostingRegressor(max_iter=150, learning_rate=0.06, max_depth=4, random_state=self.seed).fit(F, Y[:, j]) for j in range(Y.shape[1])]
        return self
    def predict(self, X):
        F = flat_feats(X); return np.stack([m.predict(F) for m in self.ms], 1)
