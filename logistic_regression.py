import numpy as np


class LogisticRegression:
    """Gradyan inişi ile eğitilen ikili lojistik regresyon."""

    def __init__(self, learning_rate=0.1, n_iters=5000, threshold=0.5):
        self.learning_rate = learning_rate
        self.n_iters = n_iters
        self.threshold = threshold
        self.w = None
        self.loss_history = []

    def _add_bias(self, X):
        return np.hstack([np.ones((X.shape[0], 1)), X])

    @staticmethod
    def _sigmoid(z):
        # Çok büyük |z| değerlerinde exp taşmasın diye sınırla
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))

    @staticmethod
    def _bce_loss(y, p):
        # log(0) hatasını önlemek için p'yi 0 ve 1'e tam eşit olmaktan uzak tut
        eps = 1e-15
        p = np.clip(p, eps, 1 - eps)
        return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))

    def fit(self, X, y):
        Xb = self._add_bias(X)
        y = y.astype(float)
        m, n = Xb.shape
        self.w = np.zeros(n)
        self.loss_history = []

        for _ in range(self.n_iters):
            p = self._sigmoid(Xb @ self.w)          # olasılık tahminleri
            self.loss_history.append(self._bce_loss(y, p))
            gradient = (1 / m) * Xb.T @ (p - y)
            self.w -= self.learning_rate * gradient

        return self

    def predict_proba(self, X):
        return self._sigmoid(self._add_bias(X) @ self.w)

    def predict(self, X):
        return (self.predict_proba(X) >= self.threshold).astype(int)


if __name__ == "__main__":
    from preprocessing import prepare_data

    X_train, X_test, y_train, y_test = prepare_data()

    model = LogisticRegression().fit(X_train, y_train)

    print("Ağırlıklar:", np.round(model.w, 4))
    print("\nİlk BCE kaybı:", round(model.loss_history[0], 4),
          " son BCE kaybı:", round(model.loss_history[-1], 4))
    print("Son 100 iterasyondaki kayıp değişimi:",
          model.loss_history[-100] - model.loss_history[-1])

    egitim_acc = np.mean(model.predict(X_train) == y_train)
    test_acc = np.mean(model.predict(X_test) == y_test)
    print(f"\nEğitim doğruluğu = {egitim_acc:.4f}, test doğruluğu = {test_acc:.4f}")

    p = model.predict_proba(X_test)
    print(f"\nTest olasılıklarının aralığı: min = {p.min():.3f}, max = {p.max():.3f}")
    print("0'ın altındaki:", np.sum(p < 0), " 1'in üstündeki:", np.sum(p > 1))