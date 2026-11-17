import numpy as np


class LinearRegressionClassifier:
    """Doğrusal regresyonu 0/1 etiketlere uydurup 0.5 eşiğiyle sınıflandırma yapar."""

    def __init__(self, method="normal", learning_rate=0.05, n_iters=5000, threshold=0.5):
        self.method = method              # "normal" (kapalı form) veya "gd" (gradyan inişi)
        self.learning_rate = learning_rate
        self.n_iters = n_iters
        self.threshold = threshold
        self.w = None
        self.loss_history = []

    def _add_bias(self, X):
        # X'in başına 1'lerden oluşan bir sütun ekle -> w[0] bias terimi olur
        return np.hstack([np.ones((X.shape[0], 1)), X])

    def fit(self, X, y):
        Xb = self._add_bias(X)
        y = y.astype(float)

        if self.method == "normal":
            # w = (X^T X)^(-1) X^T y
            self.w = np.linalg.pinv(Xb.T @ Xb) @ Xb.T @ y

        elif self.method == "gd":
            m, n = Xb.shape
            self.w = np.zeros(n)
            self.loss_history = []
            for _ in range(self.n_iters):
                error = Xb @ self.w - y                  # tahmin - gerçek
                self.loss_history.append(np.mean(error ** 2))  # MSE
                gradient = (2 / m) * Xb.T @ error
                self.w -= self.learning_rate * gradient

        else:
            raise ValueError("method 'normal' veya 'gd' olmalı")

        return self

    def predict_score(self, X):
        # Ham sürekli çıktı (0-1 aralığında olmak zorunda değil!)
        return self._add_bias(X) @ self.w

    def predict(self, X):
        return (self.predict_score(X) >= self.threshold).astype(int)


if __name__ == "__main__":
    from preprocessing import prepare_data

    X_train, X_test, y_train, y_test = prepare_data()

    model_ne = LinearRegressionClassifier(method="normal").fit(X_train, y_train)
    model_gd = LinearRegressionClassifier(method="gd").fit(X_train, y_train)

    print("Normal denklem ağırlıkları:", np.round(model_ne.w, 4))
    print("Gradyan inişi ağırlıkları: ", np.round(model_gd.w, 4))
    print("İki çözüm arasındaki en büyük fark:", np.max(np.abs(model_ne.w - model_gd.w)))

    print("\nGD ilk MSE:", round(model_gd.loss_history[0], 4),
          " son MSE:", round(model_gd.loss_history[-1], 4))

    for isim, model in [("Normal denklem", model_ne), ("Gradyan inişi", model_gd)]:
        egitim_acc = np.mean(model.predict(X_train) == y_train)
        test_acc = np.mean(model.predict(X_test) == y_test)
        print(f"\n{isim}: eğitim doğruluğu = {egitim_acc:.4f}, test doğruluğu = {test_acc:.4f}")

    skor = model_ne.predict_score(X_test)
    print(f"\nTest skorlarının aralığı: min = {skor.min():.3f}, max = {skor.max():.3f}")
    print("0'ın altındaki skor sayısı:", np.sum(skor < 0), " 1'in üstündeki:", np.sum(skor > 1))