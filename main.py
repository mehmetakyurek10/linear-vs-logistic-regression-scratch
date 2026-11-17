import os
import numpy as np
import matplotlib.pyplot as plt

from preprocessing import prepare_data
from linear_regression import LinearRegressionClassifier
from logistic_regression import LogisticRegression
from evaluation import (compute_metrics, print_results, plot_confusion_matrices,
                        plot_loss_curves, plot_output_distributions)

os.makedirs("figures", exist_ok=True)

# 1) Veriyi hazırla
X_train, X_test, y_train, y_test = prepare_data()

# 2) Modelleri eğit
lin_model = LinearRegressionClassifier(method="gd").fit(X_train, y_train)
log_model = LogisticRegression().fit(X_train, y_train)

# 3) Aynı test verisi üzerinde tahmin
lin_pred = lin_model.predict(X_test)
log_pred = log_model.predict(X_test)

# 4) Metrikler
results = {
    "Doğrusal Regresyon": compute_metrics(y_test, lin_pred),
    "Lojistik Regresyon": compute_metrics(y_test, log_pred),
}
print_results(results)

farkli = np.sum(lin_pred != log_pred)
print(f"\nİki modelin farklı tahmin yaptığı test örneği sayısı: {farkli} / {len(y_test)}")

# 5) Grafikler
plot_confusion_matrices(results, "figures/confusion_matrices.png")
plot_loss_curves(lin_model.loss_history, log_model.loss_history, "figures/loss_curves.png")
plot_output_distributions(lin_model.predict_score(X_test), log_model.predict_proba(X_test),
                          y_test, "figures/output_distributions.png")

plt.show()