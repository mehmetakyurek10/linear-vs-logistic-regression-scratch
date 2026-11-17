import numpy as np
import matplotlib.pyplot as plt


def confusion_matrix(y_true, y_pred):
    """[[TN, FP],
        [FN, TP]] şeklinde 2x2 karmaşıklık matrisi döndürür."""
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    tp = np.sum((y_true == 1) & (y_pred == 1))
    return np.array([[tn, fp], [fn, tp]])


def compute_metrics(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    accuracy = (tp + tn) / cm.sum()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {"cm": cm, "accuracy": accuracy, "precision": precision,
            "recall": recall, "f1": f1}


def print_results(results):
    isimler = list(results.keys())

    print("\n" + "=" * 62)
    print(f"{'Metrik':<12}" + "".join(f"{isim:>25}" for isim in isimler))
    print("-" * 62)
    for anahtar, etiket in [("accuracy", "Accuracy"), ("precision", "Precision"),
                            ("recall", "Recall"), ("f1", "F1-skoru")]:
        print(f"{etiket:<12}" + "".join(f"{results[i][anahtar]:>25.4f}" for i in isimler))
    print("=" * 62)

    for isim in isimler:
        tn, fp, fn, tp = results[isim]["cm"].ravel()
        print(f"\n{isim} karmaşıklık matrisi:")
        print(f"   TN = {tn:3d}   FP = {fp:3d}")
        print(f"   FN = {fn:3d}   TP = {tp:3d}")


def plot_confusion_matrices(results, path):
    fig, eksenler = plt.subplots(1, len(results), figsize=(11, 4.5))
    etiketler = ["Diyabet yok (0)", "Diyabet var (1)"]

    for eksen, (isim, r) in zip(eksenler, results.items()):
        cm = r["cm"]
        eksen.imshow(cm, cmap="Blues")
        for i in range(2):
            for j in range(2):
                renk = "white" if cm[i, j] > cm.max() / 2 else "black"
                eksen.text(j, i, cm[i, j], ha="center", va="center", fontsize=16, color=renk)
        eksen.set_xticks([0, 1], etiketler)
        eksen.set_yticks([0, 1], etiketler)
        eksen.set_xlabel("Tahmin edilen sınıf")
        eksen.set_ylabel("Gerçek sınıf")
        eksen.set_title(isim)

    plt.tight_layout()
    plt.savefig(path, dpi=150)


def plot_loss_curves(lin_loss, log_loss, path):
    # İki kayıp farklı birimlerde (MSE ve BCE), bu yüzden ayrı grafiklerde
    fig, eksenler = plt.subplots(1, 2, figsize=(11, 4))

    eksenler[0].plot(lin_loss, color="steelblue")
    eksenler[0].set_title("Doğrusal Regresyon (Gradyan İnişi)")
    eksenler[0].set_xlabel("İterasyon")
    eksenler[0].set_ylabel("MSE")

    eksenler[1].plot(log_loss, color="indianred")
    eksenler[1].set_title("Lojistik Regresyon (Gradyan İnişi)")
    eksenler[1].set_xlabel("İterasyon")
    eksenler[1].set_ylabel("Binary Cross-Entropy")

    for eksen in eksenler:
        eksen.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)


def plot_output_distributions(lin_scores, log_probs, y_true, path):
    fig, eksenler = plt.subplots(1, 2, figsize=(12, 4.5))
    bins = np.linspace(-0.4, 1.4, 37)   # iki grafikte aynı aralık, karşılaştırılabilsin

    for eksen, cikti, baslik in [(eksenler[0], lin_scores, "Doğrusal Regresyon çıktıları"),
                                 (eksenler[1], log_probs, "Lojistik Regresyon olasılıkları")]:
        eksen.hist(cikti[y_true == 0], bins=bins, alpha=0.6, color="steelblue", label="Gerçek sınıf 0")
        eksen.hist(cikti[y_true == 1], bins=bins, alpha=0.6, color="indianred", label="Gerçek sınıf 1")
        eksen.axvline(0.5, color="black", linestyle="--", label="Eşik (0.5)")
        eksen.axvspan(-0.4, 0, color="gray", alpha=0.15)    # geçersiz olasılık bölgeleri
        eksen.axvspan(1, 1.4, color="gray", alpha=0.15)
        eksen.set_title(baslik)
        eksen.set_xlabel("Model çıktısı")
        eksen.set_ylabel("Hasta sayısı")
        eksen.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(path, dpi=150)