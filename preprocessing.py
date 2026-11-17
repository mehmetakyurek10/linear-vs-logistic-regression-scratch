import numpy as np
import pandas as pd

COLUMNS = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
           "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"]

# Bu sütunlarda 0 değeri biyolojik olarak imkânsız, eksik veri anlamına geliyor
ZERO_AS_MISSING = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]


def load_data(path="data/pima-indians-diabetes.csv"):
    df = pd.read_csv(path, header=None, names=COLUMNS)
    X = df.drop(columns="Outcome").astype(float)
    y = df["Outcome"].to_numpy()
    return X, y


def stratified_split(X, y, test_ratio=0.2, seed=42):
    """Her sınıftan aynı oranda örnek alarak eğitim/test ayrımı yapar."""
    rng = np.random.default_rng(seed)
    train_idx, test_idx = [], []

    for cls in np.unique(y):
        idx = np.where(y == cls)[0]      # bu sınıfa ait satırların indeksleri
        rng.shuffle(idx)                 # karıştır
        n_test = int(round(len(idx) * test_ratio))
        test_idx.extend(idx[:n_test])
        train_idx.extend(idx[n_test:])

    # Sınıflar arka arkaya dizilmesin diye son bir kez karıştır
    train_idx = rng.permutation(train_idx)
    test_idx = rng.permutation(test_idx)

    return (X.iloc[train_idx].copy(), X.iloc[test_idx].copy(),
            y[train_idx], y[test_idx])


def fill_missing(X_train, X_test, cols=ZERO_AS_MISSING):
    """0 değerlerini, eğitim kümesindeki sıfır olmayan değerlerin medyanıyla doldurur."""
    X_train, X_test = X_train.copy(), X_test.copy()
    medians = {}

    for col in cols:
        median = X_train.loc[X_train[col] != 0, col].median()
        medians[col] = median
        X_train[col] = X_train[col].replace(0, median)
        X_test[col] = X_test[col].replace(0, median)

    return X_train, X_test, medians


def standardize(X_train, X_test):
    """z = (x - ortalama) / std. İstatistikler sadece eğitim kümesinden."""
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    return (X_train - mean) / std, (X_test - mean) / std, mean, std


def prepare_data(path="data/pima-indians-diabetes.csv", test_ratio=0.2, seed=42):
    """Tüm ön işleme adımlarını sırayla uygular."""
    X, y = load_data(path)
    X_train, X_test, y_train, y_test = stratified_split(X, y, test_ratio, seed)
    X_train, X_test, _ = fill_missing(X_train, X_test)
    X_train, X_test, _, _ = standardize(X_train.to_numpy(), X_test.to_numpy())
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    # Bu kısım sadece dosyayı doğrudan çalıştırınca devreye girer (test amaçlı)
    X, y = load_data()
    X_tr, X_te, y_tr, y_te = stratified_split(X, y)
    print("Eğitim:", X_tr.shape, " Test:", X_te.shape)
    print("Eğitim sınıf dağılımı:", np.bincount(y_tr), f"(pozitif oranı %{y_tr.mean()*100:.1f})")
    print("Test sınıf dağılımı:  ", np.bincount(y_te), f"(pozitif oranı %{y_te.mean()*100:.1f})")

    X_tr, X_te, medians = fill_missing(X_tr, X_te)
    print("\nDoldurmada kullanılan medyanlar:")
    for col, m in medians.items():
        print(f"  {col:15s}: {m}")
    print("Doldurma sonrası kalan sıfır sayısı:", int((X_tr[ZERO_AS_MISSING] == 0).sum().sum()))

    X_tr, X_te, _, _ = standardize(X_tr.to_numpy(), X_te.to_numpy())
    print("\nStandardizasyon sonrası eğitim ortalamaları:", np.round(X_tr.mean(axis=0), 3))
    print("Standardizasyon sonrası eğitim std'leri:    ", np.round(X_tr.std(axis=0), 3))
    print("Test ortalamaları (0'a yakın ama tam 0 değil):", np.round(X_te.mean(axis=0), 3))