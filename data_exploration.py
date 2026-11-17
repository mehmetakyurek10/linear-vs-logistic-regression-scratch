import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

os.makedirs("figures", exist_ok=True)

sutunlar = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
            "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"]
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)
df = pd.read_csv("data/pima-indians-diabetes.csv", header=None, names=sutunlar)

print("Boyut:", df.shape)
print()
print(df.head())
print()
print(df.describe())
print()
print("Sınıf dağılımı:")
print(df["Outcome"].value_counts())
print()
print("Sıfır değer sayıları:")
supheli = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
for sutun in supheli:
    sayi = (df[sutun] == 0).sum()
    oran = sayi / len(df) * 100
    print(f"{sutun:15s}: {sayi:4d}  (%{oran:.1f})")

# Grafikler için sıfırları NaN yapıyoruz ki çizimleri bozmasınlar
# (Bu sadece görselleştirme için; asıl temizliği ön işlemede yapacağız)
df_grafik = df.copy()
df_grafik[supheli] = df_grafik[supheli].replace(0, np.nan)

# 1) Sınıf dağılımı
plt.figure(figsize=(5, 4))
df["Outcome"].value_counts().sort_index().plot(kind="bar", color=["steelblue", "indianred"])
plt.xticks([0, 1], ["Diyabet yok (0)", "Diyabet var (1)"], rotation=0)
plt.ylabel("Örnek sayısı")
plt.title("Sınıf Dağılımı")
plt.tight_layout()
plt.savefig("figures/class_distribution.png", dpi=150)

# 2) Her özelliğin sınıflara göre histogramı
ozellikler = sutunlar[:-1]
fig, eksenler = plt.subplots(2, 4, figsize=(16, 7))
for eksen, ozellik in zip(eksenler.flatten(), ozellikler):
    eksen.hist(df_grafik[df_grafik["Outcome"] == 0][ozellik].dropna(), bins=25, alpha=0.6, label="0", color="steelblue")
    eksen.hist(df_grafik[df_grafik["Outcome"] == 1][ozellik].dropna(), bins=25, alpha=0.6, label="1", color="indianred")
    eksen.set_title(ozellik)
    eksen.legend()
plt.tight_layout()
plt.savefig("figures/feature_histograms.png", dpi=150)

# 3) Glucose - BMI saçılım grafiği (doğrusal olmama kanıtı)
plt.figure(figsize=(7, 5))
for sinif, renk in [(0, "steelblue"), (1, "indianred")]:
    alt = df_grafik[df_grafik["Outcome"] == sinif]
    plt.scatter(alt["Glucose"], alt["BMI"], c=renk, alpha=0.6, s=20, label=f"Outcome = {sinif}")
plt.xlabel("Glucose")
plt.ylabel("BMI")
plt.title("Glucose - BMI (sınıflara göre)")
plt.legend()
plt.tight_layout()
plt.savefig("figures/glucose_bmi_scatter.png", dpi=150)

# 4) Korelasyon matrisi
korelasyon = df_grafik.corr()
plt.figure(figsize=(9, 7))
plt.imshow(korelasyon, cmap="coolwarm", vmin=-1, vmax=1)
plt.colorbar()
plt.xticks(range(len(sutunlar)), sutunlar, rotation=90)
plt.yticks(range(len(sutunlar)), sutunlar)
for i in range(len(sutunlar)):
    for j in range(len(sutunlar)):
        plt.text(j, i, f"{korelasyon.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)
plt.title("Korelasyon Matrisi")
plt.tight_layout()
plt.savefig("figures/correlation_matrix.png", dpi=150)

print()
print("Outcome ile korelasyonlar:")
print(korelasyon["Outcome"].sort_values(ascending=False))

plt.show()