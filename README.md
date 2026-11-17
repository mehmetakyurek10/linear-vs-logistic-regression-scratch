# Doğrusal ve Lojistik Regresyon ile İkili Sınıflandırma

Pima Indians Diabetes veri kümesi üzerinde, sıfırdan kodlanmış doğrusal ve
lojistik regresyon modellerinin sınıflandırma performansının karşılaştırılması.

## Kurulum

pip install -r requirements.txt

## Çalıştırma

python data_exploration.py # keşifsel veri analizi ve grafikler
python main.py # modellerin eğitimi, değerlendirme ve karşılaştırma

## Dosyalar

- data/pima-indians-diabetes.csv : veri kümesi (768 örnek, 8 özellik, 2 sınıf)
- data_exploration.py : veri kümesinin incelenmesi ve görselleştirilmesi
- preprocessing.py : eksik değer doldurma, stratified bölme, standardizasyon
- linear_regression.py : doğrusal regresyon (normal denklem ve gradyan inişi)
- logistic_regression.py : lojistik regresyon (gradyan inişi, BCE kaybı)
- evaluation.py : karmaşıklık matrisi, accuracy, precision, recall, F1 ve grafikler
- main.py : tüm adımları sırayla çalıştıran ana dosya
- figures/ : üretilen tüm grafikler
