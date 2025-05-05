# Laporan Proyek Machine Learning Predictive Analythic Nilai Tukar Mata Uang USD to IDR (Time Series / Forecasting) - Bagas Cahyawiguna

## Domain Proyek

Fluktuasi nilai tukar mata uang asing, khususnya USD ke IDR, merupakan topik yang sangat penting dalam perekonomian Indonesia. Nilai tukar yang tidak stabil dapat memengaruhi sektor perdagangan, investasi, dan kebijakan moneter. Oleh karena itu, kemampuan untuk memprediksi nilai tukar mata uang asing sangat dibutuhkan oleh para pelaku ekonomi, investor, hingga pembuat kebijakan untuk mengambil keputusan yang lebih baik.

Dengan perkembangan teknologi, pendekatan berbasis data seperti machine learning menjadi solusi potensial dalam menghasilkan prediksi nilai tukar yang lebih akurat. Proyek ini bertujuan untuk memanfaatkan model machine learning untuk memprediksi nilai tukar USD ke IDR menggunakan data historis yang diperoleh dari Yahoo Finance.

### Mengapa Masalah Ini Harus Diselesaikan?

* **Stabilitas Ekonomi**: Prediksi nilai tukar dapat membantu dalam penyusunan kebijakan fiskal dan moneter.
* **Perencanaan Bisnis**: Bisnis yang bergantung pada impor atau ekspor sangat terpengaruh oleh perubahan nilai tukar.
* **Investasi**: Investor membutuhkan prediksi nilai tukar untuk mengelola risiko portofolio internasional.

### Referensi Terkait

* Siregar, R. Y., & Rajan, R. S. (2006). "Models of exchange rate determination: How relevant are they for developing economies in Asia?" *Journal of Asian Economics*, 17(1), 86-103. [https://doi.org/10.1016/j.asieco.2006.01.004](https://doi.org/10.1016/j.asieco.2006.01.004)
* Zhang, G. P. (2003). "Time series forecasting using a hybrid ARIMA and neural network model." *Neurocomputing*, 50, 159–175.

## Business Understanding

Proyek ini bertujuan untuk mengembangkan model machine learning yang mampu memprediksi nilai tukar USD ke IDR berdasarkan data historis. Prediksi ini berguna dalam pengambilan keputusan strategis, terutama bagi para pelaku ekonomi seperti investor, importir/eksportir, dan pembuat kebijakan.

### Problem Statements

1. Nilai tukar USD terhadap IDR memiliki fluktuasi yang tinggi dan sulit diprediksi secara akurat dengan metode tradisional.
2. Kurangnya alat prediktif berbasis data yang bisa memberikan estimasi nilai tukar dalam jangka pendek.
3. Diperlukan model machine learning yang mampu belajar dari data historis untuk menghasilkan prediksi nilai tukar yang akurat dan reliabel.

### Goals

1. Menganalisis data historis nilai tukar USD/IDR untuk menemukan pola atau tren yang relevan.
2. Membangun model prediktif berbasis machine learning untuk memperkirakan nilai tukar USD/IDR pada periode mendatang.
3. Mengevaluasi performa model prediktif menggunakan metrik evaluasi yang sesuai, seperti RMSE atau MAE.


### Solution Statements

* Menggunakan beberapa pendekatan model prediktif time series, yaitu:

  * **ARIMA (AutoRegressive Integrated Moving Average)** sebagai baseline model untuk data stasioner.
  * **SARIMA (Seasonal ARIMA)** untuk mengakomodasi pola musiman pada data nilai tukar.
  * **ARIMA-GARCH** untuk menangani sifat volatilitas (heteroskedastisitas) yang umum dalam data finansial.
  * **LSTM (Long Short-Term Memory)**, jenis Recurrent Neural Network (RNN) yang dirancang untuk mempelajari dependensi jangka panjang pada data sekuensial seperti time series.
* Masing-masing model dievaluasi dengan metrik **Root Mean Squared Error (RMSE)** dan **Mean Absolute Error (MAE)** untuk menilai performa prediksi.
* Model terbaik dipilih berdasarkan nilai metrik evaluasi terendah serta pertimbangan kompleksitas model dan kestabilan prediksi.

## Data Understanding

Data yang digunakan dalam proyek ini diperoleh dari Yahoo Finance dengan simbol **USDIDR=X**, yang merepresentasikan nilai tukar harian mata uang USD terhadap IDR. Dataset ini mencakup periode historis dari tahun 2001 hingga 2025, dengan total data yang didapatkan sebanyak **6010** data, 5 kolom **Index**, dan 1 Kolom khusus **MultiIndex**. Data ini cukup bersih tanpa adanya Missing Values tetapi terdapat **6 Duplicated data** dengan **2 Outliers** yang masih layak untuk dibersihkan melalui tahap ***Preparation***.

Sumber data: https://finance.yahoo.com/quote/USDIDR=X/history

Dataset ini diunduh dalam format CSV dan mencakup kolom-kolom berikut:

### Variabel-variabel dalam dataset:

* `Date`: Tanggal pengamatan.
* `Open`: Nilai tukar saat pasar dibuka.
* `High`: Nilai tukar tertinggi dalam satu hari.
* `Low`: Nilai tukar terendah dalam satu hari.
* `Close`: Nilai tukar saat pasar ditutup.
* `Volume`: Tidak relevan dalam konteks nilai tukar mata uang (selalu 0), sehingga diabaikan dalam analisis.

### Exploratory Data Analysis (EDA)

* Visualisasi tren nilai tukar menunjukkan adanya **pola naik-turun musiman** serta **volatilitas tinggi** dalam beberapa periode krisis keuangan.
* Distribusi nilai tukar bersifat **non-normal**, dengan **2 outlier** signifikan.
* Plot autokorelasi (ACF) dan partial autocorrelation (PACF) dilakukan untuk mengidentifikasi **orde AR dan MA** pada model ARIMA.
* Uji stasioneritas dilakukan menggunakan **Augmented Dickey-Fuller (ADF)** untuk memastikan apakah data perlu di-differencing.

## Data Preparation

Tahapan ini dilakukan untuk menyiapkan data agar dapat digunakan oleh model time series dan deep learning. Data preparation dilakukan secara berurutan sebagai berikut:

### 1. **Drop Unused Column 'Volume'**

* Kolom volume tidak akan digunakan untuk prediksi nilai tukar mata uang karena memiliki nilai yang selalu 0 pada keseluruhan datanya. Kolom volume biasanya menunjukkan jumlah transaksi yang terjadi dalam satu periode waktu tertentu (biasanya harian) yang berarti kolom tersebut biasa digunakan untuk melihat transaksi saham dan forex.
  
### 2. **Handling Missing Values & Duplicated Data**

* **Tidak ditemukan missing values** pada kolom harga (Open, High, Low, Close).
* **Terdapat 6 duplikasi data**, strategi yang dilakukan adalah **menghapus data yang terduplikasi** tersebut.

### 3. **Fokus pada Kolom 'Close'**

* Model prediksi difokuskan untuk memprediksi nilai **penutupan (Close)**.
* Kolom-kolom lain seperti `Open`, `High`, dan `Low` tidak digunakan agar model tetap sederhana dan fokus.

### 4. **Handling Outliers**

* **Ditemukannya 2 outliers**, strategi yang dilakukan adalah **menghapus 2 outliers** tersebut karena dengan jumlah data yang banyak (6000an data), dibanding 2 outliers yang dihapus mungkin tidak akan berpengaruh secara signifikan.

### 5. **Feature Engineering untuk Model Statistik**

* Dilakukan **differencing** pada data `Close` untuk menjadikan data stasioner (khususnya untuk model ARIMA/SARIMA).
* Ditambahkan **lag features** seperti nilai tukar hari sebelumnya (lag-1, lag-2, dst.) untuk input ke model prediktif tradisional.

### 6. **Normalisasi Data untuk LSTM**

* Untuk model LSTM, data `Close` dinormalisasi menggunakan **MinMaxScaler** ke rentang [0, 1] agar jaringan saraf dapat belajar secara efektif.
* Setelah prediksi, hasil dipetakan kembali ke skala asli dengan **inverse transform**.

### 7. **Data Splitting**

* Dataset dibagi menjadi **train dan test set** secara **chronological split** (bukan random), agar urutan waktu tetap terjaga.
* Proporsi umum: sekitar 80% untuk pelatihan dan 20% untuk pengujian.

### 8. **Reshaping untuk LSTM**

* Data input untuk LSTM direstrukturisasi ke dalam bentuk **[samples, timesteps, features]**, sesuai dengan arsitektur input RNN. Dimana:
    * samples = total sample training/test
    * timesteps = 30 (panjang sequence)
    * features = 1 (karena hanya menggunakan kolom Close)

## Modeling

Untuk memprediksi nilai tukar USD ke IDR, beberapa algoritma pemodelan deret waktu digunakan dan dibandingkan untuk menemukan pendekatan terbaik. Model-model yang digunakan mencakup pendekatan statistik dan deep learning:

### 1. **ARIMA (AutoRegressive Integrated Moving Average)**

**Cara Kerja Model**: ARIMA adalah model time series yang terdiri dari tiga komponen utama:
* AR (AutoRegressive): prediksi masa depan didasarkan pada observasi masa lalu (p lag sebelumnya).
* I (Integrated): jumlah diferensiasi yang dilakukan untuk membuat data stasioner (d).
* MA (Moving Average): prediksi berdasarkan kesalahan residu dari model sebelumnya (q).

ARIMA bekerja dengan mengubah data menjadi stasioner (melalui differencing), lalu membangun model berdasarkan korelasi antara nilai saat ini dan masa lalu serta kesalahan masa lalu.

Parameter yang Digunakan:
* p: jumlah lag pada komponen AR
* d: orde differencing
* q: jumlah lag pada komponen MA

Model dipilih menggunakan plot ACF, PACF, dan bantuan fungsi auto_arima

* **ARIMA(p,d,q)** digunakan sebagai baseline model.
* Parameter ditentukan berdasarkan ACF/PACF plot dan auto_arima.
* Model ini cukup baik untuk data stasioner dan linear.
* Tidak dapat menangkap musiman atau volatilitas tinggi dengan baik.

### 2. **SARIMA (Seasonal ARIMA)**

**Cara Kerja Model**: SARIMA adalah perluasan dari ARIMA yang menambahkan komponen musiman, cocok untuk data yang memiliki pola musiman (misalnya pola bulanan atau tahunan).

Struktur SARIMA:

**SARIMA(p,d,q)(P,D,Q,s)**
* (p,d,q): komponen ARIMA biasa.
* (P,D,Q): komponen musiman.
* s: panjang siklus musiman (season / misal s=12 untuk bulanan dalam 1 tahun).

SARIMA memperhitungkan autokorelasi jangka pendek dan musiman sekaligus.

* Menambahkan komponen musiman `(P,D,Q,s)` ke ARIMA untuk menangani pola musiman (season).
* Digunakan saat pola musiman tahunan atau bulanan teridentifikasi dari data tukar.
* Lebih kompleks dibanding ARIMA biasa, tetapi dapat memberikan hasil lebih stabil untuk data musiman.

### 3. **ARIMA-GARCH**

**Cara Kerja Model**: ARIMA-GARCH adalah model hybrid yang digunakan untuk data finansial yang menunjukkan volatilitas tinggi atau heteroskedastisitas.
* **Langkah 1**: Model ARIMA digunakan untuk memodelkan tren data dan menghilangkan autokorelasi.
* **Langkah 2**: Sisa error (residual) dari ARIMA sering kali tidak white noise, melainkan memiliki volatilitas. Oleh karena itu, GARCH (Generalized Autoregressive Conditional Heteroskedasticity) digunakan untuk memodelkan variansi residual tersebut.

GARCH dapat menangkap fluktuasi besar (volatilitas) yang tidak bisa ditangani ARIMA sendiri.

* Parameter: ARIMA: p, d, q dan GARCH: (p, q): jumlah lag dari variansi sebelumnya dan error sebelumnya
* Kombinasi ARIMA (untuk memodelkan tren) dan GARCH (untuk memodelkan volatilitas/residual).
* ARIMA digunakan terlebih dahulu untuk menghilangkan autokorelasi.
* Residual dari ARIMA kemudian dimodelkan dengan GARCH untuk menangkap heteroskedastisitas (volatilitas tidak konstan).
* Cocok untuk data keuangan yang menunjukkan fluktuasi ekstrem atau volatilitas tinggi.

### 4. **LSTM (Long Short-Term Memory Neural Network)**

**Cara Kerja Model**: LSTM adalah jenis Recurrent Neural Network (RNN) yang sangat efektif untuk data time series, karena mampu mengingat informasi jangka panjang melalui arsitektur memori internalnya (cell state dan gates).

Komponen utama LSTM:
* Forget Gate: memutuskan informasi lama yang perlu dibuang.
* Input Gate: memutuskan informasi baru yang akan disimpan.
* Output Gate: menentukan output dari unit LSTM.

LSTM sangat cocok untuk mempelajari pola jangka panjang dalam data sekuensial seperti nilai tukar harian.

**Arsitektur Model**:
* Input sequence: Menggunakan data dengan input sequence sepanjang **30 hari sebelumnya** untuk memprediksi hari berikutnya.
* Output: prediksi 1 hari ke depan
  
**Layer**:
* 1 LSTM layer dengan 50 unit
* 1 Dense layer sebagai output
* Normalisasi input dengan MinMaxScaler (rentang 0–1)
* Optimizer: Adam
* Loss function: Mean Squared Error (MSE)
* Callbacks: EarlyStopping (patience=5) untuk mencegah overfitting.

### Tuning dan Pemilihan Model

* Setiap model dievaluasi menggunakan metrik MAE, RMSE, dan MAPE.
* Tidak dilakukan extensive hyperparameter tuning (seperti GridSearchCV untuk ARIMA/SARIMA, atau tuning unit-layer untuk LSTM) karena fokus pada pembandingan awal antar pendekatan.
* Model terbaik dipilih berdasarkan performa di data uji.

## Evaluation

Pada proyek ini, tujuan utama adalah memprediksi nilai tukar USD ke IDR sebagai masalah regresi time series. Oleh karena itu, metrik evaluasi yang digunakan difokuskan pada akurasi prediksi nilai numerik, yaitu:

### 1. **MAE (Mean Absolute Error)**
   MAE mengukur rata-rata selisih absolut antara nilai aktual dan nilai hasil prediksi.
   Formula:

   $$
   \text{MAE} = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|
   $$

   Metrik ini memberikan informasi seberapa jauh prediksi dari nilai sebenarnya dalam satuan yang sama dengan target (Rupiah), sehingga mudah dipahami.

### 2. **RMSE (Root Mean Squared Error)**
   RMSE adalah akar dari rata-rata kuadrat selisih antara nilai aktual dan nilai prediksi.
   Formula:

   $$
   \text{RMSE} = \sqrt{ \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 }
   $$

   RMSE sensitif terhadap outlier karena penalti untuk kesalahan besar lebih tinggi.

### 3. **MAPE (Mean Absolute Percentage Error)**
   MAPE mengukur persentase rata-rata kesalahan absolut terhadap nilai aktual.
   Formula:

   $$
   \text{MAPE} = \frac{100\%}{n} \sum_{i=1}^{n} \left| \frac{y_i - \hat{y}_i}{y_i} \right|
   $$

   Metrik ini sangat berguna dalam konteks bisnis karena menyatakan seberapa besar kesalahan prediksi dalam persentase, sehingga dapat digunakan untuk mengukur performa relatif model terhadap skala harga.

### Hasil Evaluasi

Model dievaluasi menggunakan data uji (test set), dan berikut adalah hasil evaluasi dari model LSTM yang digunakan dalam proyek ini:

|    Model    |     MAE     |     RMSE    |    MAPE   |
| ----------- | ----------- | ----------- | --------- |
| LSTM        | 65.486293   | 92.236954   | 0.432285  |
| ARIMA       | 1644.529932 | 1798.863891 | 11.139310 |
| ARIMA-GARCH | 1644.529932 | 1798.863891 | 11.139310 |
| SARIMA      | 2212.802759 | 2260.489106 | 14.816056 |
### Interpretasi Best Model

* **MAE sebesar 65,49 Rupiah** menunjukkan bahwa rata-rata prediksi hanya meleset sekitar 65 Rupiah dari nilai sebenarnya, yang tergolong sangat rendah untuk data nilai tukar dengan kisaran puluhan ribu.
* **RMSE sebesar 92,24 Rupiah** mengindikasikan tidak ada kesalahan besar (outlier) yang signifikan karena nilainya masih relatif rendah.
* **MAPE sebesar 0,43%** menunjukkan bahwa model memiliki tingkat kesalahan prediksi yang sangat kecil jika dibandingkan dengan nilai tukar aktual, menandakan model sangat andal dalam konteks ini.

**Apakah model menjawab problem statements?**
* Model LSTM yang dikembangkan berhasil menjawab permasalahan fluktuasi tinggi pada nilai tukar dengan memberikan prediksi jangka pendek yang akurat (MAPE < 0.5%).

**Apakah goals tercapai?**
* Tujuan untuk membangun model prediktif berbasis data tercapai dengan model LSTM sebagai hasil terbaik (MAE rendah, RMSE rendah).

**Apakah solusi berdampak?**
* Model ini dapat digunakan oleh investor dan pengambil kebijakan sebagai alat bantu prediktif untuk merencanakan strategi finansial, mengelola risiko, atau menyesuaikan keputusan ekonomi terhadap potensi perubahan nilai tukar.

Dengan hasil ini, model **LSTM** terbukti memiliki performa prediksi yang sangat baik dalam memodelkan dan meramalkan nilai tukar USD ke IDR, menjadikannya solusi utama dalam proyek ini.


**---Ini adalah bagian akhir laporan---**
