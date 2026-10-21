# 🌸 Iris Çiçeği Sınıflandırıcı Web Servisi

PyTorch ile eğitilmiş bir Neural Network modelini FastAPI ile web servisi olarak sunan ve modern bir HTML/CSS/JavaScript arayüzüne sahip proje.

## 📋 Özellikler

- **PyTorch Neural Network**: 4 input → 16 → 16 → 3 output mimarisi
- **FastAPI Backend**: Hızlı ve modern Python web framework'ü
- **Responsive Frontend**: HTML, CSS, JavaScript ile modern kullanıcı arayüzü
- **Real-time Predictions**: Anlık tahmin yapma ve olasılık görselleştirme
- **RESTful API**: Kolay entegrasyon için API endpoint'leri

## 🚀 Kurulum

### 1. Gerekli Paketleri Yükleyin

```bash
pip install -r requirements.txt
```

### 2. Modeli Eğitin ve Kaydedin

Eğer zaten eğitilmiş bir modeliniz varsa:

```python
import torch
from model import IrisClassifier

# Modelinizi yükleyin
model = IrisClassifier()
# ... (eğitim kodlarınız)

# Modeli kaydedin
torch.save(model.state_dict(), 'iris_model.pth')
```

Veya hazır scripti kullanın:

```bash
python export_model.py
```

Bu script:
- Iris veri setini yükler
- Modeli eğitir (200 epoch)
- Modeli `iris_model.pth` olarak kaydeder
- Kaydedilen modeli test eder
- Örnek tahminler yapar

### 3. Web Servisini Başlatın

```bash
uvicorn main:app --reload
```

Veya:

```bash
python main.py
```

Servis şu adreste çalışacaktır: `http://localhost:8000`


## 🔧 Model Mimarisi

```python
class IrisClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_layer_stack = nn.Sequential(
            nn.Linear(4, 16),    # Input layer: 4 özellik
            nn.ReLU(),
            nn.Linear(16, 16),   # Hidden layer
            nn.ReLU(),
            nn.Linear(16, 3)     # Output layer: 3 sınıf
        )
```

**Girdi Özellikleri:**
- Sepal Length (Taç Yaprak Uzunluğu)
- Sepal Width (Taç Yaprak Genişliği)
- Petal Length (Çanak Yaprak Uzunluğu)
- Petal Width (Çanak Yaprak Genişliği)

**Çıktı Sınıfları:**
- Setosa
- Versicolor
- Virginica

## 📡 API Kullanımı

### Health Check

```bash
curl http://localhost:8000/health
```

Yanıt:
```json
{
    "status": "healthy",
    "model_loaded": true
}
```

### Tahmin Yapma

```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
         "sepal_length": 5.1,
         "sepal_width": 3.5,
         "petal_length": 1.4,
         "petal_width": 0.2
     }'
```

Yanıt:
```json
{
    "prediction": "Setosa",
    "class_id": 0,
    "probabilities": {
        "Setosa": 98.45,
        "Versicolor": 1.32,
        "Virginica": 0.23
    },
    "confidence": 98.45
}
```

## 🎨 Kullanıcı Arayüzü

Web arayüzüne erişmek için tarayıcınızda şu adresi açın:

```
http://localhost:8000
```

Arayüz özellikleri:
- ✨ Modern ve responsive tasarım
- 📊 Gerçek zamanlı tahmin sonuçları
- 📈 Olasılık dağılımı görselleştirmesi
- 🎯 Güven skoru göstergesi
- ⚡ Hızlı ve kullanıcı dostu

## 🧪 Model Performansı

Export script'i çalıştırdığınızda model performansını görebilirsiniz:

```
Epoch:   0 | Loss: 1.0986 | Acc:  33.33% | Test Loss: 1.0986 | Test Acc:  33.33%
Epoch:  20 | Loss: 0.7234 | Acc:  66.67% | Test Loss: 0.6845 | Test Acc:  70.00%
Epoch:  40 | Loss: 0.4521 | Acc:  88.33% | Test Loss: 0.3912 | Test Acc:  93.33%
...
Epoch: 180 | Loss: 0.0823 | Acc:  98.33% | Test Loss: 0.0745 | Test Acc:  96.67%

✅ Eğitim tamamlandı!
   Final Test Accuracy: 96.67%
```

## 💡 Önemli Notlar

### Model Kaydetme

PyTorch'ta model kaydetmenin iki yolu vardır:

**1. Sadece Model Ağırlıklarını Kaydetme (Önerilen):**
```python
torch.save(model.state_dict(), 'iris_model.pth')
```

**2. Tüm Modeli Kaydetme:**
```python
torch.save(model, 'iris_model.pth')
```

Bu projede **state_dict** yöntemi kullanılmıştır çünkü:
- Daha esnek (model sınıfını ayrı tutabilirsiniz)
- Daha küçük dosya boyutu
- Daha güvenli (pickle sorunlarından kaçınır)

### Model Yükleme

Model yüklerken aynı model mimarisine ihtiyacınız vardır:

```python
from model import IrisClassifier

# Model nesnesini oluştur
model = IrisClassifier()

# Ağırlıkları yükle
model.load_state_dict(torch.load('iris_model.pth'))

# Evaluation moduna geç
model.eval()
```

### Inference (Tahmin) Modu

Tahmin yaparken mutlaka:

```python
model.eval()  # Dropout ve BatchNorm katmanlarını devre dışı bırak

with torch.inference_mode():  # Veya torch.no_grad()
    predictions = model(input_data)
```

## 🐛 Sorun Giderme

### Model Dosyası Bulunamadı

Eğer `iris_model.pth` dosyası yoksa:

```bash
python export_model.py
```

### Port Zaten Kullanımda

8000 portu kullanımdaysa farklı bir port kullanın:

```bash
uvicorn main:app --reload --port 8001
```

### Modül Bulunamadı Hatası

Tüm bağımlılıkların yüklü olduğundan emin olun:

```bash
pip install -r requirements.txt

