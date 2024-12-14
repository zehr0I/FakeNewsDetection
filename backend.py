from flask import Flask, jsonify, request
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import re

# Gerekli NLTK veri setlerini indir
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

# Metin temizleme fonksiyonu
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    tokens = word_tokenize(text)

    stop_words = set(stopwords.words("english"))
    tokens = [token for token in tokens if token not in stop_words]

    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]

    cleaned_text = " ".join(tokens)
    return cleaned_text

# Flask uygulaması başlat
app = Flask(__name__)

# Modeli yükle
loaded_model = load_model("lstm_model.h5")

# Maksimum dizi uzunluğu
max_sequence_length = 50

# Tahmin için rota oluştur
@app.route('/predict', methods=["POST"])
def predict():
    json_data = request.json
    data = json_data.get("news", "")
    if not data:
        return jsonify({"error": "The 'news' field is required"}), 400

    # Veriyi temizle
    new_data = clean_text(data)

    # Tokenizer oluştur ve metni işleme al
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts([new_data])

    sequence = tokenizer.texts_to_sequences([new_data])
    padded_sequence = pad_sequences(sequence, padding="pre", maxlen=max_sequence_length)

    # Tahmin yap
    prediction = loaded_model.predict(padded_sequence)
    binary_prediction = np.round(prediction)

    # Yanıtı JSON olarak döndür
    response = {"prediction": int(binary_prediction)}
    return jsonify(response)

# Uygulamayı çalıştır
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=6000, debug=True)

