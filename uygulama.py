import streamlit as st
import tensorflow as tf
import joblib
import numpy as np
from deep_translator import GoogleTranslator

# Sayfa Yapısı
st.set_page_config(page_title="Yelp Analiz Dedektifi", page_icon="🔍")
st.title("🔍 Model Analiz Paneli")


@st.cache_resource
def load_assets():
    model = tf.keras.models.load_model('yelp_nlp_model.keras')
    vect = joblib.load('yelp_vectorizer.joblib')
    return model, vect


model, vect = load_assets()

yorum = st.text_area("Analiz edilecek metni girin:")

if st.button("Derin Analiz Yap"):
    if yorum:
        # 1. ÇEVİRİ AŞAMASI
        translation = GoogleTranslator(source='auto', target='en').translate(yorum)
        st.subheader("1. Aşama: Çeviri Sonucu")
        st.info(f"Sistemin algıladığı İngilizce metin: **{translation}**")

        # 2. VEKTÖRLEŞTİRME
        v_metin = vect.transform([translation]).toarray()

        # 3. TAHMİN VE OLASILIKLAR
        tahmin_olasiliklari = model.predict(v_metin, verbose=0)[0]

        st.subheader("2. Aşama: Modelin Karar Yüzdeleri")
        col1, col2 = st.columns(2)
        col1.metric("1 Yıldız Olasılığı", f"%{tahmin_olasiliklari[0] * 100:.2f}")
        col2.metric("5 Yıldız Olasılığı", f"%{tahmin_olasiliklari[1] * 100:.2f}")

        # 4. SONUÇ (Eğer 0. indis büyükse 1 yıldızdır)
        st.subheader("3. Aşama: Nihai Karar")
        sinif = np.argmax(tahmin_olasiliklari)

        if sinif == 1:
            st.success("🌟 SONUÇ: 5 YILDIZ (POZİTİF)")
        else:
            st.error("😡 SONUÇ: 1 YILDIZ (NEGATİF)")
    else:
        st.warning("Lütfen bir metin girin.")