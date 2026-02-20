import streamlit as st
import numpy as np
from PIL import Image
import time

# Import de tes fonctions (assure-toi que les fichiers sont dans le même dossier)
from noise import add_gaussian_noise, add_salt_and_pepper
from filters import (blur_filter, gaussian_filter, median_filter, 
                     bilateral, nlm_denoising)
from metrics import pnsr, mse

# Pour pouvoir appliquer sur du rgb
def apply_on_rgb(img, filter_func, **kwargs):
    if len(img.shape) == 2: # Déjà en gris
        return filter_func(img, **kwargs)
    
    channels = []
    for i in range(img.shape[2]): # On boucle sur R, G et B
        channels.append(filter_func(img[:, :, i], **kwargs))
    
    # On empile et on s'assure de rester entre 0 et 255
    return np.clip(np.stack(channels, axis=2), 0, 255).astype(np.uint8)

# app streamlit
st.set_page_config(page_title="Denoising Project - Yang Benjamin", layout="wide")
st.title("🔬 Image Denoising Project - Yang Benjamin")
st.sidebar.header("Paramètres")

# 1. CHARGEMENT DE L'IMAGE
uploaded_file = st.sidebar.file_uploader("Choisir une image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Lecture et conversion en Array
    raw_img = np.array(Image.open(uploaded_file))
    
    # Option : Passage en Gris
    is_gray = st.sidebar.checkbox("Travailler en Niveaux de Gris")
    if is_gray and len(raw_img.shape) == 3:
        # Formule de luminance standard
        original_img = np.dot(raw_img[...,:3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)
    else:
        original_img = raw_img

    # 2. AJOUT DE BRUIT
    st.sidebar.subheader("🕹️ Simulation de Bruit")
    noise_type = st.sidebar.selectbox("Type de bruit", ["Aucun", "Gaussien", "Poivre et Sel"])
    
    noisy_img = original_img.copy()
    if noise_type == "Gaussien":
        sig_n = st.sidebar.slider("Sigma Bruit", 1, 100, 25)
        noisy_img = add_gaussian_noise(original_img, sigma=sig_n).astype(np.uint8)
    elif noise_type == "Poivre et Sel":
        prob = st.sidebar.slider("Ratio Bruit", 0.01, 0.2, 0.05)
        noisy_img = add_salt_and_pepper(original_img, ratio=prob).astype(np.uint8)

    # 3. FILTRAGE (DENOISING)
    st.sidebar.subheader("🧼 Nettoyage")
    filter_type = st.sidebar.selectbox("Filtre", ["Aucun", "Moyenneur", "Gaussien", "Médian", "Bilatéral", "NLM"])
    
    denoised_img = noisy_img.copy()
    start_time = 0
    
    if filter_type != "Aucun":
        t0 = time.time()
        
        if filter_type == "Moyenneur":
            s = st.sidebar.slider("Taille Noyau", 3, 11, 3, step=2)
            denoised_img = apply_on_rgb(noisy_img, blur_filter, size=s)
            
        elif filter_type == "Gaussien":
            s = st.sidebar.slider("Taille", 3, 11, 3, step=2)
            sig = st.sidebar.slider("Sigma Spatial", 0.1, 5.0, 1.0)
            denoised_img = apply_on_rgb(noisy_img, gaussian_filter, size=s, sigma=sig)
            
        elif filter_type == "Médian":
            s = st.sidebar.slider("Taille", 3, 11, 3, step=2)
            denoised_img = apply_on_rgb(noisy_img, median_filter, size=s)
            
        elif filter_type == "Bilatéral":
            s = st.sidebar.slider("Taille", 3, 11, 5, step=2)
            ss = st.sidebar.slider("Sigma Spatial", 1.0, 10.0, 2.0)
            si = st.sidebar.slider("Sigma Intensité", 1, 100, 30)
            denoised_img = apply_on_rgb(noisy_img, bilateral, size=s, sigma_s=ss, sigma_i=si)
            
        elif filter_type == "NLM":
            st.warning("Attention : Le NLM peut être lent sur de grandes images.")
            sw = st.sidebar.slider("Fenêtre Recherche", 7, 31, 11, step=2)
            pw = st.sidebar.slider("Fenêtre Patch", 3, 9, 5, step=2)
            h_val = st.sidebar.slider("H (Lissage)", 1, 100, 20)
            denoised_img = apply_on_rgb(noisy_img, nlm_denoising, h=h_val, search_size=sw, patch_size=pw)
            
        start_time = time.time() - t0

    # 4. AFFICHAGE DES RÉSULTATS
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Originale")
        st.image(original_img, use_container_width=True)
        st.write("---")
        
    with col2:
        st.subheader("Bruitée")
        st.image(noisy_img, use_container_width=True)
        st.metric("PSNR initial", f"{pnsr(original_img, noisy_img):.2f} dB")
        
    with col3:
        st.subheader("Restaurée")
        st.image(denoised_img, use_container_width=True)
        psnr_final = pnsr(original_img, denoised_img)
        mse_final = mse(original_img, denoised_img)
        st.metric("PSNR Final", f"{psnr_final:.2f} dB", delta=f"{psnr_final - pnsr(original_img, noisy_img):.2f} dB")
        st.write(f"**MSE:** {mse_final:.1f}")
        if start_time > 0:
            st.write(f"⏱️ Temps : {start_time:.2f}s")

else:
    st.info("Veuillez charger une image dans la barre latérale pour commencer.")