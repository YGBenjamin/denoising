import numpy as np

def add_gaussian_noise(img, sigma=20):
    noise = np.random.normal(0, sigma, img.shape)
    noise_img = np.clip(img+noise, 0, 255)

    return noise_img

def add_salt_and_pepper(img, ratio=0.05):
    sap = np.random.random(img.shape)
    noise_img = np.copy(img)
    noise_img[sap < ratio/2]=0 #les points aléatoires en dessous de ratio/2 deviennent absolument noirs
    noise_img[sap > 1-ratio/2]=255 #les points au dessus de 1-ratio/2 deviennent absolument blancs

    return noise_img