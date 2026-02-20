import numpy as np

def mse(img1, img2):
    #Je suppose ici que les deux images ont la même taille, pour aller plus vite...
    m = img1.shape[0]
    n = img1.shape[1]
    return np.sum((img1.astype(np.float32)-img2.astype(np.float32))**2)/(m*n)

def pnsr(img1, img2):
    mse_val = mse(img1, img2)
    if mse_val== 0 : return float('inf') #images identiques
    return 20*np.log10(255/np.sqrt(mse_val)) #donne le résultat en dB