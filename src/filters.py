import numpy as np

def convolution(image, C):
    C_n, C_m = C.shape[0]-1, C.shape[1]-1
    img_b = np.zeros((image.shape[0]+C_n, image.shape[1]+C_m)).astype(np.float32)
    decalage = C_n//2
    img_b[decalage:-decalage, decalage:-decalage] = image

    res = np.zeros_like(image).astype(np.float32)
    n, m = res.shape[0], res.shape[1]
    for i in range(C.shape[0]):
        for j in range(C.shape[1]):
            res += C[i, j]*img_b[i:n+i, j:m+j]

    return res

def blur_filter(img, size):
    kernel_blur = np.ones((size, size)) / size**2
    return convolution(img, kernel_blur)

def gaussian_kernel(size, sigma=0.8):
    ax = np.linspace(-size//2, size//2, size)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2)/(2.0*sigma**2))
    return kernel/np.sum(kernel) #on normalise

def gaussian_filter(img, size, sigma=0.8):
    kernel_gaussian = gaussian_kernel(size, sigma)
    return convolution(img, kernel_gaussian)

def median_filter(img, size=5):
    img_b = np.zeros((img.shape[0]+size-1, img.shape[1]+size-1)).astype(np.float32)
    decalage = size//2
    img_b[decalage:-decalage, decalage:-decalage] = img
    n = img_b.shape[0]
    m = img_b.shape[1]

    layers = [] #on crée une big matrice de tous nos décalages
    for i in range(size):
        for j in range(size):
            layers.append(img_b[i:n-2*decalage+i, j:m-2*decalage+j])

    clean_img = np.median(layers, axis=0) #on applique la fonction sur les indices par rapport à l'axe 0

    return clean_img

def bilateral(img, size, sigma_s, sigma_i):
    img = img.astype(np.float32)
    #on refait la même manip pour pouvoir agrandir l'image
    img_b = np.zeros((img.shape[0]+size-1, img.shape[1]+size-1)).astype(np.float32)
    decalage = size//2
    img_b[decalage:-decalage, decalage:-decalage] = img
    n = img_b.shape[0]
    m = img_b.shape[1]

    #clean_img = np.zeros_like(img)

    kernel_g = gaussian_kernel(size=size, sigma=sigma_s)
    #layers = []
    pixels = np.zeros_like(img) # on va additioner toutes les valeurs des pixels
    poids = np.zeros_like(img) # on va garder tous les poids pour normaliser

    for i in range(size):
        for j in range(size):
            W_s = kernel_g[i,j]
            layer = img - img_b[i:n-2*decalage+i, j:m-2*decalage+j] #Ic - Iv
            W_r = np.exp(-(layer**2)/(2*sigma_i**2)) # calcul de la différence d'intensité
            poids_final = W_r*W_s # multiplier les deux poids
            poids += poids_final
            pixels += poids_final*img_b[i:n-2*decalage+i, j:m-2*decalage+j]

    return pixels/poids
    

def nlm_denoising(img, h, search_size=21, patch_size=7, sigma_patch=0.8):
    #Préparation
    img = img.astype(np.float32)
    rayon_search = search_size//2
    rayon_patch = patch_size//2
    img_pad = np.pad(img, rayon_search+rayon_patch, mode='reflect') #comme le img_b d'avant mais optimisé avec numpy 
    kernel = gaussian_kernel(size=patch_size, sigma=sigma_patch)
    
    pixels = np.zeros_like(img)
    poids = np.zeros_like(img)
    n, m = img_pad.shape
    N, M = img.shape

    # On parcourt tous les décalages possibles dans la fenêtre de recherche
    for dx in range(search_size):
        for dy in range(search_size):
            layer = img_pad[dx:N+dx, dy:M+dy] #version décalée

            # calculer la distance entre deux patchs, revient à comparer les gaussiennes, puisuqe ce sont des moyennes pondérées des patchs ! quelle idée de génie
            diff = (img-layer)**2
            ssd_patch = convolution(diff, kernel)
            
            w = np.exp(-ssd_patch/(h**2))
            
            poids += w
            pixels += w*layer

    return pixels / poids