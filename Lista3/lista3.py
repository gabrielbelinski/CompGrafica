import numpy as np
import cv2 as cv
import sys
import matplotlib.pyplot as plt

def rgb_para_cinza(imagem):
    imagem = imagem[:,:,0]/3 + imagem[:,:,1]/3 + imagem[:,:,2]/3
    return imagem.astype(np.uint8)

def transformada_fourier_2d(img):
    matriz_espectro = np.zeros((img.shape[0], img.shape[1]), dtype=np.complex128)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            soma_esp = 0
            for x in range(img.shape[0]):
                for y in range(img.shape[1]):
                    angulo = -2*np.pi*(i*x/img.shape[0] + j*y/img.shape[1])
                    soma_esp += img[x,y] * (np.cos(angulo) + 1j*np.sin(angulo))
            matriz_espectro[i,j] = soma_esp
    return matriz_espectro

def fourier(img, nome_arquivo):
    #magnitude = np.abs(transformada_fourier_2d(img))
    dft = cv.dft(np.float32(img), flags=cv.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    espectro = np.log(1+cv.magnitude(dft_shift[:,:,0], dft_shift[:,:,1]))
    fig, axes = plt.subplots(1,2)
    ax = axes.ravel()
    ax[0].imshow(img, cmap='gray')
    ax[0].title.set_text('Imagem de entrada')
    ax[0].set_axis_off()
    ax[1].imshow(espectro, cmap='gray')
    ax[1].title.set_text('Espectro de Fourier')
    ax[1].set_axis_off()
    fig.tight_layout()
    fig.savefig('espectrofourier_{}.png'.format(nome_arquivo))
    plt.close()

def passa_baixa(img, nome_arquivo):
    rows, cols = img.shape
    mask = np.zeros((rows, cols, 2))
    mask[rows//2, cols//2, 0] = 1
    mask[:,:,0] = cv.GaussianBlur(mask[:,:,0], (201, 201), 0)
    mask[:,:,1] = mask[:,:,0]
    dft = cv.dft(np.float32(img), flags=cv.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)

    filtered_shift = dft_shift*mask
    filtered = np.fft.ifftshift(filtered_shift)
    img_back = cv.idft(filtered)
    img_back = cv.magnitude(img_back[:,:,0], img_back[:,:,1])
    img_back = cv.normalize(img_back, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)

    fig, axes = plt.subplots(1,2)
    ax = axes.ravel()
    ax[0].imshow(img, cmap='gray')
    ax[0].title.set_text('Imagem de entrada')
    ax[0].set_axis_off()
    ax[1].imshow(img_back, cmap='gray')
    ax[1].title.set_text('Filtro passa-baixa')
    ax[1].set_axis_off()
    fig.tight_layout()
    fig.savefig('passabaixa_{}.png'.format(nome_arquivo))
    plt.close()
    return dft_shift
    

def passa_alta(img, nome_arquivo):
    radius = 80
    dft_shift = passa_baixa(img, nome_arquivo)
    rows, cols = img.shape
    crow, ccol = rows // 2, cols //2
    dft_shift[crow-radius:crow+radius, ccol-radius:ccol+radius, 0] = 0  
    dft_shift[crow-radius:crow+radius, ccol-radius:ccol+radius, 1] = 1 
    filtered = np.fft.ifftshift(dft_shift) 
    img_back = cv.idft(filtered)  
    img_back = cv.magnitude(img_back[:,:,0], img_back[:,:,1])  
    #img_back = cv.normalize(img_back, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)
    fig, axes = plt.subplots(1,2)
    ax = axes.ravel()
    ax[0].imshow(img, cmap='gray')
    ax[0].title.set_text('Imagem de entrada')
    ax[0].set_axis_off()
    ax[1].imshow(img_back, cmap='gray')
    ax[1].title.set_text('Filtro passa-alta')
    ax[1].set_axis_off()
    fig.tight_layout()
    fig.savefig('passaalta_{}.png'.format(nome_arquivo))
    plt.close()

def rejeita_banda(img, img2, nome_arquivo):
    img2 = img2.astype(np.uint8)
    mask = np.zeros((img2.shape[0], img2.shape[1],2), np.uint8)
    mask[:,:,0] = img2
    mask[:,:,1] = img2
    dft = cv.dft(np.float32(img), flags=cv.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    espectro = np.log(1+cv.magnitude(dft_shift[:,:,0], dft_shift[:,:,1]))
    filtered_shift = dft_shift * mask
    filtered = np.fft.ifftshift(filtered_shift)
    img_back = cv.idft(filtered)
    img_back = cv.magnitude(img_back[:,:,0], img_back[:,:,1])
    #img_back = cv.normalize(img_back, None, 0, 255, cv.NORM_MINMAX)
    fig, axes = plt.subplots(2,2)
    ax = axes.ravel()
    ax[0].imshow(img, cmap='gray')
    ax[0].title.set_text('Imagem de entrada')
    ax[0].set_axis_off()
    ax[1].imshow(img_back, cmap='gray')
    ax[1].title.set_text('Imagem filtrada')
    ax[1].set_axis_off()
    ax[2].imshow(espectro, cmap='gray')
    ax[2].title.set_text('Espectro de Fourier')
    ax[2].set_axis_off()
    ax[3].imshow(mask[:,:,0], cmap='gray')
    ax[3].title.set_text('Filtro Rejeita-Banda')
    ax[3].set_axis_off()
    fig.tight_layout()
    fig.savefig('rejeitabanda_{}.png'.format(nome_arquivo))
    plt.close()

def passa_banda(img, img2, nome_arquivo):
    img2 = img2.astype(np.uint8)
    mask = np.zeros((img2.shape[0], img2.shape[1],2), np.uint8)
    mask[:,:,0] = img2
    mask[:,:,1] = img2
    dft = cv.dft(np.float32(img), flags=cv.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    espectro = np.log(1+cv.magnitude(dft_shift[:,:,0], dft_shift[:,:,1]))
    filtered_shift = dft_shift * mask
    filtered = np.fft.ifftshift(filtered_shift)
    img_back = cv.idft(filtered)
    img_back = cv.magnitude(img_back[:,:,0], img_back[:,:,1])
    #img_back = cv.normalize(img_back, None, 0, 255, cv.NORM_MINMAX)
    fig, axes = plt.subplots(2,2)
    ax = axes.ravel()
    ax[0].imshow(img, cmap='gray')
    ax[0].title.set_text('Imagem de entrada')
    ax[0].set_axis_off()
    ax[1].imshow(img_back, cmap='gray')
    ax[1].title.set_text('Imagem filtrada')
    ax[1].set_axis_off()
    ax[2].imshow(espectro, cmap='gray')
    ax[2].title.set_text('Espectro de Fourier')
    ax[2].set_axis_off()
    ax[3].imshow(mask[:,:,0], cmap='gray')
    ax[3].title.set_text('Filtro Passa-Banda')
    ax[3].set_axis_off()
    fig.tight_layout()
    fig.savefig('passabanda_{}.png'.format(nome_arquivo))
    plt.close()

def gera_filtro_passabanda(img, radius_int, radius_ext, nome_arquivo):
    rows, cols = img.shape
    crow, ccol = rows//2, cols//2
    filtro = np.zeros((rows, cols), np.uint8)
    for i in range(rows):
        for j in range(cols):
            dist=np.sqrt((i-crow)**2+(j-ccol)**2)
            if (radius_int <= dist) and (dist <= radius_ext):
                filtro[i,j] = 1  
    plt.imsave('filtro_passabanda_{}.png'.format(nome_arquivo), filtro, cmap='gray')

def gera_filtro_rejeitabanda(img, radius_int, radius_ext, nome_arquivo):
    rows, cols = img.shape
    crow, ccol = rows // 2, cols // 2
    filtro=np.ones((rows, cols), np.uint8)
    for i in range(rows):
        for j in range(cols):
            dist=np.sqrt((i-crow)**2+(j-ccol)**2)
            if (radius_int <= dist) and (dist <= radius_ext):
                filtro[i, j]=0  
    plt.imsave('filtro_rejeitabanda_{}.png'.format(nome_arquivo), filtro, cmap='gray')
    
def main():
    nomes_arquivos = ['arara.png', 'barra1.png', 'barra2.png', 'barra3.png', 'barra4.png', 'teste.tif',  'img_gabriel.jpg']
    for nome in nomes_arquivos:
        img = cv.imread(nome)
        fourier(rgb_para_cinza(img), nome.split('.')[0])
    
    img = cv.imread(nomes_arquivos[5])
    passa_alta(rgb_para_cinza(img), nomes_arquivos[5].split('.')[0])
    img = cv.imread(nomes_arquivos[6])
    passa_alta(rgb_para_cinza(img), nomes_arquivos[6].split('.')[0])

    img = cv.imread(nomes_arquivos[0])
    img2 = cv.imread('arara_filtro.png')
    rejeita_banda(rgb_para_cinza(img), rgb_para_cinza(img2), 'arara')
    
    gera_filtro_passabanda(rgb_para_cinza(cv.imread(nomes_arquivos[5])), 50, 130, 'teste')
    gera_filtro_rejeitabanda(rgb_para_cinza(cv.imread(nomes_arquivos[5])), 80, 130, 'teste')
    passa_banda(rgb_para_cinza(cv.imread(nomes_arquivos[5])), rgb_para_cinza(cv.imread('filtro_passabanda_teste.png')), 'teste')
    rejeita_banda(rgb_para_cinza(cv.imread(nomes_arquivos[5])), rgb_para_cinza(cv.imread('filtro_rejeitabanda_teste.png')), 'teste')
    gera_filtro_passabanda(rgb_para_cinza(cv.imread(nomes_arquivos[6])), 50, 130, 'img_gabriel')
    gera_filtro_rejeitabanda(rgb_para_cinza(cv.imread(nomes_arquivos[6])), 80, 130, 'img_gabriel')
    passa_banda(rgb_para_cinza(cv.imread(nomes_arquivos[6])), rgb_para_cinza(cv.imread('filtro_passabanda_img_gabriel.png')), 'img_gabriel')
    rejeita_banda(rgb_para_cinza(cv.imread(nomes_arquivos[6])), rgb_para_cinza(cv.imread('filtro_rejeitabanda_img_gabriel.png')), 'img_gabriel')
    


if __name__ == '__main__':
    main()






    



        



            