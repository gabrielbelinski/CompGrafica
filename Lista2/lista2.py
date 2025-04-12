import cv2 as cv
import numpy as np
import sys

def rgb_para_cinza(imagem):
    imagem = imagem[:,:,0]/3 + imagem[:,:,1]/3 + imagem[:,:,2]/3
    return imagem.astype(np.uint8)

def suavizacao_md_vizinhanca(img, tam_janela, nome_arquivo):
    img = img.astype(np.float32)
    img_final = np.zeros_like(img)
    deslocamento = tam_janela // 2
    
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            soma_md = 0
            cont = 0
            for di in range(-deslocamento, deslocamento + 1):
                for dj in range(-deslocamento, deslocamento + 1):
                    v_i = i+di
                    v_j = j+dj
                    if (v_i >= 0 and v_i < img.shape[0]) and (v_j >= 0 and v_j < img.shape[1]):
                        soma_md += img[v_i, v_j]
                        cont+=1
                    else:
                        soma_md += 0
                        cont+=1
            img_final[i,j] = soma_md/cont

    img_final = np.clip(img_final, 0, 255).astype(np.uint8)
    #return img_final
    cv.imwrite('{}_suavizacao_md_viz.png'.format(nome_arquivo), img_final)
    
def suavizacao_kvizinhos(img, tam_janela, k, nome_arquivo):
    img = img.astype(np.float32)
    img_final = np.zeros_like(img)
    deslocamento = tam_janela // 2

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            dist_vizinhos = []
            for di in range(-deslocamento, deslocamento + 1):
                for dj in range(-deslocamento, deslocamento + 1):
                    v_i = i + di
                    v_j = j + dj
                    if (v_i >= 0 and v_i < img.shape[0]) and (v_j >= 0 and v_j < img.shape[1]) and (v_i != i or v_j != j):
                        dist = np.sqrt(di**2+dj**2)
                        dist_vizinhos.append((dist, img[v_i,v_j]))

            dist_vizinhos.sort(key=lambda e: e[0])
            soma_intensidades = 0
            for vizinho in dist_vizinhos[:k]:
                soma_intensidades += vizinho[1]
            img_final[i, j] = soma_intensidades/k

    img_final = np.clip(img_final, 0, 255).astype(np.uint8)
    #return img_final
    cv.imwrite('{}_suavizacao_k_viz.png'.format(nome_arquivo), img_final)
   
def suavizacao_mediana(img, tam_janela, nome_arquivo):
    img = img.astype(np.float32)
    img_final = np.zeros_like(img)
    deslocamento = tam_janela // 2

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            vizinhos = []
            for di in range(-deslocamento, deslocamento + 1):
                for dj in range(-deslocamento, deslocamento + 1):
                    v_i = i + di
                    v_j = j + dj
                    if (v_i >= 0 and v_i < img.shape[0]) and (v_j >= 0 and v_j < img.shape[1]):
                        vizinhos.append(img[v_i, v_j])
            vizinhos.sort()
            if len(vizinhos) % 2 != 0:
                img_final[i,j] = vizinhos[len(vizinhos)//2]
            else:
                img_final[i,j] = (vizinhos[len(vizinhos)//2 - 1] + vizinhos[len(vizinhos)//2])

            
    img_final = np.clip(img_final, 0, 255).astype(np.uint8) 
    #return img_final
    cv.imwrite('{}_suavizacao_mediana.png'.format(nome_arquivo), img_final) 
        
def laplaciano(img, nome_arquivo):
    kernel = np.array([[0,-1,0],
              [-1,4,-1],
              [0,-1,0]])
    img = img.astype(np.float32)
    img_final = np.zeros_like(img)

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            soma = 0
            for di in range(-1, 2):
                for dj in range(-1, 2):
                    v_i = i + di
                    v_j = j + dj
                    if (v_i >= 0 and v_i < img.shape[0]) and (v_j >= 0 and v_j < img.shape[1]):
                        soma += img[v_i,v_j] * kernel[di+1,dj+1]
            img_final[i,j] = soma

    img_final = np.clip(img_final, 0, 255).astype(np.uint8)
    #return img_final
    cv.imwrite('{}_laplaciano.png'.format(nome_arquivo), img_final)

def detector_bordas_roberts(img, nome_arquivo):
    h1 = np.array([[1,0],
              [0,-1]])
    h2 = np.array([[0,1],
              [-1,0]])
    img = img.astype(np.float32)
    img_final = np.zeros_like(img)

    for i in range(1, img.shape[0]-1):
        for j in range(1, img.shape[1]-1):
            dx = 0
            dy = 0
            for di in range(-1, 1):
                for dj in range(-1, 1):
                    dx += img[i+di, j+dj] * h1[di+1,dj+1]
                    dy += img[i+di, j+dj] * h2[di+1,dj+1]
            magnitude = np.sqrt(dx**2 + dy**2)
            img_final[i,j] = magnitude
    
    img_final = np.clip(img_final, 0, 255).astype(np.uint8)
    #return img_final
    cv.imwrite('{}_detector_roberts.png'.format(nome_arquivo), img_final)

def detector_bordas_prewitt(img, nome_arquivo):
    h1 = np.array([[-1,-1,-1],
              [0,0,0],
              [1,1,1]])
    h2 = np.array([[-1,0,1],
              [-1,0,1],
              [-1,0,1]])
    img = img.astype(np.float32)
    img_final = np.zeros_like(img)

    for i in range(1, img.shape[0]-1):
        for j in range(1, img.shape[1]-1):
            dx = 0
            dy = 0
            for di in range(-1, 2):
                for dj in range(-1, 2):
                    dx += img[i+di, j+dj] * h1[di+1,dj+1]
                    dy += img[i+di, j+dj] * h2[di+1,dj+1]
            magnitude = np.sqrt(dx**2 + dy**2)
            img_final[i,j] = magnitude
    
    img_final = np.clip(img_final, 0, 255).astype(np.uint8)
    #return img_final
    cv.imwrite('{}_detector_prewitt.png'.format(nome_arquivo), img_final)

def detector_bordas_sobel(img, nome_arquivo):

    h1 = np.array([[-1,-2,-1],
              [0,0,0],
              [1,2,1]])
    
    h2 = np.array([[-1,0,1],
              [-2,0,2],
              [-1,0,1]])
    
    img = img.astype(np.float32)
    img_final = np.zeros_like(img)

    for i in range(1, img.shape[0]-1):
        for j in range(1, img.shape[1]-1):
            dx = 0
            dy = 0
            for di in range(-1, 2):
                for dj in range(-1, 2):
                    dx += img[i+di, j+dj] * h1[di+1,dj+1]
                    dy += img[i+di, j+dj] * h2[di+1,dj+1]
            magnitude = np.sqrt(dx**2 + dy**2)
            img_final[i,j] = magnitude

    img_final = np.clip(img_final, 0, 255).astype(np.uint8)
    #return img_final
    cv.imwrite('{}_detector_sobel.png'.format(nome_arquivo), img_final)

def main():
    #nome_arqv = input('Digite o nome do arquivo: ')
    img = rgb_para_cinza(cv.imread(sys.argv[1]))
    nome_sem_extensao = sys.argv[1].split('.')[0]
    suavizacao_md_vizinhanca(img, 5, nome_sem_extensao)
    suavizacao_kvizinhos(img, 5, 9, nome_sem_extensao)
    suavizacao_mediana(img, 5, nome_sem_extensao)
    laplaciano(img, nome_sem_extensao)
    detector_bordas_roberts(img, nome_sem_extensao)
    detector_bordas_prewitt(img, nome_sem_extensao)
    detector_bordas_sobel(img, nome_sem_extensao)
    
    #cv.imshow('Imagem', img)
    #cv.waitKey(0)
    #cv.destroyAllWindows()
    
if __name__ == '__main__':
    main()
    