import numpy as np
import cv2 as cv


def rgb_para_cinza(imagem):
    imagem = imagem[:,:,0]/3 + imagem[:,:,1]/3 + imagem[:,:,2]/3
    return imagem.astype(np.uint8)

def inversao(imagem):
    imagem = 255 - imagem
    return imagem.astype(np.uint8)

def normalizacao(imagem, c=None, d=None):
    if c == None and d == None:
        imagem = (imagem - imagem.min()) * (255 / (imagem.max() - imagem.min())) 
    else:
        imagem = (imagem - imagem.min()) * ((d-c) / (imagem.max() - imagem.min())) + c
    return imagem.astype(np.uint8)

def limiarizacao(imagem, limiar):
    img_final = np.zeros_like(imagem)
    for i in range(imagem.shape[0]):
        for j in range(imagem.shape[1]):
            if imagem[i][j] > limiar:
                img_final[i][j] = 255
            else:
                img_final[i][j] = 0

    return img_final

def alg_otsu(img):
    img = rgb_para_cinza(img)
    histograma = [0 for _ in range(256)]
    total_pixels = img.shape[1] * img.shape[0]
    limiar_ot = 0
    var_max = 0

    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            intensidade = img[x][y]
            histograma[intensidade] += 1
    
    for i in range(0, 256):
        histograma[i] = histograma[i]/total_pixels

    for t in range(0, 256):
        w1 = np.sum(histograma[0:t])
        w2 = np.sum(histograma[t:256])
        if w1 == 0 and w2 == 0:
            continue
        media1 = 0
        media2 = 0

        for i in range(0, t):
            media1 += i * histograma[i]
        media1 = media1/w1 if w1 != 0 else 0
        for i in range(t, 256):
            media2 += i * histograma[i]
        media2 = media2/w2 if w2 != 0 else 0

        var = w1 * w2 * (media1 - media2)**2
        if var > var_max:
            var_max = var
            limiar_ot = t

    return limiarizacao(img, limiar_ot)

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

def histograma(imagem, nome_arquivo):
    vetor = np.zeros(256, dtype=int)
    lista_cores = [i for i in range(256)]
    for i in range(256):
        vetor[i] = np.sum(imagem == i)
    opcao = int(input("Selecione o tipo de histograma desejado:\n1-Histograma\n2-Histograma Normalizado\n3-Histograma Acumulado\n4-Histograma Acumulado Normalizado\n5-Equalizacao histograma\n"))
    match opcao:
        case 1:
            plt.bar(lista_cores, vetor)
            plt.title("Histograma")
        case 2:
            vetor = normaliza_vetor(vetor)
            plt.bar(lista_cores, vetor)
            plt.title("Histograma normalizado")
        case 3:
            vetor = np.cumsum(vetor)
            plt.bar(lista_cores, vetor)
            plt.title("Histograma acumulado")
        case 4:
            vetor = normaliza_vetor(np.cumsum(vetor))
            plt.bar(lista_cores, vetor)
            plt.title("Histograma acumulado normalizado")
        case 5:
            histograma_acumulado = np.cumsum(vetor)
            if len(imagem.shape) == 3:
                imagem = rgb_para_cinza(imagem)
            min_ha = min(histograma_acumulado) 
            T = (histograma_acumulado-min_ha) / ((imagem.shape[0]*imagem.shape[1]) - min_ha) * (256-1)
            img_equalizada = T[imagem]
            img_equalizada = (255 * (img_equalizada/img_equalizada.max())).astype(np.uint8)
            cv.imshow('Equalizacao de histograma', img_equalizada)
            cv.waitKey(0)
            cv.destroyAllWindows()

        case _:
            print("Opcao invalida\n")
            opcao = int(input(
                "Selecione o tipo de histograma desejado:\n1-Histograma\n2-Histograma Normalizado\n3-Histograma Acumulado\n4-Histograma Acumulado Normalizado\n5-Equalizacao de histograma\n"))

    if opcao != 5:
        plt.xlabel("Intensidade")
        plt.ylabel("Frequencia")
        #plt.savefig("histograma{}".format(nome_arquivo))
        plt.show()