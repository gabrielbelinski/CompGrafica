import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

def rgb_para_cinza(imagem):
    imagem = imagem[:,:,0]/3 + imagem[:,:,1]/3 + imagem[:,:,2]/3
    return imagem.astype(np.uint8)

def inversao(imagem):
    imagem = 255 - imagem
    return imagem.astype(np.uint8)

def normalizacao(imagem):
    c = float(input("Digite o valor de c: "))
    d = float(input("Digite o valor de d: "))
    if c == 0:
        imagem = (imagem - imagem.min()) * (d / (imagem.max() - imagem.min())) + c
    else:
        imagem = (imagem - imagem.min()) * ((d-c) / (imagem.max() - imagem.min())) + c
    return imagem.astype(np.uint8)

def operador_logaritmico(imagem):
    imagem = imagem.astype(np.float32)
    c = 255 / np.log(1+imagem.max())
    imagem = c * np.log(1+np.abs(imagem))
    return normalizacao(imagem)

def operador_potencia(imagem):
    c = float(input("Digite o valor de c: "))
    gama = float(input("Digite o valor de gama: "))
    imagem = c*np.power(imagem,gama)
    return normalizacao(imagem)

def fatiamento(imagem):
    imagem = (imagem > 128) * 255
    return imagem.astype(np.uint8)

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


def normaliza_vetor(vetor):
    vetor_normalizado = np.zeros(256, dtype=float)
    total_frequencias = np.sum(vetor)
    for i in range(256):
        vetor_normalizado[i] = vetor[i]/total_frequencias

    return vetor_normalizado


def main():
    nome_arquivo = input("Digite o nome do arquivo: ")
    img = cv.imread(nome_arquivo)
    print('Imagem {}: {} {}'.format(nome_arquivo, img.shape,img.dtype))
    print('1 - RGB para niveis de cinza', '\n2 - Inversao', '\n3 - Normalizacao', '\n4 - Operador logaritmico', '\n5 - Operador potencia', '\n6 - Fatiamento', '\n7 - Histograma\n')

    op = int(input('Digite o numero da operacao desejada: '))
    match op:
        case 1:
            img = rgb_para_cinza(img)
        case 2:
            img = inversao(img)
        case 3:
            img = normalizacao(img)
        case 4:
            img = operador_logaritmico(img)
        case 5:
            img = operador_potencia(img)
        case 6:
            img = fatiamento(img)
        case 7:
            histograma(img, nome_arquivo)
            #histograma(img[:, :, 0], nome_arquivo)
            #histograma(img[:, :, 1], nome_arquivo)
            #histograma(img[:,:,2], nome_arquivo)
        case _:
            print("Opcao invalida")
            op = int(input('Digite o numero da operacao desejada: '))

    cv.imshow(nome_arquivo, img)
    cv.waitKey(0)
    cv.destroyAllWindows()

if __name__ == '__main__':
    while True:
        main()
        continua_execucao = int(input("Voce deseja executar o programa novamente? 1 - Sim, 2 - Nao\n"))
        if continua_execucao == 2:
            break
