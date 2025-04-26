import numpy as np
import cv2 as cv
import queue as qe

def rgb_para_cinza(imagem):
    imagem = imagem[:,:,0]/3 + imagem[:,:,1]/3 + imagem[:,:,2]/3
    return imagem.astype(np.uint8)

def normalizacao(imagem, c=None, d=None):
    if c == None and d == None:
        imagem = (imagem - imagem.min()) * (255 / (imagem.max() - imagem.min())) 
    else:
        imagem = (imagem - imagem.min()) * ((d-c) / (imagem.max() - imagem.min())) + c
    return imagem.astype(np.uint8)

def salva_resultados(lista_resultados, n_exercicio):
    for i in range(len(lista_resultados)):
        cv.imwrite(f'exercicio{n_exercicio}_{i+1}.png', lista_resultados[i])

def mediana(img, tam_janela):
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
    return img_final

def loop_filtra_mediana(img, qtd_aplicacoes, tam_janela):
    resultados = []
    nova_img = rgb_para_cinza(img)
    for i in range(0, qtd_aplicacoes):
        nova_img = mediana(nova_img, tam_janela)
        resultados.append(nova_img)
    salva_resultados(resultados, 1)

def aplica_convolucao(img, mascara):
    pad_a = len(mascara) //2
    pad_l = len(mascara[0]) //2
    resultado = np.zeros_like(img)

    for x in range(pad_a, img.shape[0]-pad_a):
        for y in range(pad_l, img.shape[1]-pad_l):
            soma = 0
            for kx in range(0, len(mascara)):
                for ky in range(0, len(mascara[0])):
                    soma += img[x+kx-pad_a][y+ky-pad_l] * mascara[kx][ky]
            resultado[x][y] = soma

    return resultado

def limiarizacao(imagem, limiar):
    img_final = np.zeros_like(imagem)
    for i in range(imagem.shape[0]):
        for j in range(imagem.shape[1]):
            if imagem[i][j] > limiar:
                img_final[i][j] = 255
            else:
                img_final[i][j] = 0

    return img_final

def filtro_ptsisolados(img, limiar):
    img = rgb_para_cinza(img)
    h1 = np.array([[-1,-1,-1],
              [-1,8,-1],
              [-1,-1,-1]], dtype=np.float32)
    img = img.astype(np.float32)
    img_filtrada = aplica_convolucao(img, h1)
    img_bin = limiarizacao(img_filtrada, limiar)
    return img_bin.astype(np.uint8)
    
def deteccao_linhas(img, limiar):
    img = rgb_para_cinza(img)
    mascaras = {'Horizontal': np.array([[-1,-1,-1],
              [2,2,2],
              [-1,-1,-1]], dtype=np.float32),
                'Vertical': np.array([[-1,2,-1],
              [-1,2,-1],
              [-1,2,-1]], dtype=np.float32),
                'Mais45°': np.array([[-1,-1,2],
              [-1,2,-1],
              [2,-1,-1]], dtype=np.float32),
                'Menos45°': np.array([[2,-1,-1],
              [-1,2,-1],
              [-1,-1,2]], dtype=np.float32)
            }
    nova_img = np.zeros_like(img)
    resultados = []
    resultados.append(img)
    for tipo, msc in mascaras.items():
        img_conv = aplica_convolucao(img, msc)
        img_bin = limiarizacao(normalizacao(img_conv), limiar)
        resultados.append(img_bin)

    for r in range(1, len(resultados)):
        nova_img = np.logical_or(nova_img, resultados[r])
    
    nova_img = normalizacao(nova_img.astype(np.uint8))
    resultados.append(nova_img)
    salva_resultados(resultados, 3)

def get_coordinates(img):
    coordinates = []

    def mouse_callback(event, x, y, flags, param):
        if event == cv.EVENT_LBUTTONDOWN: 
            coordinates.extend([x, y])
            print(f"Coordenadas selecionadas: (x={x}, y={y})")
            cv.destroyAllWindows() 
    
    cv.imshow('Selecione um ponto', img)
    cv.setMouseCallback('Selecione um ponto', mouse_callback)
    
    while len(coordinates) < 2:
        cv.waitKey(20)
    
    return (coordinates[0], coordinates[1])

def cresc_regiao(img, seedx, seedy, limiar):
    if len(img.shape) == 3:
        img_cinza = rgb_para_cinza(img)
    else:
        img_cinza = img.copy()
    vizx = [1, -1, 0, 0]  
    vizy = [0, 0, 1, -1]
    visitado = np.zeros(img_cinza.shape, dtype=bool)
    
    valor_semente = img_cinza[seedy, seedx]
    fila1 = qe.Queue()
    fila2 = qe.Queue()
    visitado[seedy, seedx] = True
    fila1.put(seedx)
    fila2.put(seedy)
    mascara = np.zeros_like(img_cinza)

    while not fila1.empty():
        x = fila1.get()
        y = fila2.get()
        mascara[y, x] = 255
        for i in range(len(vizx)):   
            nx = x + vizx[i]
            ny = y + vizy[i]
            if (0 <= nx < img_cinza.shape[1] and 0 <= ny < img_cinza.shape[0] and not visitado[ny, nx]):
                if abs(int(img_cinza[ny, nx]) - int(valor_semente)) < limiar:
                    visitado[ny, nx] = True
                    fila1.put(nx)
                    fila2.put(ny)
    
    if len(img.shape) == 3:
        img[mascara == 255] = [0, 0, 255]
        resultado = img
    else:
        nova_img = cv.cvtColor(img_cinza, cv.COLOR_BGR2RGB)
        nova_img[mascara == 255] = [0, 0, 255]
        resultado = nova_img
                
    return resultado.astype(np.uint8)

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

def main():
    nomes_arquivos = ['circuito.tif', 'pontos.png', 'linhas.png', 
                       'igreja.png', 'root.jpg', 'harewood.jpg', 'nuts.jpg', 'snow.jpg', 'img_gabriel.jpg']
    imgs_lidas = [cv.imread(r) for r in nomes_arquivos]

    loop_filtra_mediana(imgs_lidas[0], 3, 5)

    cv.imwrite('exercicio2.png', filtro_ptsisolados(imgs_lidas[1], 150))

    deteccao_linhas(imgs_lidas[2], 80)

    nova_imagem4 = cv.Canny(imgs_lidas[3], 100, 230)
    cv.imwrite('exercicio4.png', nova_imagem4)

    seed_x, seed_y = get_coordinates(imgs_lidas[4])
    img_cinza4 = rgb_para_cinza(imgs_lidas[4])
    cv.imwrite('exercicio5.png', cresc_regiao(img_cinza4, seed_x, seed_y, 60))
    
    resultados_ex6 = [alg_otsu(a) for a in imgs_lidas[5:]]
    salva_resultados(resultados_ex6, 6)

if __name__ == '__main__':
    main()