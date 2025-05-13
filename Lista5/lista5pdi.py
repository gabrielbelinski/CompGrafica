import cv2 as cv
import numpy as np

EE1 = np.array([[1,1,1],
                [1,1,1],
                [1,1,1]])
EE2 = np.array([[0,1,0],
                [1,1,1],
                [0,1,0]])

def erosao(img, ee):
    img_altura, img_largura = img.shape
    ee_altura, ee_largura = ee.shape
    offset_h, offset_w = ee_altura // 2, ee_largura // 2
    nova_img = np.zeros_like(img)
        
    for i in range(offset_h, img_altura - offset_h):
        for j in range(offset_w, img_largura - offset_w):
            vizinhanca = img[i-offset_h:i+offset_h+1, j-offset_w:j+offset_w+1]
            if np.all(vizinhanca[ee == 1] == 1):
                nova_img[i, j] = 1
                    
    return nova_img.astype(np.uint8)

def dilatacao(img, ee):
    img_altura, img_largura = img.shape
    altura_ee, largura_ee = ee.shape
    offset_h, offset_w = altura_ee // 2, largura_ee // 2
    nova_img = np.zeros_like(img)
    
    for i in range(offset_h, img_altura - offset_h):
        for j in range(offset_w, img_largura - offset_w):
            vizinhanca = img[i-offset_h:i+offset_h+1, j-offset_w:j+offset_w+1]
            if np.any(vizinhanca[ee == 1] == 1):
                nova_img[i, j] = 1
                
    return nova_img.astype(np.uint8)

def abertura(img, ee):
    img_e = erosao(img, ee)
    img_d = dilatacao(img_e, ee)
    return img_d

def fechamento(img, ee):
    img_d = dilatacao(img, ee)
    img_e = erosao(img_d, ee)
    return img_e

def extracao_fronteiras(img, ee):
    img_er = erosao(img, ee)
    img_dl = dilatacao(img, ee)
    bordas_internas = img - img_er
    bordas_externas = img_dl - img
    return bordas_internas, bordas_externas

def preenchimento_regiao(img, ponto_semente=None, cor_preenchimento=255, tolerancia=10):
    if ponto_semente is None:
        contornos, _ = cv.findContours(img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        if len(contornos) > 0:
            maior_contorno = max(contornos, key=cv.contourArea)
            M = cv.moments(maior_contorno)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                ponto_semente = (cx, cy)
            else:
                ponto_semente = (img.shape[1]//2, img.shape[0]//2)
        else:
            ponto_semente = (img.shape[1]//2, img.shape[0]//2)
    
    img_preenchida = flood_fill_manual(img, ponto_semente, cor_preenchimento)
    return img_preenchida

def flood_fill_manual(img, seed, new_value):
    img_filled = img.copy()
    height, width = img.shape
    
    old_value = img[seed[1], seed[0]]
    
    if old_value == new_value:
        return img_filled
    
    stack = [seed]
    
    while stack:
        x, y = stack.pop()
        
        if x < 0 or x >= width or y < 0 or y >= height:
            continue
            
        if img_filled[y, x] == old_value:
            img_filled[y, x] = new_value
            
            stack.append((x + 1, y))
            stack.append((x - 1, y))
            stack.append((x, y + 1))
            stack.append((x, y - 1))
    
    return img_filled

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
   
def extrai_componente(img, pt_inicial):
    _, imag = cv.threshold(img, 127, 255, cv.THRESH_BINARY)

    xk = np.zeros_like(imag)
    xk[pt_inicial] = 255
    
    b = cv.getStructuringElement(cv.MORPH_CROSS, (50,50))
    while True:
        xk_prev = xk.copy()
        dl = cv.dilate(xk_prev, b)
        xk = cv.bitwise_and(dl, imag)
        if np.array_equal(xk,xk_prev):
            break
    
    xk_col = cv.cvtColor(xk, cv.COLOR_GRAY2BGR)
    xk_col[xk == 255] = (0, 255, 255)

    return imag, xk_col

def main():
    imagemex1 = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ])

    nomes_arquivos = ['quadrados.png', 'ruidos.png', 'cachorro.png', 'gato.png', 'img_gabriel.jpg']
    lista_ee = [EE1, EE2]
    lista_imgs_ex1 = []
    
    for elemento in lista_ee:
        imagem_er = erosao(imagemex1, elemento)
        imagem_dl = dilatacao(imagemex1, elemento)
        img_er_aj, img_dl_aj = cv.resize(imagem_er * 255, (400, 240), interpolation=cv.INTER_NEAREST), cv.resize(imagem_dl * 255, (400, 240), interpolation=cv.INTER_NEAREST)
        lista_imgs_ex1.append(img_er_aj)
        lista_imgs_ex1.append(img_dl_aj)
                        
    for i, img in enumerate(lista_imgs_ex1):
        cv.imwrite(f'img_ex1_{i}.png', img)
    
    img_quadrados = cv.imread(nomes_arquivos[0], cv.THRESH_BINARY)
    ee_quadrados = cv.getStructuringElement(cv.MORPH_RECT, (50, 50))
    img_quadrados_erodida = cv.erode(img_quadrados, ee_quadrados, iterations=1)
    img_quadrados_dilatada = cv.dilate(img_quadrados_erodida, ee_quadrados, iterations=1)
    cv.imwrite('ex2_eros.png', img_quadrados_erodida)
    cv.imwrite('ex2_dil.png', img_quadrados_dilatada)

    img_ruidos = cv.imread(nomes_arquivos[1], cv.IMREAD_GRAYSCALE)
    _, img_bin = cv.threshold(img_ruidos, 127, 255, cv.THRESH_BINARY)
    img_bin = img_bin // 255
    ee_ruidos = cv.getStructuringElement(cv.MORPH_RECT, (35,35))
    ruidos_ab = abertura(img_bin, ee_ruidos)
    ruidos_fc = fechamento(ruidos_ab, ee_ruidos)
    cv.imwrite('ex3_ab.png', ruidos_ab * 255)
    cv.imwrite('ex3_fc.png', ruidos_fc * 255)
    
    img_cachorro = cv.imread(nomes_arquivos[2], cv.IMREAD_GRAYSCALE)
    _, img_bin_ca = cv.threshold(img_cachorro, 127, 255, cv.THRESH_BINARY)
    img_bin_ca = img_bin_ca // 255
    ee_cachorro = cv.getStructuringElement(cv.MORPH_RECT, (15,15))
    borda_int_cachorro, borda_ext_cachorro = extracao_fronteiras(img_bin_ca, ee_cachorro)
    cv.imwrite('ex4_interna.png', borda_int_cachorro * 255)
    cv.imwrite('ex4_externa.png', borda_ext_cachorro * 255)

    img_gato = cv.imread(nomes_arquivos[3], cv.IMREAD_GRAYSCALE)
    _, img_bin_gato = cv.threshold(img_gato, 127, 255, cv.THRESH_BINARY)
    img_gato_preenchida = preenchimento_regiao(img_bin_gato)
    cv.imwrite('ex5_preenchido.png', img_gato_preenchida)

    img_qds = cv.imread(nomes_arquivos[0], cv.IMREAD_GRAYSCALE)
    x, y = get_coordinates(img_qds)
    img_qds_destacada, img_qd_isolado = extrai_componente(img_qds, (x,y))
    cv.imwrite('ex6_destacada.png', img_qds_destacada)
    cv.imwrite('ex6_isolado.png', img_qd_isolado)

    img_gabriel = cv.imread(nomes_arquivos[4], cv.IMREAD_GRAYSCALE)
    ee_gabriel = cv.getStructuringElement(cv.MORPH_RECT, (50,50))
    img_gabriel_erodida = cv.erode(img_gabriel, ee_gabriel, iterations=1)
    cv.imwrite('ex7_erosao.png', img_gabriel_erodida)
    img_gabriel_dilatada = cv.dilate(img_gabriel, ee_gabriel, iterations=1)
    cv.imwrite('ex7_dilatacao.png', img_gabriel_dilatada)
    img_gabriel_final = img_gabriel_dilatada - img_gabriel_erodida
    cv.imwrite('ex7_gradiente.png', img_gabriel_final)
    
if __name__ == '__main__':
    main()