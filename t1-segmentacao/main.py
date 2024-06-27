#===============================================================================
# Exemplo: segmentação de uma imagem em escala de cinza.
#-------------------------------------------------------------------------------
# Autor: Bogdan T. Nassu
# Universidade Tecnológica Federal do Paraná
#=============================================================================== <-------INTEGRANTES AQUI
# Nome: Luiz Gustavo dos Santos RA: 2086905
# Nome: Johny Kwiatkowski Oh    RA: 2090333
#===============================================================================

import sys
import timeit
import numpy as np
import cv2

#===============================================================================

INPUT_IMAGE =  'arroz.bmp'

# TODO: ajuste estes parâmetros!
NEGATIVO = False
THRESHOLD = 0.8
ALTURA_MIN = 10
LARGURA_MIN = 10
N_PIXELS_MIN = 25

#===============================================================================

# tamanho máximo da imagem (usado para comparar encontrar elemento de cada blob)
# MAX_WIDTH = 999999
# MAX_HEIGHT = 999999

# classe auxiliar utilizada para encapsular o conteúdo do dicionário
class Blob:
    def __init__(self, label, min_y, min_x):
        self.label = label
        self.n_pixels = 1
        self.max_x = -1
        self.min_x = min_x
        self.max_y = -1
        self.min_y = min_y
    def get_blob_as_dict(self):
        return {
            "label": self.label,
            "n_pixels": self.n_pixels,
            "T": self.min_y,
            "L": self.min_x,
            "B": self.max_y,
            "R": self.max_x
        }
    def big_enough(self, min_width, min_height, min_pixels):
        return (self.max_x - self.min_x) >= min_width and (self.max_y - self.min_y) >= min_height and self.n_pixels > min_pixels
    # atualiza os valores mínimos e máximos caso necessário
    def try_update_bounds(self, y, x):
        if x < self.min_x:
            self.min_x = x    
        if x > self.max_x:
            self.max_x = x    
        if y < self.min_y:
            self.min_y = y    
        if y > self.max_y:
            self.max_y = y 

#===============================================================================

def binariza (img, threshold):
    ''' Binarização simples por limiarização.

Parâmetros: img: imagem de entrada. Se tiver mais que 1 canal, binariza cada
              canal independentemente.
            threshold: limiar.
            
Valor de retorno: versão binarizada da img_in.'''

    # TODO: escreva o código desta função.
    # Dica/desafio: usando a função np.where, dá para fazer a binarização muito
    # rapidamente, e com apenas uma linha de código!

    return np.where(img < threshold, np.float32(0), np.float32(1))
#-------------------------------------------------------------------------------
def rotula (img, largura_min, altura_min, n_pixels_min):
    '''Rotulagem usando flood fill. Marca os objetos da imagem com os valores
[0.1,0.2,etc].

Parâmetros: img: imagem de entrada E saída.
            largura_min: descarta componentes com largura menor que esta.
            altura_min: descarta componentes com altura menor que esta.
            n_pixels_min: descarta componentes com menos pixels que isso.

Valor de retorno: uma lista, onde cada item é um vetor associativo (dictionary)
com os seguintes campos:

'label': rótulo do componente.
'n_pixels': número de pixels do componente.
'T', 'L', 'B', 'R': coordenadas do retângulo envolvente de um componente conexo,
respectivamente: topo, esquerda, baixo e direita.'''
    # TODO: escreva esta função.
    # Use a abordagem com flood fill recursivo.
    
    # dimensões da imagem
    max_y = img.shape[0]
    max_x = img.shape[1]
    output = np.zeros((max_y, max_x), dtype=int)
    # lista contendo os dicionários
    data = []
    
    label = 1
    for y in range(1, max_y - 1):
        for x in range(1, max_x - 1):
            index_valid_and_unfilled = img[y][x] == 1 and output[y][x] == 0
            if index_valid_and_unfilled:
                # * Done
                #  O negócio de inicializar os extremos mínimos com 999999 até
                #  funciona aqui, mas vale lembrar que, a rigor, não é seguro.
                #  Eu iniciaria com as coordenadas do 1o pixel encontrado para o
                #  blob!
                blob = Blob(label, y, x)
                inunda(label, output, img, y, x, blob)
                if blob.big_enough(largura_min, altura_min, n_pixels_min):
                    data.append(blob.get_blob_as_dict())
                # * Done
                #  Seria melhor ter a linha 116 fora do if. Do jeito que está
                #  até funciona, mas se fosse usar a imagem rotulada para algo
                #  depois, vocês poderiam ter vários objetos muito pequenos
                #  (ruído) com o mesmo rótulo de um objeto válido.
                label += 1
    return data

#===============================================================================
        
    
def inunda(label, output, img, y, x, blob):
    max_y = img.shape[0]
    max_x = img.shape[1]
    
    # * Done
    # Esse teste da already_filled não está redundante? Digo, vocês só descem na
    # recursão quando output[y+j][x+i] == 0. Aliás, vocês estão primeiro olhando
    # esta posição para só depois conferir se ela estava dentro da imagem!!!
    in_bounds = y >= 0 and y < max_y and x >= 0 and x < max_x
    already_filled = output[y][x] != 0
    black_pixel = img[y][x] == 1
    if not in_bounds or already_filled or not black_pixel:
        return 0
    
    blob.n_pixels += 1   
    output[y][x] = label
    blob.try_update_bounds(y, x)

    for j in range(-1, 2, 1):
        for i in range(-1, 2, 1):
            # unfilled_and_black = output[y + j][x + i] == 0 and img[y + j][x + i] == 1
            # if in_bounds and unfilled_and_black:
                # blob.n_pixels += 1   
                inunda(label, output, img, y + j, x + i, blob)
                
def main ():

    # Abre a imagem em escala de cinza.
    img = cv2.imread (INPUT_IMAGE, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print ('Erro abrindo a imagem.\n')
        sys.exit ()

    # É uma boa prática manter o shape com 3 valores, independente da imagem ser
    # colorida ou não. Também já convertemos para float32.
    img = img.reshape ((img.shape [0], img.shape [1], 1))
    img = img.astype (np.float32) / 255

    # Mantém uma cópia colorida para desenhar a saída.
    img_out = cv2.cvtColor (img, cv2.COLOR_GRAY2BGR)

    # Segmenta a imagem.
    if NEGATIVO:
        img = 1 - img
    img = binariza (img, THRESHOLD)
    cv2.imshow ('01 - binarizada', img)
    cv2.imwrite ('01 - binarizada.png', img*255)

    start_time = timeit.default_timer ()
    componentes = rotula (img, LARGURA_MIN, ALTURA_MIN, N_PIXELS_MIN)
    n_componentes = len (componentes)
    print ('Tempo: %f' % (timeit.default_timer () - start_time))
    print ('%d componentes detectados.' % n_componentes)

    # Mostra os objetos encontrados.
    for c in componentes:
        cv2.rectangle (img_out, (c ['L'], c ['T']), (c ['R'], c ['B']), (0,0,1))

    cv2.imshow ('02 - out', img_out)
    cv2.imwrite ('02 - out.png', img_out*255)
    cv2.waitKey ()
    cv2.destroyAllWindows ()


if __name__ == '__main__':
    main ()

#===============================================================================
