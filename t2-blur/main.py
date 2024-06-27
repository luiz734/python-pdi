# ---------------------------------------------------------------------------- #
# Nome: Luiz Gustavo dos Santos RA: 2086905
# Nome: Johny Kwiatkowski Oh    RA: 2090333
# ---------------------------------------------------------------------------- #

import sys
import timeit
import numpy as np
import cv2

INPUT_IMAGE = 'exemplos/a01 - Original.bmp'
# INPUT_IMAGE = 'exemplos/b01 - Original.bmp'
# INPUT_IMAGE = 'xxx.bmp'
CHANNELS = 3
WINDOW_SIZE_X = 3
WINDOW_SIZE_Y = 13

def blur_ingenuo(img):
   max_y = img.shape[0]
   max_x = img.shape[1]
   output = np.empty((max_y, max_x, CHANNELS), dtype=np.float32)

   half_window_y = WINDOW_SIZE_Y // 2
   half_window_x = WINDOW_SIZE_X // 2
   # ignora a margem
   for y in range(0 + half_window_y, max_y - half_window_y, 1):
      for x in range(0 + half_window_x, max_x - half_window_x, 1):
         for c in range(0, CHANNELS, 1):
            # usando slices (mais rápido)
            window = img[y - half_window_y: y + half_window_y + 1:1, x - half_window_x: x + half_window_x + 1:1, c]
            window_sum = np.sum(window)
            output[y, x, c] = window_sum / (WINDOW_SIZE_X * WINDOW_SIZE_Y)

            # usando loops (MUITO lento, porém legível. Faz a mesma coisa)
            # window_sum = 0
            # for j in range(-half_window, half_window + 1, 1):
            #    for i in range(-half_window, half_window + 1, 1):
            #       window_sum += img[y + j][x + i]
            # output[y][x] = window_sum / (WINDOW_SIZE * WINDOW_SIZE)

   return output

def blur_filtro_separado(img):
   max_y = img.shape[0]
   max_x = img.shape[1]
   # buffer auxiliar necessário
   buffer = np.empty((max_y, max_x, CHANNELS), dtype=np.float32)
   output = np.empty((max_y, max_x, CHANNELS), dtype=np.float32)
      
   half_window_y = WINDOW_SIZE_Y // 2
   half_window_x = WINDOW_SIZE_X // 2
   # horizontal 
   for y in range(0 , max_y, 1):
      for x in range(0 + half_window_x, max_x - half_window_x, 1):
         for c in range(0, CHANNELS, 1):
            horizontal_window = img[y, x - half_window_x: x + half_window_x + 1: 1, c] 
            sum_horizontal = np.sum(horizontal_window)
            buffer[y, x, c] = sum_horizontal / WINDOW_SIZE_X

   # vertical
   for y in range(0 + half_window_y, max_y - half_window_y, 1):
      for x in range(0, max_x, 1):
         for c in range(0, CHANNELS, 1):
            vertical_window = buffer[y - half_window_y: y + half_window_y + 1: 1, x, c]
            sum_vertical = np.sum(vertical_window)
            output[y, x, c] = sum_vertical / WINDOW_SIZE_Y

   return output

def blur_integral(img):
   max_y = img.shape[0]
   max_x = img.shape[1]
   buffer = np.empty((max_y, max_x, CHANNELS), dtype=np.float32)
   output = np.empty((max_y, max_x, CHANNELS), dtype=np.float32)

   # copia os elementos da primeira linha/coluna da imagem para o buffer
   buffer[0,:,:] = img[0,:,:]
   buffer[:,0,:] = img[:,0,:]

   # cria a imagem integral
   for y in range(0, max_y, 1):
      for x in range(0, max_x, 1):
         for c in range(0, CHANNELS, 1):
            buffer[y, x, c] = img[y, x, c]
            if y > 0:
               buffer[y, x, c] += buffer[y - 1, x, c]
            if x > 0:
               buffer[y, x, c] += buffer[y, x - 1, c]
            if y > 0 and x > 0:
               buffer[y, x, c] -= buffer[y - 1, x - 1, c]
         
   half_window_y = WINDOW_SIZE_Y // 2
   half_window_x = WINDOW_SIZE_X // 2

   for y in range(0, max_y, 1):
      for x in range(0, max_x, 1):
         # min: as coordenadas (cima-1, esquerda-1) da janela
         min_index_y = np.clip(y - half_window_y - 1, 0, max_y - 1)
         min_index_x = np.clip(x - half_window_x - 1, 0, max_x - 1)
         # max: as coordenadas (baixo, direita) da janela
         max_index_y = np.clip(y + half_window_y, 0, max_y - 1)
         max_index_x = np.clip(x + half_window_x, 0, max_x - 1)

         for c in range(0, CHANNELS, 1):
            # soma total até o canto inferior direito da janela 
            full_rect_sum =  buffer[max_index_y, max_index_x, c]

            # as sobras podem ou não existir (não existe quando os lados
            # esquerdo-1 ou cima-1 da janela ultrapassam os limites da imagem.
            # Estes casos devem ser tratados separadamente)
            went_over_left = x - half_window_x - 1 < 0
            went_over_top = y - half_window_y - 1 < 0
            went_over_some = went_over_left or went_over_top

            leftover_x = leftover_y = leftover_xy = 0
            if not went_over_left:
               leftover_x = buffer[max_index_y, min_index_x, c]
            if not went_over_top:
               leftover_y = buffer[min_index_y, max_index_x, c]
            if not went_over_some:
               leftover_xy = buffer[min_index_y, min_index_x, c]
            
            window_sum = full_rect_sum - leftover_x - leftover_y + leftover_xy
            # total de pixels na janela
            window_size_x = (max_index_x - min_index_x)
            window_size_y = (max_index_y - min_index_y)
            # tratamento de bordas é necessário, pois as variáveis "min_index_?"
            # sempre consideram como mínimo o pixel imediatamente acima e à
            # esquerda da janela, mas isto NÃO é válido quando a janela é menor que
            # WINDOW_SIZE. Neste caso, o "min_index_?" será o elemento no canto
            # superior esquerdo DENTRO da janela, portanto, é necessário somar 1
            window_size_x = window_size_x + 1 if x <= half_window_x else window_size_x
            window_size_y = window_size_y + 1 if y <= half_window_y else window_size_y

            window_size = window_size_x * window_size_y
            output[y][x][c] = window_sum / window_size

   return output

# função auxiliar utilizada para medir o tempo de execução
def measure(blur_algorithm, img):
   # Cópia para saída
   img_out = img.copy() 
   cv2.imshow ('input', img)

   start_time = timeit.default_timer ()
   img_out = blur_algorithm(img);
   print (f'{blur_algorithm.__name__}: {(timeit.default_timer () - start_time)}')

   cv2.imshow (f'out - {blur_algorithm.__name__}', img_out)
   cv2.imwrite (f'out - {blur_algorithm.__name__}.png', img_out*255)

def main ():
   # Abre a imagem.
   img = cv2.imread (INPUT_IMAGE, cv2.IMREAD_COLOR)
   if img is None:
      print ('Erro abrindo a imagem.\n')
      sys.exit ()
   # Converte para float
   img = img.astype (np.float32) / 255

   
   # --------------------------------- Execução --------------------------------- #
   measure(blur_ingenuo, img)
   measure(blur_filtro_separado, img)
   measure(blur_integral, img)
   # ---------------------------------------------------------------------------- #


   cv2.waitKey ()
   cv2.destroyAllWindows ()

if __name__ == '__main__':
   main ()