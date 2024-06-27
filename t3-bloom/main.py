# ---------------------------------------------------------------------------- #
# Nome: Luiz Gustavo dos Santos RA: 2086905
# Nome: Johny Kwiatkowski Oh    RA: 2090333
# ---------------------------------------------------------------------------- #

import sys
import numpy as np
import cv2

CHANNELS = 3
INPUT_IMAGE = 'GT2.BMP'
# INPUT_IMAGE = 'Wind Waker GC.bmp'
BIN_THRESHOLD = 0.5
TOTAL_GAUSSIAN = 6
BOX_PER_GAUSIAN = 3
ALPHA = 0.6
BETA = 0.4

def main ():
   img = cv2.imread (INPUT_IMAGE, cv2.IMREAD_COLOR)
   if img is None:
      print ('Erro abrindo a imagem.\n')
      sys.exit ()
   img = img.astype (np.float32) / 255

   max_height = img.shape[0];
   max_width = img.shape[1];

   # converte para escala de cinza e cria a máscara
   mask = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
   mask = mask[..., np.newaxis]
   mask = np.where(mask > BIN_THRESHOLD, img, float(0))
   cv2.imwrite("mask.bmp", mask * 255)

   out_box = np.zeros((max_height, max_width, CHANNELS))
   out_gaussian = np.zeros((max_height, max_width, CHANNELS))

   sigma = 2
   for i in range(TOTAL_GAUSSIAN):
      box_blur(mask, out_box, sigma)
      gaussian_blur(mask, out_gaussian, sigma)
      sigma *= 2
      
   # salva as máscaras utilizadas 
   cv2.imwrite ('mask_box.bmp', out_box * 255)
   cv2.imwrite ('mask_gaussian.bmp', out_gaussian* 255)

   # mostra e salva as images
   out_box = np.where(out_box < 1, out_box, float(1) )
   out_box = ALPHA * img + BETA * out_box;
   cv2.imshow ('out_box', out_box)
   cv2.imwrite ('out_box.bmp', out_box * 255)

   out_gaussian = np.where(out_gaussian < 1, out_gaussian, float(1) )
   out_gaussian = ALPHA * img + BETA * out_gaussian;
   cv2.imshow ('out_gaussian', out_gaussian)
   cv2.imwrite ('out_gaussian.bmp', out_gaussian * 255)

   cv2.waitKey ()
   cv2.destroyAllWindows ()

def box_blur(mask, out_box, sigma):
   # https://www.w3.org/TR/SVG11/filters.html#feGaussianBlur
   ksize = int(np.floor(sigma * 3 * np.sqrt(2 * np.pi )/4 + 0.5))
   ksize = ksize + 1 if ksize % 2 == 0 else ksize
   blur = mask.copy()
   for j in range(BOX_PER_GAUSIAN):
      blur = cv2.blur(blur, (ksize, ksize));
   # salva as imagens intermediárias
   cv2.imwrite (f'sigma{sigma}_box.png', blur*255)
   out_box += blur

def gaussian_blur(mask, out_gaussian, sigma):
   # slides da disciplina (cobre 99.73% da área da curva Gaussiana)
   ksize = (6 * sigma) + 1
   blur = cv2.GaussianBlur(mask, (ksize, ksize), sigma)
   # salva as imagens intermediárias
   cv2.imwrite (f'sigma{sigma}_gaussian.png', blur*255)
   out_gaussian += blur;

if __name__ == '__main__':
   main ()