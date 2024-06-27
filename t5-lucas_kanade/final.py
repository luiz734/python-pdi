optical flow
I(x,y,t) -> I(x+dx,y+dy,t+dt) DERIVADAS!
Optic flow equation: Achar vetor (u,v)
Não é possível resolver com apenas 1 pixel

Horn–Schunck Method
tenta encontrar (u,v) olhando vizinhos
Assume que o (u,v) de 1 será próximo da média de (u,v)s dos vizinhos
Abordagem global (valor propagado = smoothness)

Lucas–Kanade Method 
Assume que para um janela pequena, (u,v) será o mesmo
Para uma janela 5x5, 25 equações (mais que o suficiente)
Abordagem local (se preocupa com o próprio quadrado)

Aperture problem
Movimento ambíguo (falta informações)
Não é possível determinar (u,v)
Bordas não são o suficiente (Importância dos cantos)



