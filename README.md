# Diagnóstico de Osteoartrite Femorotibial através de imagens de raio X

Esse trabalho foi elaborado para a __Processamento e Análise de Imagens__


## Descrição:
A osteoartrite (`artrose`) é uma doença que se caracteriza pelo desgaste da cartilagem articular e por
alterações ósseas nas articulações. O raio X é o principal exame para diagnóstico da doença que é
classificada pela escala de Kellgren & Lawrence (`KL`) , de acordo com o seu grau de severidade. O
diagnóstico de artrose é confirmado para `KL > 1`.

## Dataset:
O conjunto de imagens a ser usado está disponível em
https://data.mendeley.com/datasets/56rmx5bjcr/1


## Especificações do programa: 

O ambiente deve ser totalmente gráfico e deverá oferecer as seguintes opções acessadas por
menus:


## 1º parte

- [x] Ler e visualizar imagens nos formatos PNG e JPG. As imagens podem ter qualquer
resolução e número de tons de cinza (normalmente variando entre 8 e 16 bits por
pixel);

- [x] Recortar uma sub-região de tamanho arbitrário com o mouse e salvá-la como arquivo.

- [x] Buscar, em uma imagem qualquer, uma região previamente recortada ou lida de
arquivo. Indicar com um retângulo a posição onde foi detectada. Uma técnica que
pode ser usada para isso é a correlação cruzada:

$CCN = \frac{1}{S_A S_B}\sum_{x,y}(A(x,y)-m_A)(B(x,y)-m_B)$

$m_l=\frac{1}{N}\sum_{x,y}l(x,y)$

$S_l =\sqrt{\sum_{x,y} (l(x,y)-m_A)^2}$

onde `A` e `B` são a imagem e região buscada, com respectivos valores de intensidade
médios (`m`) e desvios-padrões (`s`). O valor máximo de `CCN` indicará a posição mais
provável de ocorrência de `B` em `A`.


## 2 parte:
- [ ] Sistema de cache 

- [ ] Ler os diretórios onde estarão as imagens usadas para treino, validação e teste dos classificadores utilizados.

- [ ] Realizar aumento de dados através de espelhamento horizontal e equalização de histogramas.

- [ ] Especificar e treinar pelo menos 2 classificadores, exibindo-se o tempo de execução na interface:

- [ ] Classificador raso, exceto rede neural, utilizando características extraídas da
imagem (projeções, textura, histogramas, descritores de forma, etc.) __SVM__ ()

- [ ] Rede neural convolucional sorteada para o grupo. Utilize os pesos
disponíveis no modelo, retreinando a parte completamente conectada com o dataset de raio X __GOOGLENET__

- [ ] Classificar o conjunto de teste em 2 opções: classificação binária (normal x artrose) e com as 5 classes KL. O tempo de execução deve ser medido e exibido na interface, juntamente com a matriz de confusão e as métricas de sensibilidade, especificidade, precisão, acurácia e escore F1.


## Extra 

- [ ]  PONTOS EXTRAS: Classificador XGBoost1 (Se o grupo tiver usado o XGBoost como classificador raso deverá implementar um outro raso.)

## Artigo de referencia:

_Pingjun Chen, Linlin Gao, Xiaoshuang Shi, Kyle Allen, Lin Yang. “Fully automatic knee
osteoarthritis severity grading using deep neural networks with a novel ordinal loss”. Computerized Medical Imaging and Graphics 75:84-92, 2019._