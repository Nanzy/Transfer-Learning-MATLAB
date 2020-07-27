lettera = imread('letterA2.jpg');
inputSize = net.Layers(1).InputSize;
lettera1 = augmentedImageDatastore(inputSize(1:2),lettera);
tic
[etichetta, probs]= classify(netTransfer, lettera1);  
imshow(lettera);                    
title(char(etichetta));    
toc

