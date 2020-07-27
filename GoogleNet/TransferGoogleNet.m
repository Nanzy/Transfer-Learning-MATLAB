net = googlenet;

lgraph = layerGraph(net);
inputSize = net.Layers(1).InputSize;

lgraph = removeLayers(lgraph, {'loss3-classifier','prob','output'});

numClasses = numel(categories(imdsTrain.Labels))
newLayers = [
    fullyConnectedLayer(numClasses,'Name','fc','WeightLearnRateFactor',20,'BiasLearnRateFactor',20)
    softmaxLayer('Name','softmax')
    classificationLayer('Name','classoutput')];
lgraph = addLayers(lgraph,newLayers);

lgraph = connectLayers(lgraph,'pool5-drop_7x7_s1','fc');

layers = lgraph.Layers;
connections = lgraph.Connections;
layers(1:110) = freezeWeights(layers(1:110));
lgraph = createLgraphUsingConnections(layers,connections);

pixelRange = [-30 30];
imageAugmenter = imageDataAugmenter( ...
    'RandXReflection',true, ...
    'RandXTranslation',pixelRange, ...
    'RandYTranslation',pixelRange);
augimdsTrain = augmentedImageDatastore(inputSize(1:2),imdsTrain, ...
    'DataAugmentation',imageAugmenter);

augimdsValidation = augmentedImageDatastore(inputSize(1:2),imdsValidation);
augimdsPrediction = augmentedImageDatastore(inputSize(1:2),imdsPrediction);

options = trainingOptions('sgdm',...
    'MiniBatchSize',64,...
    'MaxEpochs',14,...
    'InitialLearnRate',1e-3,...
    'Shuffle','every-epoch',...
    'Plots','training-progress',...
    'ValidationData',augimdsValidation,...
    'ValidationFrequency',32, ...
    'VerboseFrequency', 32,...
    'ValidationPatience',Inf, ...
    'ExecutionEnvironment','gpu');

[net, infoTraining] = trainNetwork(augimdsTrain,lgraph,options);

[YPred,probs] = classify(net,augimdsPrediction);
YValidation = imdsPrediction.Labels;
accuracy = mean(YPred == YValidation)