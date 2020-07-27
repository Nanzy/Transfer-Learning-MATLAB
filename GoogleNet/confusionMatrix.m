numCorrect = nnz( YValidation == YPred)
fracCorrect = numCorrect/numel(YPred)
[letconf,letnames] = confusionmat(imdsPrediction.Labels,YPred)
heatmap(letnames,letnames,letconf);