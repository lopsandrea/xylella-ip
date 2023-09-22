

%%
[X,C] = smote(dataTrain, [], 'Class',YTrain );
% Addestra il classificatore SVM
svmClassifier1 = fitcsvm(X, C, 'KernelFunction', 'polynomial');

% Effettua previsioni sul set di test
[predictedLabels, predictScores] = predict(svmClassifier1, dataTest);

% Calcola e visualizza la curva ROC
rocObj = rocmetrics(YTest, predictScores, [0,1]);
plot(rocObj)

confusionchart(Y(idx), predictedLabels)
% Calcola l'accuratezza del classificatore
accuracy = sum(predictedLabels == YTest) / sum(idx);

fprintf('Accuratezza del classificatore SVM: %.2f%%\n', accuracy * 100);



%%
k = 5; % Ad esempio, utilizza una convalida incrociata a 5 fold
cv = cvpartition(size(trainingData, 1), 'KFold', k);

accuracies = zeros(k, 1); % Memorizza le accuratezze di ciascun fold

for fold = 1:k
    % Estrai gli indici del fold di addestramento e del fold di test
    trainIdx = training(cv, fold);
    testIdx = test(cv, fold);
    
    % Estrai i dati di addestramento e test per questo fold
    dataTrain = trainingData(trainIdx, :);
    dataTest = trainingData(testIdx, :);
    
    % Addestra il classificatore SVM su questo fold
    [X, C] = smote(dataTrain, [], 'Class', labels(trainIdx));
    svmClassifier2 = fitcsvm(X, C, 'KernelFunction', 'polynomial');
    
    % Effettua previsioni sul fold di test
    predictedLabels = predict(svmClassifier2, dataTest);
    
    % Valuta l'accuratezza di questo fold e memorizzala
    accuracy = sum(predictedLabels == labels(testIdx)) / sum(testIdx);
    accuracies(fold) = accuracy;
end

meanAccuracy = mean(accuracies);
fprintf('Accuratezza media convalida incrociata k-fold: %.2f%%\n', meanAccuracy * 100);

%%
% Esegui PCA sui dati di addestramento
[X,C] = smote(dataTrain, [], 'Class',labels(~idx) );
[coeff,scoreTrain,~,~,explained,mu] = pca(X);
numComponentsToKeep = find(cumsum(explained)>95,1);
% Riduci la dimensionalità dei dati di addestramento e test
dataTrainReduced = scoreTrain(:, 1:numComponentsToKeep);
dataTestReduced = (dataTest - mu) * coeff(:, 1:numComponentsToKeep);
% Addestra il classificatore SVM sui dati ridotti
svmClassifier = fitcsvm(dataTrainReduced, C, 'KernelFunction', 'polynomial');
% Effettua previsioni sul set di test ridotto
[predictedLabels, predictScores] = predict(svmClassifier, dataTestReduced);
% Calcola l'accuratezza del classificatore
accuracy = sum(predictedLabels == labels(idx)) / sum(idx);
fprintf('Accuratezza del classificatore SVM con PCA: %.2f%%\n', accuracy * 100);

% Calcola la matrice di confusione
confusionMat = confusionmat(labels(idx), predictedLabels);
disp('Matrice di confusione:');
disp(confusionMat);
