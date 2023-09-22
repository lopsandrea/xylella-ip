% Define the number of folds for cross-validation
nK = 2;

% Create a cross-validation partition
c = cvpartition(t, 'KFold', nK);

% Loop through each fold
for k = 1:nK
    % Split the dataset into training and testing sets for this fold
    x_train = originalX(c.training(k), :);
    x_test = originalX(c.test(k), :);
    t_train = originalT(c.training(k), :);
    t_test = originalT(c.test(k), :);

    %% Linear Discriminant Analysis (LDA)
    do_lda = true;
    if do_lda
        ldaMdl = fitcdiscr(x_train, t_train, 'DiscrimType', 'pseudoLinear');
        weights = abs(ldaMdl.Coeffs(2, 1).Linear);
        [weights_sorted, indexes] = sort(weights, 'descend');

        feature_out = 30;
        weights_selected = weights_sorted(1:feature_out);
        indexes_selected = indexes(1:feature_out);

        threshold_weight = 8e-1;
        feature_mask = weights > threshold_weight;
        x_train_reduced_lda = x_train(:, feature_mask);
        x_test_reduced_lda = x_test(:, feature_mask);
    end

    %% Neighbourhood Component Analysis (NCA)
    do_nca = true;
    if do_nca
        ncaMdl = fscnca(x_train, t_train);
        weights = ncaMdl.FeatureWeights;

        feature_out = 30;
        [weights_sorted, indexes] = sort(weights, 'descend');
        weights_selected = weights_sorted(1:feature_out);
        indexes_selected = indexes(1:feature_out);

        x_train_reduced_nca = x_train(:, indexes_selected);
        x_test_reduced_nca = x_test(:, indexes_selected);
    end

    %% Principal Component Analysis (PCA)
    var_p = 95;
    x_mean = mean(x_train);
    x_centered = x_train - x_mean;

    % Calculate PCA
    warning('off')
    [coeff, score, latent, tsquared, explained, mu] = pca(x_centered, 'Centered', false);
    warning('on')
    x_train_PCA = x_centered * coeff;
    
    % Determine the number of principal components to preserve variance
    n_comp = numel(explained);
    for ii = 1:n_comp
        sum_ = sum(explained(1:ii));
        if sum_ > var_p, break; end
    end
    num_pc = ii;

    % Reduce the dimensions of the training and testing data using PCA
    x_train_PCA_r = x_train_PCA(:, 1:num_pc);
    x_test_centered = x_test - x_mean;
    x_test_PCA = x_test_centered * coeff;
    x_test_PCA_r = x_test_PCA(:, 1:num_pc);

    % Prepare data for different workflows
    x_tr{1} = x_train';
    x_tr{2} = x_train_PCA_r';
    x_tr{3} = x_train_reduced_lda';
    x_tr{4} = x_train_reduced_nca';

    x_te{1} = x_test';
    x_te{2} = x_test_PCA_r';
    x_te{3} = x_test_reduced_lda';
    x_te{4} = x_test_reduced_nca';
    nWorkflow = 4;

    t_train = t_train';
    t_test = t_test';

    % Loop through different workflows
    for i = 1:nWorkflow
        % Dataset for the current workflow
        x_train = x_tr{i};
        x_test = x_te{i};

        %% Configure Multi-Layer Perceptron (MLP)
        hiddenLayerSize = [10 10];
        trainFcn = 'traingdx';
        costFunction = 'crossentropy';
        epochs = 500;
        lastLayerActivationFunction = 'logsig';

        % Create and configure the neural network
        net = patternnet(hiddenLayerSize, trainFcn);
        net.trainParam.epochs = epochs;
        net.performFcn = costFunction;
        net.layers{end}.transferFcn = lastLayerActivationFunction;
        net.divideFcn = 'dividerand';
        net.divideParam.trainRatio = 100/100;
        net.divideParam.valRatio = 0/100;
        net.divideParam.testRatio = 0/100;
        net = configure(net, x_train, t_train);
        net = init(net);
        [net, tr] = train(net, x_train);
        y_train = net(x_train);
        y_test = net(x_test);

        %% Metrics for Neural Network
        figure, plotconfusion(t_test, y_test), title('Test Confusion Matrix NN');
        figure, plotroc(t_test, y_test);

        % Calculate AUC for ROC
        [X, Y, T, AUC, optroc] = perfcurve(t_test, y_test, 1);
        AUCC{i, k} = AUC;
        accuracyNN(i, k) = sum(t_test == (y_test > T(optroc(1) == X & optroc(2) == Y)), 'all') / numel(y_test);

        %% Other Classifiers
        pred = x_train';
        resp = t_train';

        %% Linear Classification (LC)
        LCModel = fitclinear(pred, resp, 'Learner', 'logistic');
        [y_test, score] = LCModel.predict(x_test');
        figure, plotconfusion(t_test, y_test'), title('Test Confusion Matrix Linear Classification');
        figure, plotroc(t_test, score(:, 2)'), title("ROC Linear Classification");
        [X, Y, T, AUC] = perfcurve(t_test, score(:, 2)', 1);
        AUCLM{i, k} = AUC;
        accuracyLM(i, k) = sum(t_test == y_test', 'all') / numel(y_test);

        %% Support Vector Machine (SVM)
        SVMModel = fitcsvm(pred, resp, 'KernelFunction', 'linear', 'Standardize', true, 'ClassNames', [0, 1]);
        [y_test, score] = predict(SVMModel, x_test');
        figure, plotconfusion(t_test, y_test'), title('Test Confusion Matrix SVM');
        figure, plotroc(t_test, score(:, 2)'), title("ROC SVM");
        [X, Y, T, AUC] = perfcurve(t_test, score(:, 2)', 1);
        AUCSVM{i, k} = AUC;
        accuracySVM(i, k) = sum(t_test == y_test', 'all') / numel(y_test);
    end
end

% Plot the accuracy results for different workflows
figure
boxplot([accuracyNN(1, :)' accuracyLM(1, :)' accuracySVM(1, :)'], ["ANN" "LR" "SVM"]);
title("Workflow Unreduced");
ylabel("Accuracy");

figure
boxplot([accuracyNN(2, :)' accuracyLM(2, :)' accuracySVM(2, :)'], ["ANN" "LR" "SVM"]);
title("Workflow PCA");
ylabel("Accuracy");

figure
boxplot([accuracyNN(3, :)' accuracyLM(3, :)' accuracySVM(3, :)'], ["ANN" "LR" "SVM"]);
title("Workflow LDA");
ylabel("Accuracy");

figure
boxplot([accuracyNN(4, :)' accuracyLM(4, :)' accuracySVM(4, :)'], ["ANN" "LR" "SVM"]);
title("Workflow NCA");
ylabel("Accuracy");
