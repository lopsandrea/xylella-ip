nK = 10;
c = cvpartition(t,'KFold',nK);

for k = 1 : nK
    x_train = X(c.training(k),:);
    x_test = X(c.test(k),:);
    t_train = T(c.training(k),:);
    t_test = T(c.test(k),:);

    %% LDA
    do_lda = true;
    if do_lda
        ldaMdl = fitcdiscr(x_train,t_train, 'DiscrimType','pseudoLinear');
        weigths = abs(ldaMdl.Coeffs(2,1).Linear);
        [weigths_sorted,indexes] = sort(weigths, 'descend');

        feature_out = 30;
        weights_selected = weigths_sorted(1:feature_out);
        indexes_selected = indexes(1:feature_out);

        threshold_weight = 8e-1;
        weights_thresholded = weigths(weigths > threshold_weight);

        feature_mask = weigths > threshold_weight;

        x_train_reduced_lda = x_train(:, feature_mask);
        x_test_reduced_lda = x_test(:, feature_mask);

    end

    %% NCA
    do_nca = true;
    if do_nca
        ncaMdl = fscnca(x_train,t_train);
        weights = ncaMdl.FeatureWeights;
        feature_out = 30;
        [weigths_sorted,indexes] = sort(weights, 'descend');
        weights_selected = weigths_sorted(1:feature_out);
        indexes_selected = indexes(1:feature_out);
        x_train_reduced_nca = x_train(:, indexes_selected);
        x_test_reduced_nca = x_test(:, indexes_selected);

    end

    %% PCA
    var_p = 95;
    x_mean = mean(x_train);
    x_centered = x_train - x_mean;
    warning('off')
    [coeff,score,latent,tsquared,explained,mu] = pca(x_centered,'Centered',false);
    warning('on')
    x_train_PCA = x_centered*coeff;
    n_comp = numel(explained);
    for ii = 1:n_comp
        % Calcolo la somma dei primi "ii" elementi di explained
        sum_ = sum(explained(1:ii));
        if sum_ > var_p, break; end
    end
    num_pc = ii;

    fprintf("For preserving %.2f%s of variance, you have to use %d PC\n", var_p, "%", num_pc);
    x_train_PCA_r = x_train_PCA(:,1:num_pc);

    figure
        bar(explained/100)
        xlim([0 num_pc])
        title(sprintf("Variance to explain = %.2f",var_p/100))
        xlabel(sprintf("First %d Principal Component",num_pc))
        ylabel("Variance explained")

    x_test_centered = x_test - x_mean;
    x_test_PCA = x_test_centered*coeff;
    x_test_PCA_r = x_test_PCA(:,1:num_pc);

    x_tr{1}=x_train';
    x_tr{2}=x_train_PCA_r';
    x_tr{3}=x_train_reduced_lda';
    x_tr{4}=x_train_reduced_nca';

    x_te{1}=x_test';
    x_te{2}=x_test_PCA_r';
    x_te{3}=x_test_reduced_lda';
    x_te{4}=x_test_reduced_nca';
    nWorkflow = 4;

    t_train = t_train';
    t_test = t_test';
    for i = 1 : nWorkflow
        %% Dataset
        x_train = x_tr{i};
        x_test = x_te{i};
        %% Configure MLP
        hiddenLayerSize = [10 10];
        trainFcn = 'traingdx';
        costFunction = 'crossentropy';
        epochs = 500;
        lastLayerActivationFunction = 'logsig';
        net = patternnet(hiddenLayerSize, trainFcn);
        net.trainParam.epochs = epochs;
        net.performFcn = costFunction;
        net.layers{end}.transferFcn = lastLayerActivationFunction;
        net.divideFcn = 'dividerand';
        net.divideParam.trainRatio = 100/100;
        net.divideParam.valRatio   = 0/100;
        net.divideParam.testRatio  = 0/100;
        net = configure(net,x_train,t_train);
        net = init(net);
        [net,tr] = train(net,x_train,t_train);
        y_train = net(x_train);
        y_test = net(x_test);

        %% Metrics Test
        figure, plotconfusion(t_test,y_test),title('Test Confusion Matrix NN') 
        figure, plotroc(t_test,y_test)

        %% AUC
        [X,Y,T,AUC,optroc] = perfcurve(t_test,y_test,1);
        AUCC{i,k} = AUC;
        accuracyNN(i,k) = sum(t_test == (y_test>T(optroc(1)==X&optroc(2)==Y)),'all')/numel(y_test);
        %% Other classifiers
        pred = x_train';
        resp = t_train';

        %% Linear regression
        LCModel = fitclinear(pred,resp,'Learner','logistic');
        [y_test,score] = LCModel.predict(x_test');
        figure
            plotconfusion(t_test,y_test')
            title('Test Confusion Matrix Linear Classification')
        figure
            plotroc(t_test,score(:,2)')
            title("ROC Linear Classification")
        [X,Y,T,AUC] = perfcurve(t_test,score(:,2)',1);

        AUCLM{i,k} = AUC;
        accuracyLM(i,k) = sum(t_test == y_test','all')/numel(y_test);
        %% SVM
        SVMModel = fitcsvm(pred,resp,'KernelFunction','linear',...
            'Standardize',true,'ClassNames',[0,1]);

        [y_test,score] = predict(SVMModel,x_test');
        figure
            plotconfusion(t_test,y_test')
            title('Test Confusion Matrix SVM')
        figure
            plotroc(t_test,score(:,2)')
            title("ROC SVM")
        [X,Y,T,AUC] = perfcurve(t_test,score(:,2)',1);

        AUCSVM{i,k} = AUC;
        accuracySVM(i,k) = sum(t_test == y_test','all')/numel(y_test);
    end
end
figure
    boxplot([accuracyNN(1,:)' accuracyLM(1,:)' accuracySVM(1,:)'],["ANN" "LR" "SVM"])
    title("Workflow Unreduced")
    ylabel("Accuracy")
figure
    boxplot([accuracyNN(2,:)' accuracyLM(2,:)' accuracySVM(2,:)'],["ANN" "LR" "SVM"])
    title("Workflow PCA")
    ylabel("Accuracy")
figure
    boxplot([accuracyNN(3,:)' accuracyLM(3,:)' accuracySVM(3,:)'],["ANN" "LR" "SVM"])
    title("Workflow LDA")
    ylabel("Accuracy")
figure
    boxplot([accuracyNN(4,:)' accuracyLM(4,:)' accuracySVM(4,:)'],["ANN" "LR" "SVM"])
    title("Workflow NCA")
    ylabel("Accuracy")
