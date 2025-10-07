function state = step7_task1_cnn(state, cfg)
% Train a compact CNN and evaluate on validation set.

    assert(isfield(state,'step7') && isfield(state.step7,'imdsTrain'), ...
        'Run step7_dataset first.');

    imdsTrain = state.step7.imdsTrain;
    imdsVal   = state.step7.imdsVal;
    classes   = state.step7.classes;
    inputSize = state.step7.inputSize;  % e.g., [32 32 1]

    % datastores with resize + grayscale
    augTrain = augmentedImageDatastore(inputSize(1:2), imdsTrain, ...
        'ColorPreprocessing','rgb2gray');
    augVal   = augmentedImageDatastore(inputSize(1:2), imdsVal, ...
        'ColorPreprocessing','rgb2gray');

    % LeNet-like tiny CNN
    numClasses = numel(classes);
    % LeNet-like tiny CNN (one layer per line; no commas)
numClasses = numel(classes);
layers = [
    imageInputLayer(inputSize,'Normalization','zerocenter')  % 若版本支持也可用 'zscore'
    convolution2dLayer(3,32,'Padding','same')
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2,'Stride',2)

    convolution2dLayer(3,64,'Padding','same')
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2,'Stride',2)

    convolution2dLayer(3,128,'Padding','same')
    batchNormalizationLayer
    reluLayer

    fullyConnectedLayer(256)
    reluLayer
    dropoutLayer(0.3)

    fullyConnectedLayer(numClasses)
    softmaxLayer
    classificationLayer
];


    opts = trainingOptions('adam', ...
        'MiniBatchSize', 64, ...
        'MaxEpochs', 15, ...
        'Shuffle','every-epoch', ...
        'ValidationData', augVal, ...
        'ValidationFrequency', 50, ...
        'Verbose', false);

    t0 = tic;
    net = trainNetwork(augTrain, layers, opts);
    trainTime = toc(t0);

    % Validation accuracy
    YVal = imdsVal.Labels;
    YPred = classify(net, augVal);
    valAcc = mean(YPred == YVal);

    % Confusion chart
    if isfield(cfg,'paths') && isfield(cfg.paths,'figures')
        f = figure('Name','Step7 - CNN Confusion','Color','w');
        confusionchart(YVal, YPred,'RowSummary','row-normalized','ColumnSummary','column-normalized');
        exportgraphics(f, fullfile(cfg.paths.figures,'step7_cnn_confusion.png'), 'Resolution',150);
        close(f);
    end

    % save to state
    state.step7.cnn = struct('net',net,'valAcc',valAcc,'trainTime',trainTime);
    fprintf('[Step7/CNN] Val Acc = %.3f, Train Time = %.2fs\n', valAcc, trainTime);
end
