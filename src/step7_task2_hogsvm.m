function state = step7_task2_hogsvm(state, cfg)
% STEP7_TASK2_HOGSVM - Train a non-CNN classifier (HOG + linear SVM ECOC)
% Uses the 75/25 split prepared in step7_dataset.
%
% Outputs saved under cfg.paths.figures / cfg.paths.results.

    assert(isfield(state,'step7') && isfield(state.step7,'imdsTrain'), ...
        'Run step7_dataset first.');

    imdsTrain = state.step7.imdsTrain;
    imdsVal   = state.step7.imdsVal;
    inputSize = state.step7.inputSize;     % e.g., [32 32 1]
    cellSize  = [4 4];                     % HOG cell size

    % -------- Feature extraction --------
    fprintf('[HOG+SVM] Extracting HOG features ...\n');
    [Xtr, Ytr] = hog_from_imds(imdsTrain, inputSize, cellSize);
    [Xva, Yva] = hog_from_imds(imdsVal,   inputSize, cellSize);

    % -------- Train linear SVM (ECOC) --------
    t0 = tic;
    t = templateSVM('KernelFunction','linear','Standardize',true);
    model = fitcecoc(Xtr, Ytr, 'Learners', t, 'Coding','onevsone', ...
                     'ClassNames', categories(Ytr));
    trainTime = toc(t0);

    % -------- Predict on validation --------
    [Yp, ~] = predict(model, Xva);

    % ---- Normalize label types & category sets (fix confusionchart error) ----
    trainCats = categories(Ytr);  % use training-set categories as the canonical set

    if isa(Yp,'categorical')
        Yp = setcats(Yp, trainCats);      % align category set/order
    else
        Yp = categorical(Yp, trainCats);  % convert to categorical with fixed set
    end

    if isa(Yva,'categorical')
        Yva = setcats(Yva, trainCats);    % align category set/order
    else
        Yva = categorical(Yva, trainCats);
    end

    % Accuracy
    valAcc = mean(Yp == Yva);

    % -------- Confusion chart --------
    if isfield(cfg,'paths') && isfield(cfg.paths,'figures')
        f = figure('Name','Step7 - HOG+SVM Confusion','Color','w');
        confusionchart(Yva, Yp, 'RowSummary','row-normalized', 'ColumnSummary','column-normalized');
        try
            exportgraphics(f, fullfile(cfg.paths.figures,'step7_hogsvm_confusion.png'), 'Resolution',150);
        catch
            set(f,'PaperPositionMode','auto');
            print(f, fullfile(cfg.paths.figures,'step7_hogsvm_confusion.png'), '-dpng','-r150');
        end
        close(f);
    end

    % -------- Save to state --------
    state.step7.hogsvm = struct( ...
        'model',     model, ...
        'valAcc',    valAcc, ...
        'trainTime', trainTime, ...
        'cellSize',  cellSize);

    fprintf('[HOG+SVM] Val Acc = %.3f, Train Time = %.2fs\n', valAcc, trainTime);
end

% ================= helpers =================
function [X, Y] = hog_from_imds(imds, inputSize, cellSize)
% Extract HOG features from an imageDatastore into a matrix X and labels Y.
    reset(imds);
    n = numel(imds.Files);
    Y = imds.Labels;                              % categorical
    featLen = hog_len(inputSize(1:2), cellSize);
    X = zeros(n, featLen, 'single');

    for i = 1:n
        I = readimage(imds, i);
        if size(I,3) > 1, I = rgb2gray(I); end
        I = im2single(imresize(I, inputSize(1:2)));
        X(i,:) = single(extractHOGFeatures(I, 'CellSize', cellSize));
    end
end

function L = hog_len(sz, cellSize)
% Probe HOG feature vector length for given size/cell.
    I = zeros(sz, 'single');
    L = numel(extractHOGFeatures(I, 'CellSize', cellSize));
end
