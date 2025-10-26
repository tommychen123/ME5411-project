function state = step7_task1_cnn(state, cfg)
% STEP7_TASK1_CNN (shim) — Bridge your cnn.m script and the pipeline
% - If cfg.flags.trainCNN == true: run cnn.m to train, then load CNN_useful_1.mat
% - If cfg.flags.trainCNN == false: directly load ../results/models/CNN_useful_1.mat
% - Write net / classes / inputSize into state.step7.cnn for the apply stage

    % Paths
    resultsDir = fullfile('..','results');
    modelDir   = fullfile(resultsDir,'models');
    if ~exist(modelDir,'dir'), mkdir(modelDir); end
    modelFile  = fullfile(modelDir,'CNN_useful_1.mat');

    % Train if required (your cnn.m will save to modelFile)
    if isfield(cfg,'flags') && isfield(cfg.flags,'trainCNN') && cfg.flags.trainCNN
        fprintf('[Step7/CNN] Running your cnn.m to train...\n');
        run(fullfile(pwd,'cnn.m'));   % your script should save to modelFile
    else
        fprintf('[Step7/CNN] Skipping training; will load existing model.\n');
    end

    % Load model
    assert(exist(modelFile,'file')==2, ...
        'Model not found: %s. Please run cnn.m once to train.', modelFile);
    S = load(modelFile);  % expected fields: net, classes, inputSizeSave, useCLAHE, valAcc, trainTime (optional)

    % Map to pipeline fields
    inputSize = S.inputSizeSave;         % e.g. [128 128 1]
    classes   = S.classes;
    net       = S.net;
    valAcc    = getfield_or(S,'valAcc',NaN);
    trainTime = getfield_or(S,'trainTime',NaN);

    % Write into state (aligned with apply-side naming)
    state.step7.cnn = struct( ...
        'net',       net, ...
        'classes',   {classes}, ...
        'inputSize', inputSize, ...
        'valAcc',    valAcc, ...
        'trainTime', trainTime);

    fprintf('[Step7/CNN] Loaded model. ValAcc=%.2f%% | inputSize=[%d %d %d]\n', ...
        100*valAcc, inputSize(1), inputSize(2), numel(inputSize)>=3*inputSize(3));
end

function v = getfield_or(S, f, dv)
    if isfield(S,f), v = S.(f); else, v = dv; end
end
