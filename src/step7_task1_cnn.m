function state = step7_task1_cnn(state, cfg)
% STEP7_TASK1_CNN (shim) — 使用你自己的 cnn.m 训练脚本与模型文件
% - 若 cfg.flags.trainCNN==true：调用 cnn.m 训练，随后加载 CNN_latest.mat
% - 若 cfg.flags.trainCNN==false：直接从 ../results/models/CNN_latest.mat 加载
% - 把 net / classes / inputSize 写入 state.step7.cnn，供 apply 阶段使用

    % 路径
    resultsDir = fullfile('..','results');
    modelDir   = fullfile(resultsDir,'models');
    if ~exist(modelDir,'dir'), mkdir(modelDir); end
    modelFile  = fullfile(modelDir,'CNN_useful_1.mat');

    % 需要训练时，直接跑你的 cnn.m（脚本）
    if isfield(cfg,'flags') && isfield(cfg.flags,'trainCNN') && cfg.flags.trainCNN
        fprintf('[Step7/CNN] Running your cnn.m to train...\n');
        run(fullfile(pwd,'cnn.m'));   % 你的脚本会把模型存到 modelFile
    else
        fprintf('[Step7/CNN] Skipping training; will load existing model.\n');
    end

    % 加载模型
    assert(exist(modelFile,'file')==2, ...
        'Model not found: %s. 请先运行 cnn.m 训练一次。', modelFile);
    S = load(modelFile);  % 期望含有: net, classes, inputSizeSave, useCLAHE, valAcc, trainTime(可选)

    % 映射到 pipeline 需要的字段
    inputSize = S.inputSizeSave;         % e.g. [128 128 1]
    classes   = S.classes;
    net       = S.net;
    valAcc    = getfield_or(S,'valAcc',NaN);
    trainTime = getfield_or(S,'trainTime',NaN);

    % 写入 state（与 apply 端对齐的命名）
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
