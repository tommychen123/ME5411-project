%% cnn.m - 训练一个稳泛化的小型CNN并保存模型
% 数据需按文件夹为类的结构组织：
%   ../data/dataset_2025_aug/
%       classA/*.png
%       classB/*.png
% 输出: ../results/models/CNN_latest.mat (包含 net, classes, inputSizeSave, useCLAHE, valAcc)

clc; clear; close all; rng(42);  % 固定随机种子，便于复现

%% ====== 路径配置 ======
datasetRoot = fullfile('..','data','dataset_2025_aug');   % TODO: 换成你的数据目录
assert(exist(datasetRoot,'dir')==7, '数据集不存在: %s', datasetRoot);

resultsDir = fullfile('..','results');   if ~exist(resultsDir,'dir'), mkdir(resultsDir); end
modelDir   = fullfile(resultsDir,'models'); if ~exist(modelDir,'dir'), mkdir(modelDir); end
figDir     = fullfile(resultsDir,'figures'); if ~exist(figDir,'dir'), mkdir(figDir); end
modelFile  = fullfile(modelDir,'CNN_latest.mat');

%% ====== 超参（小数据稳健配置） ======
inputSize = [160 160 1];   % 灰度；若用预训练改成 [224 224 3]
useCLAHE  = true;

maxEpochs = 24;
miniBatch = 256;
baseLR    = 1e-3;
l2reg     = 2e-4;
valSplit  = 0.85;
valPatience = 6;

% 可选：类别均衡（下采样到最小类）
balanceMode = 'none';   % 'none' 或 'undersample'

%% ====== 读入数据 + 统一 ReadFcn（训练/推理一致）======
rf = @(fn) reader_preproc(fn, inputSize(1:2), useCLAHE, inputSize(3));
imdsAll = imageDatastore(datasetRoot, 'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames', 'ReadFcn', rf);

if strcmpi(balanceMode,'undersample')
    tbl = countEachLabel(imdsAll);
    minCount = min(tbl.Count);
    imdsAll = splitEachLabel(imdsAll, minCount, 'randomize');
end

[imdsTrain, imdsVal] = splitEachLabel(imdsAll, valSplit, 'randomized');
classes = categories(imdsTrain.Labels);
numClasses = numel(classes);
fprintf('[train] %d train / %d val / %d classes\n', numel(imdsTrain.Files), numel(imdsVal.Files), numClasses);

%% ====== 在线增广（安全：无翻转、无剪切、轻量幅度） ======
augmenter = imageDataAugmenter( ...
    'RandRotation',[0 0], ...
    'RandXTranslation',[-2 2], 'RandYTranslation',[-2 2], ...
    'RandXScale',[0.99 1.02], 'RandYScale',[0.99 1.02], ...
    'RandXReflection',false, 'RandYReflection',false);
augTrain = augmentedImageDatastore(inputSize(1:2), imdsTrain, 'DataAugmentation', augmenter);
augVal   = augmentedImageDatastore(inputSize(1:2), imdsVal);

%% ====== 类别权重（防类别不均衡） ======
cnt     = countcats(imdsTrain.Labels);
invFreq = 1 ./ max(1,double(cnt));
clsWts  = invFreq / sum(invFreq) * numClasses;

%% ====== 模型（Conv-Conv-Pool ×3 + GAP + FC） ======
layers = [
    imageInputLayer(inputSize,'Normalization','none','Name','in')

    convolution2dLayer(3,32,'Padding','same','WeightsInitializer','he','Name','c1_1')
    batchNormalizationLayer('Name','bn1_1'); reluLayer('Name','r1_1')
    convolution2dLayer(3,32,'Padding','same','WeightsInitializer','he','Name','c1_2')
    batchNormalizationLayer('Name','bn1_2'); reluLayer('Name','r1_2')
    maxPooling2dLayer(2,'Stride',2,'Name','p1')

    convolution2dLayer(3,64,'Padding','same','WeightsInitializer','he','Name','c2_1')
    batchNormalizationLayer('Name','bn2_1'); reluLayer('Name','r2_1')
    convolution2dLayer(3,64,'Padding','same','WeightsInitializer','he','Name','c2_2')
    batchNormalizationLayer('Name','bn2_2'); reluLayer('Name','r2_2')
    maxPooling2dLayer(2,'Stride',2,'Name','p2')

    convolution2dLayer(3,128,'Padding','same','WeightsInitializer','he','Name','c3_1')
    batchNormalizationLayer('Name','bn3_1'); reluLayer('Name','r3_1')
    convolution2dLayer(3,128,'Padding','same','WeightsInitializer','he','Name','c3_2')
    batchNormalizationLayer('Name','bn3_2'); reluLayer('Name','r3_2')
    dropoutLayer(0.40,'Name','drop')   % 稍强正则
    maxPooling2dLayer(2,'Stride',2,'Name','p3')

    globalAveragePooling2dLayer('Name','gap')
    fullyConnectedLayer(numClasses,'Name','fc')
    softmaxLayer('Name','sm')
    classificationLayer('Name','out','Classes',classes,'ClassWeights',clsWts)
];

%% ====== 训练选项 ======
execEnv = 'gpu'; try gpuDevice(); catch, execEnv = 'auto'; end
itersPerEpoch = max(1, floor(numel(imdsTrain.Files)/miniBatch));
opts = trainingOptions('adam', ...
    'ExecutionEnvironment', execEnv, ...
    'InitialLearnRate', baseLR, ...
    'LearnRateSchedule','piecewise','LearnRateDropFactor',0.3,'LearnRateDropPeriod',6, ...
    'L2Regularization', l2reg, ...
    'MaxEpochs', maxEpochs, 'MiniBatchSize', miniBatch, ...
    'Shuffle','every-epoch', ...
    'ValidationData', augVal, ...
    'ValidationFrequency', max(1,itersPerEpoch), ...
    'ValidationPatience', valPatience, ...
    'Verbose', false, 'Plots','none');

%% ====== 训练 ======
fprintf('[train] Training ...\n');
t0 = tic; 
net = trainNetwork(augTrain, layers, opts);
trainTime = toc(t0);

%% ====== 验证 ======
[YP,~] = classify(net, augVal); YV = imdsVal.Labels;
valAcc = mean(YP==YV);
fprintf('[train] Val acc: %.2f%%\n', 100*valAcc);

try
    f=figure('Color','w');
    confusionchart(YV, YP, 'RowSummary','row-normalized','ColumnSummary','column-normalized');
    exportgraphics(f, fullfile(figDir,'cnn_confusion.png'),'Resolution',150);
    close(f);
catch
end

%% ====== 保存模型 ======
inputSizeSave = inputSize;
save(modelFile, 'net','classes','inputSizeSave','useCLAHE','valAcc','trainTime');
fprintf('[train] Model saved: %s\n', modelFile);

%% ================= Helpers（训练/推理一致） =================
function Iout = reader_preproc(filename, outSz, useCLAHE, outC)
    I = imread(filename);
    if size(I,3)>1, I = rgb2gray(I); end
    I = im2single(I);
    if nargin<3, useCLAHE = true; end
    if nargin<4, outC = 1; end
    I = letterbox_white(I, outSz, useCLAHE);
    if outC==3, Iout = repmat(I,[1 1 3]); else, Iout = I; end
end

function Iout = letterbox_white(I, outSz, useCLAHE)
    if useCLAHE
        I = adapthisteq(I,'NumTiles',[8 8],'ClipLimit',0.02);
    end
    S = outSz(1); T = outSz(2);
    s  = min(S/size(I,1), T/size(I,2));
    nh = max(1, min(S, round(size(I,1)*s)));
    nw = max(1, min(T, round(size(I,2)*s)));
    Ir = imresize(I, [nh nw], 'nearest');
    y0 = max(1, floor((S - nh)/2) + 1);
    x0 = max(1, floor((T - nw)/2) + 1);
    y1 = min(S, y0 + nh - 1);
    x1 = min(T, x0 + nw - 1);
    Ir = Ir(1:(y1-y0+1), 1:(x1-x0+1));
    Iout = ones(S,T,'single');
    Iout(y0:y1, x0:x1) = Ir;
end
