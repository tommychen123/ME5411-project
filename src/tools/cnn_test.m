%% cnn_test.m - 带 padScale 的参数对比实验
% 比较：输入尺寸 × CLAHE × 去噪方式/强度 × padScale
% padScale: 控制缩放后占满画布的程度 (1.0=正好贴边; <1=留白; >1=略放大裁边)
%
% 输出：
%   ../results/benchmarks/cnn_sweep_pad_summary.csv
%   ../results/models/szXX_cX_dn-XXX_psX.X/

clc; clear; close all; rng(42);

%% ====== 路径配置 ======
datasetRoot = fullfile('..','data','dataset_2025');
assert(exist(datasetRoot,'dir')==7, '数据集不存在: %s', datasetRoot);

resultsDir = fullfile('..','results');   if ~exist(resultsDir,'dir'), mkdir(resultsDir); end
modelDir   = fullfile(resultsDir,'models'); if ~exist(modelDir,'dir'), mkdir(modelDir); end
figDir     = fullfile(resultsDir,'figures'); if ~exist(figDir,'dir'), mkdir(figDir); end
benchDir   = fullfile(resultsDir,'benchmarks'); if ~exist(benchDir,'dir'), mkdir(benchDir); end

%% ====== 全局训练超参 ======
maxEpochs   = 20;
miniBatch   = 256;
baseLR      = 1e-3;
l2reg       = 2e-4;
valSplit    = 0.85;
valPatience = 6;
balanceMode = 'none';

%% ====== 搜索组合 ======
inputSizeList = { [28 28 1], [32 32 1],[48 48 1]};
claheList     = [false, true];
denoiseList = { ...
    struct('method','none','param',0), ...
    struct('method','median','param',3), ...
    struct('method','gaussian','param',0.8) ...
};
padScaleList  = [1.4, 1.7];

%% ====== 开始扫描 ======
Rows = []; rowID = 0;

for ii = 1:numel(inputSizeList)
    for ic = 1:numel(claheList)
        for idn = 1:numel(denoiseList)
            for ips = 1:numel(padScaleList)
                rowID = rowID + 1;
                inputSize = inputSizeList{ii};
                useCLAHE  = claheList(ic);
                dn        = denoiseList{idn};
                padScale  = padScaleList(ips);

                % 标签字符串
                tag = sprintf('sz%dx%d_c%d_dn-%s-%g_ps%.2f', ...
                    inputSize(1), inputSize(2), useCLAHE, dn.method, dn.param, padScale);
                tag = strrep(tag,'.','p');
                outModelDir = fullfile(modelDir, tag);
                if ~exist(outModelDir,'dir'), mkdir(outModelDir); end
                modelFile = fullfile(outModelDir,'CNN_latest.mat');

                fprintf('\n===== [%03d] %s =====\n', rowID, tag);

                [valAcc, trainTime, nTrain, nVal, nClasses] = ...
                    train_once(datasetRoot, inputSize, useCLAHE, dn, padScale, ...
                               maxEpochs, miniBatch, baseLR, l2reg, valSplit, ...
                               valPatience, balanceMode, modelFile, figDir);

                Rows = [Rows; struct( ...
                    'id', rowID, ...
                    'inputH', inputSize(1), 'inputW', inputSize(2), ...
                    'useCLAHE', logical(useCLAHE), ...
                    'denoise', string(dn.method), 'param', dn.param, ...
                    'padScale', padScale, ...
                    'valAcc', valAcc, 'trainTime_s', trainTime, ...
                    'nTrain', nTrain, 'nVal', nVal, 'nClasses', nClasses, ...
                    'tag', string(tag), 'modelFile', string(modelFile) ...
                )]; %#ok<AGROW>
            end
        end
    end
end

%% ====== 导出汇总 ======
T = struct2table(Rows);
summaryCsv = fullfile(benchDir,'cnn_sweep_pad_summary.csv');
writetable(T, summaryCsv);
save(fullfile(benchDir,'cnn_sweep_pad_summary.mat'),'T');
fprintf('\n[done] Summary saved: %s\n', summaryCsv);
disp(T(:,{'tag','valAcc','trainTime_s','padScale'}));

%% ============= 子函数 =============
function [valAcc, trainTime, nTrain, nVal, nClasses] = train_once( ...
    datasetRoot, inputSize, useCLAHE, dn, padScale, ...
    maxEpochs, miniBatch, baseLR, l2reg, valSplit, valPatience, ...
    balanceMode, modelFile, figDir)

    preproc.useCLAHE = useCLAHE;
    preproc.denoise  = dn;
    preproc.padScale = padScale;
    rf = @(fn) reader_preproc_param(fn, inputSize(1:2), preproc, inputSize(3));

    imdsAll = imageDatastore(datasetRoot, 'IncludeSubfolders', true, ...
        'LabelSource', 'foldernames', 'ReadFcn', rf);
    if strcmpi(balanceMode,'undersample')
        tbl = countEachLabel(imdsAll);
        minCount = min(tbl.Count);
        imdsAll = splitEachLabel(imdsAll, minCount, 'randomize');
    end

    [imdsTrain, imdsVal] = splitEachLabel(imdsAll, valSplit, 'randomized');
    classes = categories(imdsTrain.Labels);
    nTrain = numel(imdsTrain.Files); nVal = numel(imdsVal.Files); nClasses = numel(classes);

    augmenter = imageDataAugmenter('RandXTranslation',[-2 2],'RandYTranslation',[-2 2]);
    augTrain = augmentedImageDatastore(inputSize(1:2), imdsTrain, 'DataAugmentation', augmenter);
    augVal   = augmentedImageDatastore(inputSize(1:2), imdsVal);

    cnt = countcats(imdsTrain.Labels);
    invFreq = 1 ./ max(1,double(cnt));
    clsWts  = invFreq / sum(invFreq) * nClasses;

    layers = [
        imageInputLayer(inputSize,'Normalization','none','Name','in')
        convolution2dLayer(3,32,'Padding','same','WeightsInitializer','he')
        batchNormalizationLayer; reluLayer
        convolution2dLayer(3,32,'Padding','same','WeightsInitializer','he')
        batchNormalizationLayer; reluLayer
        maxPooling2dLayer(2,'Stride',2)
        convolution2dLayer(3,64,'Padding','same'); batchNormalizationLayer; reluLayer
        convolution2dLayer(3,64,'Padding','same'); batchNormalizationLayer; reluLayer
        maxPooling2dLayer(2,'Stride',2)
        convolution2dLayer(3,128,'Padding','same'); batchNormalizationLayer; reluLayer
        dropoutLayer(0.4)
        maxPooling2dLayer(2,'Stride',2)
        globalAveragePooling2dLayer
        fullyConnectedLayer(nClasses)
        softmaxLayer
        classificationLayer('Classes',classes,'ClassWeights',clsWts)
    ];

    execEnv = 'gpu'; try gpuDevice(); catch, execEnv = 'auto'; end
    opts = trainingOptions('adam', ...
        'ExecutionEnvironment', execEnv, ...
        'InitialLearnRate', baseLR, ...
        'MaxEpochs', maxEpochs, 'MiniBatchSize', miniBatch, ...
        'Shuffle','every-epoch', ...
        'ValidationData', augVal, ...
        'ValidationPatience', valPatience, ...
        'Verbose', false);

    t0 = tic; net = trainNetwork(augTrain, layers, opts); trainTime = toc(t0);
    [YP,~] = classify(net, augVal); valAcc = mean(YP==imdsVal.Labels);
    fprintf('[train] ValAcc=%.2f%%  time=%.1fs  padScale=%.2f\n',100*valAcc,trainTime,padScale);

    inputSizeSave = inputSize; useCLAHE_save=useCLAHE;
    denoiseCfg=dn; padScale_save=padScale;
    save(modelFile,'net','classes','inputSizeSave','useCLAHE_save','denoiseCfg','padScale_save','valAcc','trainTime');
end

%% ============= 读图函数 (支持 padScale) =============
function Iout = reader_preproc_param(filename, outSz, preproc, outC)
    I = imread(filename);
    if size(I,3)>1, I = rgb2gray(I); end
    I = im2single(I);

    % 去噪
    if isfield(preproc,'denoise')
        M = preproc.denoise.method; P = preproc.denoise.param;
        switch lower(M)
            case 'median'
                k=max(3,2*floor(P/2)+1); I=medfilt2(I,[k k],'symmetric');
            case 'gaussian'
                sigma=max(0.01,P); k=max(3,2*floor(3*sigma)+1);
                I=imgaussfilt(I,sigma,'FilterSize',k);
        end
    end

    % CLAHE
    if isfield(preproc,'useCLAHE') && preproc.useCLAHE
        I=adapthisteq(I,'NumTiles',[8 8],'ClipLimit',0.02);
    end

    % letterbox + padScale
    S=outSz(1); T=outSz(2);
    padScale=1.0; 
    if isfield(preproc,'padScale'), padScale=preproc.padScale; end
    s=min(S/size(I,1), T/size(I,2)) * padScale;

    nh=max(1,round(size(I,1)*s));
    nw=max(1,round(size(I,2)*s));
    Ir=imresize(I,[nh nw],'nearest');
    Iout=ones(S,T,'single');
    y0=max(1,floor((S-nh)/2)+1);
    x0=max(1,floor((T-nw)/2)+1);
    y1=min(S,y0+nh-1);
    x1=min(T,x0+nw-1);
    Iout(y0:y1,x0:x1)=Ir(1:(y1-y0+1),1:(x1-x0+1));

    if nargin<4, outC=1; end
    if outC==3, Iout=repmat(Iout,[1 1 3]); end
end
