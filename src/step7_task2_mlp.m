function state = step7_task2_mlp(state, cfg)
% STEP7_TASK2_NON_CNN - 非CNN的多层感知机（MLP）字符分类
% 要求数据结构：
%   cfg.file.datasetRoot  -> 数据集根目录（类为子文件夹）
%   例如: ../data/p_dataset_26/
%           0/*.png  4/*.png  7/*.png  8/*.png  A/*.png  D/*.png  H/*.png
%
% 输出：
%   ../results/models/MLP_latest.mat  (params, classes, inputSize, mu, sigma, valAcc, trainTime)
%   ../results/figures/mlp_curves.png (loss/acc 曲线)
%
% state.step7.task2_mlp = struct('params',..., 'classes',..., 'inputSize',..., 'valAcc',..., 'trainTime',...)

    %% ---------- 配置 ----------
    rng(42);                             % 可复现
    datasetRoot = fullfile('..','data','dataset_2025');
    assert(exist(datasetRoot,'dir')==7, '数据集不存在: %s', datasetRoot);

    resultsDir = fullfile(cfg.paths.results);
    modelDir   = fullfile(resultsDir, 'models');
    figDir     = fullfile(resultsDir, 'figures');
    if ~exist(modelDir,'dir'), mkdir(modelDir); end
    if ~exist(figDir,'dir'),   mkdir(figDir);   end
    modelFile  = fullfile(modelDir, 'MLP_latest.mat');

    % 预处理相关（与整个工程的风格一致）
    inputSize  = [128 128 1];   % 数据本身就是 128x128；保持一致
    useCLAHE   = true;          % 适度增强细节（和你CNN一致）
    valSplit   = 0.75;          % 按你学长代码的风格：75% 训练 / 25% 验证

    % 模型/训练超参（稳泛化）
    layers     = [prod(inputSize(1:2)) 1024 64];  % 隐层(1024, 64) + 输出层会按 numClasses 自动加
    epochs     = 200;              % 足够收敛
    batchSize  = 128;              % mini-batch
    lr         = 0.02;             % 学习率（SGD）
    l2         = 1e-4;             % L2 权重衰减（仅W）
    printFreq  = 5;                % 每多少个epoch打印/记录一次

    %% ---------- 读数据（统一预处理） ----------
    rf = @(fn) reader_preproc(fn, inputSize(1:2), useCLAHE);
    imdsAll = imageDatastore(datasetRoot, 'IncludeSubfolders', true, ...
        'LabelSource', 'foldernames', 'ReadFcn', rf);

    % 划分训练/验证
    [imdsTrain, imdsVal] = splitEachLabel(imdsAll, valSplit, 'randomized');
    classes    = categories(imdsTrain.Labels);
    numClasses = numel(classes);
    fprintf('[task2-mlp] %d train / %d val / %d classes\n', numel(imdsTrain.Files), numel(imdsVal.Files), numClasses);

    % 读成矩阵（展平特征: D x N）
    Xtr = read_and_flatten(imdsTrain, inputSize);   % D x Ntr
    ytr = onehot(imdsTrain.Labels, classes);        % C x Ntr
    Xva = read_and_flatten(imdsVal,   inputSize);   % D x Nva
    yva = onehot(imdsVal.Labels, classes);          % C x Nva

    % 标准化（按训练集统计）
    mu    = mean(Xtr, 2);
    sigma = std(Xtr, 0, 2) + 1e-6;
    Xtr   = (Xtr - mu) ./ sigma;
    Xva   = (Xva - mu) ./ sigma;

    %% ---------- 初始化 MLP ----------
    params = init_mlp([layers numClasses]);  % 自动拼上输出层

    %% ---------- 训练（mini-batch SGD + ReLU + SoftmaxCE） ----------
    nTr = size(Xtr, 2);
    itersPerEpoch = ceil(nTr / batchSize);
    history = struct('epoch',[],'trLoss',[],'trAcc',[],'vaLoss',[],'vaAcc',[]);

    fprintf('[task2-mlp] Training ...\n');
    t0 = tic;
    for ep = 1:epochs
        % 打乱
        idx = randperm(nTr);
        Xtr = Xtr(:, idx);  ytr = ytr(:, idx);

        for it = 1:itersPerEpoch
            s = (it-1)*batchSize + 1;  e = min(it*batchSize, nTr);
            Xb = Xtr(:, s:e);  Yb = ytr(:, s:e);

            % 前向
            [out, caches] = forward_mlp(Xb, params);
            % 损失（CE + L2）
            [loss, dZL] = loss_ce(out, Yb);
            % 反传
            grads = backward_mlp(dZL, caches, params);
            % L2
            for li=1:numel(params.W)
                grads.W{li} = grads.W{li} + l2 * params.W{li};
            end
            % 更新（SGD）
            for li=1:numel(params.W)
                params.W{li} = params.W{li} - lr * grads.W{li};
                params.b{li} = params.b{li} - lr * grads.b{li};
            end
        end

        % 记录/评估
        if mod(ep, printFreq)==0 || ep==1 || ep==epochs
            [trLoss, trAcc] = evaluate(Xtr, ytr, params);
            [vaLoss, vaAcc] = evaluate(Xva, yva, params);
            history.epoch(end+1)  = ep;
            history.trLoss(end+1) = trLoss;
            history.trAcc(end+1)  = trAcc;
            history.vaLoss(end+1) = vaLoss;
            history.vaAcc(end+1)  = vaAcc;
            fprintf('  ep=%3d | trLoss=%.4f trAcc=%.3f | vaLoss=%.4f vaAcc=%.3f\n', ...
                ep, trLoss, trAcc, vaLoss, vaAcc);
        end
    end
    trainTime = toc(t0);

    % 最终验证
    [~, valAcc] = evaluate(Xva, yva, params);
    fprintf('[task2-mlp] Val acc: %.2f%%  (time=%.1fs)\n', 100*valAcc, trainTime);

    %% ---------- 曲线图 ----------
    try
        f = figure('Color','w','Position',[100 100 780 320]);
        subplot(1,2,1); plot(history.epoch, history.trLoss, '-o'); hold on; plot(history.epoch, history.vaLoss, '-o'); grid on;
        xlabel('epoch'); ylabel('loss'); title('Cross-Entropy'); legend('train','val','Location','best');
        subplot(1,2,2); plot(history.epoch, history.trAcc, '-o'); hold on; plot(history.epoch, history.vaAcc, '-o'); grid on;
        xlabel('epoch'); ylabel('accuracy'); title('Accuracy'); legend('train','val','Location','best');
        exportgraphics(f, fullfile(figDir,'mlp_curves.png'),'Resolution',150); close(f);
    catch
    end

    %% ---------- 保存模型 ----------
    save(modelFile, 'params','classes','inputSize','useCLAHE','mu','sigma','valAcc','trainTime');
    fprintf('[task2-mlp] Model saved: %s\n', modelFile);

    %% ---------- 更新 state ----------
    if ~isfield(state,'step7'), state.step7 = struct(); end
    state.step7.task2_mlp = struct('params',params,'classes',{classes}, ...
        'inputSize',inputSize,'valAcc',valAcc,'trainTime',trainTime,'useCLAHE',useCLAHE, ...
        'mu',mu,'sigma',sigma);
end

%% ======================= Helper Functions =======================

function I = reader_preproc(fn, outSz, useCLAHE)
% 灰度 -> single[0,1] -> (可选)CLAHE -> 白底letterbox到 outSz
    I = imread(fn);
    if size(I,3)>1, I = rgb2gray(I); end
    I = im2single(I);
    if nargin<3, useCLAHE = true; end
    I = letterbox_white(I, outSz, useCLAHE);
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
    Iout = ones(S,T,'single');       % 白底
    Iout(y0:y1, x0:x1) = Ir;         % 居中
end

function X = read_and_flatten(imds, inputSize)
% 读取datastore为 D x N 的列向量特征
    D = prod(inputSize(1:2));
    N = numel(imds.Files);
    X = zeros(D, N, 'single');
    reset(imds);
    for k = 1:N
        I = readimage(imds, k);
        X(:,k) = reshape(I, [], 1);
    end
end

function Y = onehot(lbls, classes)
    C = numel(classes);
    N = numel(lbls);
    Y = zeros(C, N, 'single');
    [~, idx] = ismember(cellstr(lbls), classes);
    Y(sub2ind([C,N], idx(:)', 1:N)) = 1;
end

function params = init_mlp(layerSizes)
% layerSizes: [D, H1, H2, ..., C]
    L = numel(layerSizes) - 1;
    params.W = cell(1,L);
    params.b = cell(1,L);
    for l=1:L
        n_in  = layerSizes(l);
        n_out = layerSizes(l+1);
        if l < L
            % He init for ReLU
            params.W{l} = randn(n_out, n_in, 'single') * sqrt(2/n_in);
        else
            % Xavier for softmax
            params.W{l} = randn(n_out, n_in, 'single') * sqrt(1/n_in);
        end
        params.b{l} = zeros(n_out, 1, 'single');
    end
end

function [out, caches] = forward_mlp(X, params)
% 隐层 ReLU，最后 softmax
    L = numel(params.W);
    A = X;
    caches = struct('Z',{});

    for l = 1:L-1
        Z = params.W{l} * A + params.b{l};
        A = relu(Z);
        caches(l).Z = Z;       %#ok<*AGROW>
        if l == 1
            caches(l).Aprev = X;
        else
            caches(l).Aprev = caches(l-1).A;
        end
        caches(l).A = A;
    end

    % 输出层
    ZL = params.W{L} * A + params.b{L};
    out = softmax(ZL);
    caches(L).Z = ZL;
    caches(L).Aprev = A;
    caches(L).A = out;
end


function [loss, dZL] = loss_ce(AL, Y)
% 多类交叉熵（softmax + CE）：loss = - mean sum (y .* log(al))
    eps_ = 1e-12;
    loss = - mean( sum(single(Y) .* log(single(AL) + eps_), 1) );
    dZL  = AL - Y;  % softmax+CE 的标准梯度
end

function grads = backward_mlp(dZL, caches, params)
% 输出层梯度已给 dZL；隐层用 ReLU 反传
    L = numel(params.W);
    grads.W = cell(1,L); grads.b = cell(1,L);

    % 输出层
    Aprev = caches(L).Aprev;   % (H x N)
    N = size(Aprev,2);
    grads.W{L} = (1/N) * dZL * Aprev';
    grads.b{L} = (1/N) * sum(dZL, 2);
    dA = params.W{L}' * dZL;

    % 隐层
    for l=L-1:-1:1
        Z   = caches(l).Z;
        Aprev = caches(l).Aprev;
        N = size(Aprev,2);
        dZ  = dA .* (Z > 0);                 % ReLU'
        grads.W{l} = (1/N) * dZ * Aprev';
        grads.b{l} = (1/N) * sum(dZ, 2);
        if l>1
            Aprevprev = caches(l-1).Aprev; %#ok<NASGU>
        end
        dA  = params.W{l}' * dZ;
    end
end

function [loss, acc] = evaluate(X, Y, params)
    [AL, ~] = forward_mlp(X, params);
    [loss, ~] = loss_ce(AL, Y);
    [~, p] = max(AL, [], 1);
    [~, t] = max(Y,  [], 1);
    acc = mean(p==t);
end

%% ---------- simple activations ----------
function A = relu(Z), A = max(0, Z); end
function S = softmax(Z)
    Z = Z - max(Z,[],1);  % 数值稳定
    S = exp(Z);
    S = S ./ sum(S,1);
end


function P = mlp_forward(x, params)
% MLP 前向预测（单样本）
% 输入: x[D×1], params.W/b
    L = numel(params.W);
    A = x;
    for l=1:L-1
        Z = params.W{l} * A + params.b{l};
        A = max(0, Z); % ReLU
    end
    ZL = params.W{L} * A + params.b{L};
    ZL = ZL - max(ZL);
    P  = exp(ZL); 
    P  = P / sum(P);
end

