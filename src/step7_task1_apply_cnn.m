function state = step7_task1_apply_cnn(state, cfg)
% STEP7_TASK1_APPLY_CNN
% 推理阶段：一次性白边缩放 + 居中；对 {A,H,4} 且 CNN 置信<0.70 时用 MLP 覆盖
% 作者: ChatGPT 改写 2025

    assert(isfield(state,'segment'),'缺少 step6 输出');
    assert(isfield(state.step7,'cnn') && isfield(state.step7.cnn,'net'),'缺少 CNN 模型');

    net       = state.step7.cnn.net;
    classes   = state.step7.cnn.classes;
    if iscategorical(classes), classes = cellstr(classes); end
    inSize    = state.step7.cnn.inputSize;
    if numel(inSize)==2, inSize(3)=1; end
    useCLAHE  = true;
    if isfield(state.step7.cnn,'useCLAHE'), useCLAHE = state.step7.cnn.useCLAHE; end

    % —— 关键参数（仅一次缩放，不重试）——
    padScale_main = 1.5;           % 白边缩放比例（一次性）
    confThr       = 0.70;          % 对 {A,H,4} 且置信低于此阈值，用 MLP 覆盖
    targetSet     = ["A","H","4"]; % 仅对这三类启用 MLP 覆盖

    % ---------------- 读取 step6 小块 ----------------
    if isfield(state.segment,'cropsBin') && ~isempty(state.segment.cropsBin)
        crops = state.segment.cropsBin;  % 白底黑字
        src = 'cropsBin';
    else
        error('未找到 state.segment.cropsBin，请确认 step6_segment 已运行');
    end
    N = numel(crops);
    fprintf('[apply] 使用 %d 个字符块 (白底黑字)，useCLAHE=%d\n', N, useCLAHE);

    % ---------------- 尝试加载 MLP（可选兜底） ----------------
    mlpAvailable = false;
    mlp = struct();
    try
        modelsDir = fullfile('..','results','models');
        if ~exist(modelsDir,'dir')
            modelsDir = fullfile('..','rerults','models'); % 容错
        end
        mf = fullfile(modelsDir,'MLP_latest.mat');
        if exist(mf,'file')~=2
            dd = dir(fullfile(modelsDir,'MLP*.mat'));
            if ~isempty(dd)
                [~,ix]=max([dd.datenum]); mf = fullfile(dd(ix).folder,dd(ix).name);
            end
        end
        if exist(mf,'file')==2
            Smlp = load(mf);
            mlp.params    = Smlp.params;
            mlp.classes   = Smlp.classes; if iscategorical(mlp.classes), mlp.classes=cellstr(mlp.classes); end
            mlp.inputSize = []; if isfield(Smlp,'inputSize'), mlp.inputSize = Smlp.inputSize; end
            mlp.mu        = []; if isfield(Smlp,'mu'),    mlp.mu=Smlp.mu; end
            mlp.sigma     = []; if isfield(Smlp,'sigma'), mlp.sigma=Smlp.sigma; end
            mlp.useCLAHE  = true; if isfield(Smlp,'useCLAHE'), mlp.useCLAHE=Smlp.useCLAHE; end
            mlpAvailable  = true;
            fprintf('[apply] 已加载 MLP 兜底: %s\n', mf);
        end
    catch
        % 忽略加载错误
    end

    % ---------------- 一次性生成 batch（白边+居中缩放，仅一次） ----------------
    X = zeros(inSize(1), inSize(2), inSize(3), N, 'single');
    inUsed = zeros(inSize(1), inSize(2), inSize(3), N, 'single'); % 实际送入网络的图

    outDbg = fullfile(cfg.paths.figures,'dbg_apply_cnn','cmp');
    if ~exist(outDbg,'dir'), mkdir(outDbg); end

    for i = 1:N
        Ci = crops{i};
        if ~isfloat(Ci), Ci = single(Ci); end
        Ci = max(0,min(1,Ci));

        Iproc = preprocess_for_cnn_pad(Ci, inSize(1:2), useCLAHE, inSize(3), padScale_main);
        X(:,:,:,i) = Iproc;
        inUsed(:,:,:,i) = Iproc;

        try
            imwrite(uint8(Ci*255),   fullfile(outDbg,sprintf('step6_bin_%02d.png',i)));
            imwrite(uint8(Iproc*255),fullfile(outDbg,sprintf('input_%02d.png',i)));
        catch
        end
    end

    % ---------------- CNN 预测（仅一次） ----------------
    scores = predict(net,X);               % [N x C]
    [conf,idx] = max(scores,[],2);
    labels = string(classes(idx));

    % ---------------- 覆盖策略：仅当 {A,H,4} 且置信<0.70，用 MLP Top-1 覆盖 ----------------
    if mlpAvailable
        for i=1:N
            if conf(i) < confThr && any(labels(i) == targetSet)
                Imlp = inUsed(:,:,1,i);      % 使用同一入网图的单通道
                x = reshape(Imlp,[],1);
                if ~isempty(mlp.mu) && ~isempty(mlp.sigma)
                    x = (x - mlp.mu) ./ mlp.sigma;
                end
                p = mlp_forward_predict_single(x, mlp.params); % C x 1
                [pMax, pIdx] = max(p);
                if pIdx>=1 && pIdx<=numel(mlp.classes)
                    labels(i) = string(mlp.classes{pIdx}); % 直接采用 MLP 结果
                    conf(i)   = pMax;
                end
            end
        end
    end

    % ---------------- 网格可视化 ----------------
    try
        cols = 12; rows = ceil(N/cols);
        f = figure('Color','w','Name','CNN inputs (final)');
        tiledlayout(rows,cols,'TileSpacing','compact','Padding','compact');
        for i=1:N
            nexttile;
            imshow(inUsed(:,:,:,i),'Border','tight');
            title(sprintf('%s(%.2f)',labels(i),conf(i)),'FontSize',8);
        end
        exportgraphics(f,fullfile(cfg.paths.figures,'step7_task1_cnn_inputs_grid.png'),'Resolution',150);
        close(f);
    catch
    end

    % 回写状态
    state.step7.task1_apply = struct('labels',labels,'scores',conf,'N',N, ...
        'padScale',padScale_main,'source',src,'useCLAHE',useCLAHE, ...
        'confThr',confThr,'targetSet',targetSet,'retry',false);
end

% ===================================================================
% 一次性白边+居中缩放（letterbox）
function Iout = preprocess_for_cnn_pad(I, outSz, useCLAHE, outC, padScale)
    if size(I,3)>1, I = rgb2gray(I); end
    I = im2single(I);
    if useCLAHE
        I = adapthisteq(I,'NumTiles',[8 8],'ClipLimit',0.02);
    end
    S = outSz(1); T = outSz(2);
    s  = min(S/size(I,1), T/size(I,2)) / padScale;
    nh = max(1, min(S, round(size(I,1)*s)));
    nw = max(1, min(T, round(size(I,2)*s)));
    Ir = imresize(I, [nh nw], 'nearest');
    canvas = ones(S,T,'single');           % 白底
    y0 = max(1, floor((S - nh)/2) + 1);
    x0 = max(1, floor((T - nw)/2) + 1);
    y1 = min(S, y0 + nh - 1);
    x1 = min(T, x0 + nw - 1);
    Ir = Ir(1:(y1-y0+1), 1:(x1-x0+1));
    canvas(y0:y1, x0:x1) = Ir;
    if nargin<4 || outC==1
        Iout = canvas;
    else
        Iout = repmat(canvas,[1 1 3]);
    end
end

% ===================================================================
% 单样本 MLP 前向（隐层 ReLU，输出 softmax）
function P = mlp_forward_predict_single(x, params)
% x: D x 1
    L = numel(params.W);
    A = x;
    for l=1:L-1
        Z = params.W{l} * A + params.b{l};
        A = max(0, Z);     % ReLU
    end
    ZL = params.W{L} * A + params.b{L};
    ZL = ZL - max(ZL,[],1);
    P  = exp(ZL); P = P ./ sum(P,1);   % C x 1
end
