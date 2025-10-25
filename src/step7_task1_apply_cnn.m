function state = step7_task1_apply_cnn(state, cfg)
% STEP7_TASK1_APPLY_CNN - 推理阶段（带大白边、居中）
% 
% 功能：
%   1. 使用 Step6 的裁剪结果 (优先 cropsBin 白底黑字)
%   2. 按 test.m 同逻辑预处理 + 可调白框 padScale
%   3. 生成 CNN 输入，并输出预测标签与置信度
%
% 作者: ChatGPT 改写 2025
% 兼容性: MATLAB R2021a+

    assert(isfield(state,'segment'),'缺少 step6 输出');
    assert(isfield(state.step7,'cnn') && isfield(state.step7.cnn,'net'),'缺少 CNN 模型');

    net       = state.step7.cnn.net;
    classes   = state.step7.cnn.classes;
    inSize    = state.step7.cnn.inputSize;
    if numel(inSize)==2, inSize(3)=1; end
    useCLAHE  = true;
    if isfield(state.step7.cnn,'useCLAHE'), useCLAHE = state.step7.cnn.useCLAHE; end

    padScale  = 1.6;  % <<< 关键参数：整体缩放时额外白边比例 (1.3~1.5 都行)

    % ---------------- 读取 step6 小块 ----------------
    if isfield(state.segment,'cropsBin') && ~isempty(state.segment.cropsBin)
        crops = state.segment.cropsBin;  % 白底黑字
        src = 'cropsBin';
    else
        error('未找到 state.segment.cropsBin，请确认 step6_segment 已运行');
    end

    N = numel(crops);
    fprintf('[apply] 使用 %d 个字符块 (白底黑字)，useCLAHE=%d\n', N, useCLAHE);

    % ---------------- 生成 batch ----------------
    X = zeros(inSize(1), inSize(2), inSize(3), N, 'single');

    outDbg = fullfile(cfg.paths.figures,'dbg_apply_cnn','cmp');
    if ~exist(outDbg,'dir'), mkdir(outDbg); end

    for i = 1:N
        Ci = crops{i};
        if ~isfloat(Ci), Ci = single(Ci); end
        Ci = max(0,min(1,Ci));

        % 预处理 + 大白框
        Iproc = preprocess_for_cnn_pad(Ci, inSize(1:2), useCLAHE, inSize(3), padScale);
        X(:,:,:,i) = Iproc;

        try
            imwrite(uint8(Ci*255), fullfile(outDbg,sprintf('step6_bin_%02d.png',i)));
            imwrite(uint8(Iproc*255), fullfile(outDbg,sprintf('input_%02d.png',i)));
        catch
        end
    end

    % ---------------- CNN 预测 ----------------
    scores = predict(net,X);
    [conf,idx] = max(scores,[],2);
    labels = string(classes(idx));

    % ---------------- 保存结果 ----------------
    T = table((1:N).',labels,conf,'VariableNames',{'Index','Label','Score'});
    writetable(T,fullfile(cfg.paths.results,'step7_task1_cnn_preds.csv'));
    fprintf('[apply] 预测完成，结果写入 step7_task1_cnn_preds.csv\n');

    % ---------------- 网格可视化 ----------------
    try
        cols = 12; rows = ceil(N/cols);
        f = figure('Color','w','Name','CNN inputs (debug)');
        tiledlayout(rows,cols,'TileSpacing','compact','Padding','compact');
        for i=1:N
            nexttile;
            imshow(X(:,:,:,i),'Border','tight');
            title(sprintf('%s(%.2f)',labels(i),conf(i)),'FontSize',8);
        end
        exportgraphics(f,fullfile(cfg.paths.figures,'step7_task1_cnn_inputs_grid.png'),'Resolution',150);
        close(f);
    catch
    end

    % 回写状态
    state.step7.task1_apply = struct('labels',labels,'scores',conf,'N',N, ...
        'padScale',padScale,'source',src,'useCLAHE',useCLAHE);
end


% ===================================================================
% 与 test.m 一致 + 增强版白框函数
function Iout = preprocess_for_cnn_pad(I, outSz, useCLAHE, outC, padScale)
    if size(I,3)>1, I = rgb2gray(I); end
    I = im2single(I);

    % CLAHE 可选
    if useCLAHE
        I = adapthisteq(I,'NumTiles',[8 8],'ClipLimit',0.02);
    end

    % ==== Step1: 按比例缩放 ====
    S = outSz(1); T = outSz(2);
    s  = min(S/size(I,1), T/size(I,2)) / padScale;  % padScale <1 表示扩大边框
    nh = max(1, min(S, round(size(I,1)*s)));
    nw = max(1, min(T, round(size(I,2)*s)));
    Ir = imresize(I, [nh nw], 'nearest');

    % ==== Step2: 放置在更大的白底中央 ====
    canvas = ones(S,T,'single');  % 白底
    y0 = max(1, floor((S - nh)/2) + 1);
    x0 = max(1, floor((T - nw)/2) + 1);
    y1 = min(S, y0 + nh - 1);
    x1 = min(T, x0 + nw - 1);
    Ir = Ir(1:(y1-y0+1), 1:(x1-x0+1));
    canvas(y0:y1, x0:x1) = Ir;

    % ==== Step3: 匹配通道 ====
    if nargin<4 || outC==1
        Iout = canvas;
    else
        Iout = repmat(canvas,[1 1 3]);
    end
end
