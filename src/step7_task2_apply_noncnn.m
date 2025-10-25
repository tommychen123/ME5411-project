function state = step7_task1_apply_cnn(state, cfg)
% STEP7_TASK1_APPLY_CNN - Inference using Step6 crops directly (with large white border)
% Uses:
%   state.segment.cropsBW or state.segment.cropsGray
%   state.step7.cnn.net / .classes / .inputSize
% Saves:
%   figures/dbg_apply_cnn/in_##.png  (真正进网的128x128)
%   results/step7_task1_cnn_preds.csv

    % ---- preconditions ----
    assert(isfield(state,'segment') && ...
          (isfield(state.segment,'cropsBW') || isfield(state.segment,'cropsGray')), ...
          'Missing Step6 crops. Run step6_segment first.');
    assert(isfield(state,'step7') && isfield(state.step7,'cnn'), ...
          'Missing CNN. Train or load model first.');

    net     = state.step7.cnn.net;
    classes = state.step7.cnn.classes;
    inSize  = state.step7.cnn.inputSize(1:2);   % e.g. [128 128]

    % ---- choose source crops (prefer BW) ----
    if isfield(state.segment,'cropsBW') && ~isempty(state.segment.cropsBW)
        crops = state.segment.cropsBW;    % cell of logical
        srcType = 'bw';
    else
        crops = state.segment.cropsGray;  % cell of double in [0,1]
        srcType = 'gray';
    end
    N = numel(crops);
    assert(N>0, 'No segments found from Step6.');

    % ---- big white border settings (you可调) ----
    targetHFrac   = 0.55;   % 字符高度占输入高度的比例（越小白边越大）
    useCLAHE      = true;   % 微提升对比，和训练端一致

    % ---- build batch + save debug images ----
    X = zeros(inSize(1), inSize(2), 1, N, 'single');
    dbgDir = fullfile(cfg.paths.figures, 'dbg_apply_cnn');
    if ~exist(dbgDir,'dir'), mkdir(dbgDir); end

    for i = 1:N
        Ci = crops{i};

        % 统一成单通道 float，白底黑字（与训练一致）
        switch srcType
            case 'bw'
                % Step6 二值：true=前景（字）。我们要黑字→0，白底→1
                Ci = single(Ci);
                Ci = 1 - Ci;                  % 前景置黑
            case 'gray'
                Ci = im2single(Ci);
                % 若灰度均值偏暗（黑底白字），则反相到白底黑字
                if mean(Ci(:)) < 0.5
                    Ci = 1 - Ci;
                end
        end

        % 居中+大白边 letterbox 到 CNN 输入大小
        Xin = letterbox_white_with_margin(Ci, inSize, useCLAHE, targetHFrac);
        X(:,:,:,i) = Xin;

        % dump debug 进网图
        try, imwrite(uint8(Xin*255), fullfile(dbgDir, sprintf('in_%02d.png', i))); catch, end
    end

    % ---- forward ----
    scores = predict(net, X);                 % [N x C]
    [conf, idx] = max(scores, [], 2);
    labels = strings(N,1);
    for i=1:N, labels(i) = string(classes(idx(i))); end

    % ---- export CSV ----
    if ~exist(cfg.paths.results,'dir'), mkdir(cfg.paths.results); end
    T = table((1:N).', labels, conf, 'VariableNames',{'Index','Label','Score'});
    writetable(T, fullfile(cfg.paths.results,'step7_task1_cnn_preds.csv'));

    % ---- optional: contact sheet（可注释掉）----
    try
        cols = 12; rows = ceil(N/cols);
        f = figure('Color','w','Name','CNN inputs (debug)'); tiledlayout(rows,cols,'TileSpacing','compact','Padding','compact');
        for i=1:N
            nexttile; imshow(X(:,:,:,i),'Border','tight');
            title(sprintf('%s(%.2f)', labels(i), conf(i)), 'FontSize',8);
        end
        exportgraphics(f, fullfile(cfg.paths.figures,'step7_task1_cnn_inputs_grid.png'), 'Resolution',150);
        close(f);
    catch
    end

    % ---- write back ----
    state.step7.task1_apply = struct('labels',labels,'scores',conf,'N',N, ...
                                     'source',srcType, 'targetHFrac',targetHFrac);
end

% ================= helpers =================
function Xin = letterbox_white_with_margin(I, outSz, useCLAHE, targetHFrac)
% 将字符等比缩放到 outSz 中的 targetHFrac*height，并居中贴到白底上
% I：单通道single，已是白底黑字（0=黑字, 1=白底）
    if nargin<3, useCLAHE = true; end
    if nargin<4, targetHFrac = 0.55; end

    % 轻度对比增强（训练端也有）
    if useCLAHE
        % I 范围[0,1]，为了避免极端噪点，clip 住
        I = max(0,min(1,I));
        I = adapthisteq(I,'NumTiles',[8 8],'ClipLimit',0.02);
    end

    S = outSz(1); T = outSz(2);
    targetH = max(4, floor(S * targetHFrac));     % 目标字符高度
    s = targetH / size(I,1);
    nh = max(1, round(size(I,1)*s));
    nw = max(1, round(size(I,2)*s));
    Ir = imresize(I, [nh nw], 'nearest');         % 保边缘清晰

    Xin = ones(S,T,'single');                     % 白底
    y0 = floor((S - nh)/2) + 1;
    x0 = floor((T - nw)/2) + 1;
    y1 = min(S, y0 + nh - 1);
    x1 = min(T, x0 + nw - 1);
    Ir = Ir(1:(y1-y0+1), 1:(x1-x0+1));
    Xin(y0:y1, x0:x1) = Ir;
end
