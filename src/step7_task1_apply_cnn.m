% STEP7_TASK1_APPLY_CNN
% Purpose:
%   - Apply the trained CNN to per-character crops produced by Step 6.
%   - Use a single-pass letterbox preprocess (white background + centered).
% Inputs:
%   state : struct carrying pipeline context
%           - state.segment.cropsBin : 1xN binary crops (white background, black glyph)
%           - state.step7.cnn.net    : trained CNN network
%           - state.step7.cnn.classes: class names
%           - state.step7.cnn.inputSize : [H W C] expected by CNN
%           - state.step7.cnn.useCLAHE  : (optional) logical
%   cfg   : struct with configuration and paths
%           - cfg.paths.figures, cfg.paths.results, cfg.paths.models
%
% Outputs (added to 'state'):
%   state.step7.task1_apply.labels : Nx1 string labels (after possible MLP override)
%   state.step7.task1_apply.scores : Nx1 confidence scores
%   state.step7.task1_apply.N      : number of crops
%   state.step7.task1_apply.*      : metadata (padScale, useCLAHE, confThr, targetSet)
%
% Saved artifacts:
%   - results/step7_task1_cnn_preds.csv                (if you write table elsewhere)
%   - results/figures/step7_task1_cnn_inputs_grid.png  (letterboxed inputs + final labels)
%   - results/figures/dbg_apply_cnn/cmp/*.png          (debug inputs)
%
% Notes:
%   - Single-pass preprocess only (no retry / no multi-scale).
%   - MLP override is limited to {A, H, 4} when CNN confidence < 0.70.
%   - CLAHE is optional and controlled by state.step7.cnn.useCLAHE.
%
% Config knobs (inside this file):
%   - padScale_main : letterbox scale factor (default 1.5)
%   - targetSet     : classes eligible for MLP override (default ["A","H","4"])

function state = step7_task1_apply_cnn(state, cfg)

    assert(isfield(state,'segment'), 'Missing step6 output.');
    assert(isfield(state.step7,'cnn') && isfield(state.step7.cnn,'net'), 'Missing CNN model.');

    net       = state.step7.cnn.net;
    classes   = state.step7.cnn.classes;
    if iscategorical(classes), classes = cellstr(classes); end
    inSize    = state.step7.cnn.inputSize;
    if numel(inSize)==2, inSize(3)=1; end
    useCLAHE  = true;
    if isfield(state.step7.cnn,'useCLAHE'), useCLAHE = state.step7.cnn.useCLAHE; end

    % ---- key params (single resize; no retries) ----
    padScale_main = 1.5;                 % letterbox scale (one pass)
    confThr       = 0.70;                % MLP override threshold for {A,H,4}
    targetSet     = ["A","H","4"];       % classes eligible for MLP override

    % ---- read step6 crops ----
    if isfield(state.segment,'cropsBin') && ~isempty(state.segment.cropsBin)
        crops = state.segment.cropsBin;  % white background, black glyph
        src = 'cropsBin';
    else
        error('state.segment.cropsBin not found. Ensure step6_segment ran.');
    end
    N = numel(crops);
    fprintf('[apply] %d character crops (white bg, black glyph), useCLAHE=%d\n', N, useCLAHE);

    % ---- try load MLP (optional fallback) ----
    mlpAvailable = false;
    mlp = struct();
    try
        modelsDir = fullfile('..','results','models');
        if ~exist(modelsDir,'dir')
            modelsDir = fullfile('..','rerults','models'); % typo-tolerant fallback
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
            mlp.classes   = Smlp.classes; if iscategorical(mlp.classes), mlp.classes = cellstr(mlp.classes); end
            mlp.inputSize = []; if isfield(Smlp,'inputSize'), mlp.inputSize = Smlp.inputSize; end
            mlp.mu        = []; if isfield(Smlp,'mu'),    mlp.mu = Smlp.mu; end
            mlp.sigma     = []; if isfield(Smlp,'sigma'), mlp.sigma = Smlp.sigma; end
            mlp.useCLAHE  = true; if isfield(Smlp,'useCLAHE'), mlp.useCLAHE = Smlp.useCLAHE; end
            mlpAvailable  = true;
        end
    catch
    end

    % ---- batch build (single letterbox) ----
    X = zeros(inSize(1), inSize(2), inSize(3), N, 'single');
    inUsed = zeros(inSize(1), inSize(2), inSize(3), N, 'single'); % actual network inputs

    outDbg = fullfile(cfg.paths.figures,'dbg_apply_cnn','cmp');
    if ~exist(outDbg,'dir'), mkdir(outDbg); end

    for i = 1:N
        Ci = crops{i};
        if ~isfloat(Ci), Ci = single(Ci); end
        Ci = max(0, min(1, Ci));

        Iproc = preprocess_for_cnn_pad(Ci, inSize(1:2), useCLAHE, inSize(3), padScale_main);
        X(:,:,:,i)      = Iproc;
        inUsed(:,:,:,i) = Iproc;

        try
            imwrite(uint8(Ci*255),    fullfile(outDbg, sprintf('step6_bin_%02d.png', i)));
            imwrite(uint8(Iproc*255), fullfile(outDbg, sprintf('input_%02d.png',     i)));
        catch
        end
    end

    % ---- CNN inference ----
    scores = predict(net, X);               % [N x C] or [N x C] depending on network
    [conf, idx] = max(scores, [], 2);
    labels = string(classes(idx));

    % ---- MLP override: only for {A,H,4} with low confidence ----
    if mlpAvailable
        for i = 1:N
            if conf(i) < confThr && any(labels(i) == targetSet)
                Imlp = inUsed(:,:,1,i);      % use the same network input (single channel)
                x = reshape(Imlp, [], 1);
                if ~isempty(mlp.mu) && ~isempty(mlp.sigma)
                    x = (x - mlp.mu) ./ mlp.sigma;
                end
                p = mlp_forward_predict_single(x, mlp.params); % C x 1
                [pMax, pIdx] = max(p);
                if pIdx>=1 && pIdx<=numel(mlp.classes)
                    labels(i) = string(mlp.classes{pIdx});
                    conf(i)   = pMax;
                end
            end
        end
    end

    % ---- visualization ----
    try
        cols = 12; rows = ceil(N/cols);
        f = figure('Color','w','Name','CNN inputs (final)');
        tiledlayout(rows, cols, 'TileSpacing', 'compact', 'Padding', 'compact');
        for i = 1:N
            nexttile;
            imshow(inUsed(:,:,:,i),'Border','tight');
            title(sprintf('%s(%.2f)', labels(i), conf(i)), 'FontSize', 8);
        end
        exportgraphics(f, fullfile(cfg.paths.figures,'step7_task1_cnn_inputs_grid.png'), 'Resolution', 150);
        close(f);
    catch
    end

    % ---- write back ----
    state.step7.task1_apply = struct( ...
        'labels',   labels, ...
        'scores',   conf, ...
        'N',        N, ...
        'padScale', padScale_main, ...
        'source',   src, ...
        'useCLAHE', useCLAHE, ...
        'confThr',  confThr, ...
        'targetSet',targetSet, ...
        'retry',    false );
end

% ===================================================================
% Single-pass letterbox (white background + centered)
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
    canvas = ones(S, T, 'single');      % white background
    y0 = max(1, floor((S - nh)/2) + 1);
    x0 = max(1, floor((T - nw)/2) + 1);
    y1 = min(S, y0 + nh - 1);
    x1 = min(T, x0 + nw - 1);
    Ir = Ir(1:(y1-y0+1), 1:(x1-x0+1));
    canvas(y0:y1, x0:x1) = Ir;
    if nargin<4 || outC==1
        Iout = canvas;
    else
        Iout = repmat(canvas, [1 1 3]);
    end
end

% ===================================================================
% Single-sample MLP forward (hidden ReLU, output softmax)
function P = mlp_forward_predict_single(x, params)
% x: D x 1
    L = numel(params.W);
    A = x;
    for l = 1:L-1
        Z = params.W{l} * A + params.b{l};
        A = max(0, Z);     % ReLU
    end
    ZL = params.W{L} * A + params.b{L};
    ZL = ZL - max(ZL,[],1);
    P  = exp(ZL); P = P ./ sum(P,1);   % C x 1
end
