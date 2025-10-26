function state = step7_task2_apply_mlp(state, cfg)
% STEP7_TASK2_APPLY_MLP
% Purpose:
%   - Apply a trained MLP model to Step-6 character crops.
%   - Auto retry once with a different padScale if confidence < 0.70.
%   - Produce a grid visualization of the final used inputs.
%
% Auto-load model from:
%   ../results/models/MLP_latest.mat
%   If missing, fallback to the newest MLP*.mat in the same folder.
%
% Inputs:
%   state.segment.cropsBin : 1xN binary crops (white background, black glyph)
%   cfg.paths.results      : path to results (models are expected here)
%   cfg.paths.figures      : path to figures
%
% Outputs (added to 'state'):
%   state.step7.task2_apply.labels : Nx1 string labels
%   state.step7.task2_apply.scores : Nx1 confidence scores
%   state.step7.task2_apply.N      : number of crops
%   state.step7.task2_apply.*      : metadata (useCLAHE, padScale, modelFile)
%
% Saved:
%   results/figures/step7_task2_mlp_inputs_grid.png

    %% ----- paths -----
    if nargin<2, cfg = struct(); end
    if isfield(cfg,'paths') && isfield(cfg.paths,'results')
        resultsDir = cfg.paths.results;
    else
        resultsDir = fullfile('..','results');
    end
    if isfield(cfg,'paths') && isfield(cfg.paths,'figures')
        figDir = cfg.paths.figures;
    else
        figDir = fullfile('..','results','figures');
    end
    if ~exist(resultsDir,'dir'), mkdir(resultsDir); end
    if ~exist(figDir,'dir'),   mkdir(figDir);   end

    %% ----- load model -----
    modelsDir = fullfile(resultsDir,'models');
    if ~exist(modelsDir,'dir')
        modelsDir = fullfile('..','rerults','models'); % typo-tolerant fallback
    end
    modelFile = fullfile(modelsDir,'MLP_latest.mat');
    if exist(modelFile,'file')~=2
        dd = dir(fullfile(modelsDir,'MLP*.mat'));
        assert(~isempty(dd),'No MLP model found in %s', modelsDir);
        [~,ix]=max([dd.datenum]);
        modelFile=fullfile(dd(ix).folder,dd(ix).name);
    end
    fprintf('[apply-mlp] Loading model: %s\n', modelFile);
    S = load(modelFile);
    params  = S.params;
    classes = S.classes; if iscategorical(classes), classes = cellstr(classes); end
    inSize  = S.inputSize; if numel(inSize)==2, inSize(3)=1; end
    useCLAHE = true; if isfield(S,'useCLAHE'), useCLAHE = S.useCLAHE; end
    mu = []; sigma = [];
    if isfield(S,'mu'),    mu = S.mu; end
    if isfield(S,'sigma'), sigma = S.sigma; end

    %% ----- inputs -----
    assert(isfield(state,'segment') && isfield(state.segment,'cropsBin'), ...
        'Missing Step-6 crops: state.segment.cropsBin');
    crops = state.segment.cropsBin;
    N = numel(crops);
    fprintf('[apply-mlp] %d character crops\n', N);

    %% ----- params -----
    padScale_default = 1.35;   % initial letterbox scale
    padScale_retry   = 1.20;   % retry scale if low confidence
    D = prod(inSize(1:2)); %#ok<NASGU>

    %% ----- storage for visualization & results -----
    Ximg   = zeros(inSize(1), inSize(2), inSize(3), N, 'single');  % final used inputs
    labels = strings(N,1);
    scores = zeros(N,1);

    %% ----- inference with optional low-confidence retry -----
    fprintf('\n=== MLP predictions (with low-confidence retry) ===\n');
    for i=1:N
        Ci = crops{i};
        if ~isfloat(Ci), Ci = single(Ci); end
        Ci = max(0, min(1, Ci));

        % first pass
        [label1, conf1, probs1, Iproc1] = run_once(Ci, params, classes, inSize, useCLAHE, padScale_default, mu, sigma);
        chosenI = Iproc1; chosenLabel = label1; chosenConf = conf1; chosenProbs = probs1;

        % retry if confidence is low
        if conf1 < 0.7
            [label2, conf2, probs2, Iproc2] = run_once(Ci, params, classes, inSize, useCLAHE, padScale_retry, mu, sigma);
            if conf2 > conf1
                chosenI = Iproc2; chosenLabel = label2; chosenConf = conf2; chosenProbs = probs2;
                fprintf('#%02d → Pred: %-3s  Conf: %.2f%%  (retry improved from %.2f%%)\n', ...
                        i, chosenLabel, chosenConf*100, conf1*100);
            else
                fprintf('#%02d → Pred: %-3s  Conf: %.2f%%  (retry no improvement)\n', ...
                        i, chosenLabel, chosenConf*100);
            end
        else
            fprintf('#%02d → Pred: %-3s  Conf: %.2f%%\n', i, chosenLabel, chosenConf*100);
        end

        % print probability vector
        fprintf('   probs: ');
        for c=1:numel(classes)
            fprintf('%s=%.2f ', classes{c}, chosenProbs(c)*100);
        end
        fprintf('\n');

        % keep final choice
        labels(i) = chosenLabel;
        scores(i) = chosenConf;
        Ximg(:,:,:,i) = chosenI;
    end
    fprintf('=================================================\n');

    %% ----- grid visualization -----
    try
        cols = 12; rows = ceil(N/cols);
        f = figure('Color','w','Name','MLP inputs (final used)');
        tiledlayout(rows, cols, 'TileSpacing','compact','Padding','compact');
        for i=1:N
            nexttile;
            imshow(Ximg(:,:,:,i),'Border','tight');
            title(sprintf('%s(%.2f)', labels(i), scores(i)), 'FontSize', 8);
        end
        exportgraphics(f, fullfile(figDir,'step7_task2_mlp_inputs_grid.png'), 'Resolution', 150);
        close(f);
        fprintf('[apply-mlp] Saved grid: %s\n', fullfile(figDir,'step7_task2_mlp_inputs_grid.png'));
    catch
        % ignore visualization errors
    end

    %% ----- write back -----
    state.step7.task2_apply = struct( ...
        'labels',   labels, ...
        'scores',   scores, ...
        'N',        N, ...
        'useCLAHE', useCLAHE, ...
        'padScale', [padScale_default padScale_retry], ...
        'modelFile', modelFile);
end

% ================= internal helpers =================
function [label, conf, probs, Iproc] = run_once(Ci, params, classes, inSize, useCLAHE, padScale, mu, sigma)
    Iproc = preprocess_for_cnn_pad(Ci, inSize(1:2), useCLAHE, inSize(3), padScale);
    x = reshape(Iproc(:,:,1), [], 1);
    if ~isempty(mu) && ~isempty(sigma)
        x = (x - mu) ./ sigma;
    end
    probs = mlp_forward_predict(x, params);
    [conf, idx] = max(probs);
    label = string(classes(idx));
end

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
    canvas = ones(S, T, 'single'); % white background
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

function P = mlp_forward_predict(X, params)
% X: D x 1
    L = numel(params.W);
    A = X;
    for l = 1:L-1
        Z = params.W{l} * A + params.b{l};
        A = max(0, Z); % ReLU
    end
    ZL = params.W{L} * A + params.b{L};
    ZL = ZL - max(ZL, [], 1);
    P  = exp(ZL); P = P ./ sum(P, 1);
end
