% STEP2_FILTER
% Purpose:
%   - Apply mean (box) filtering with adjustable window size(s)
%   - Use a toolbox-free implementation (no padarray/imfilter)
%   - Save individual outputs and a comparison grid for the report
%
% Inputs:
%   state : struct carrying pipeline context
%           requires state.imgEnh.minmax (or state.imgGray as fallback)
%   cfg   : struct with configuration
%           cfg.paths.figures : output figure path
%           cfg.filter.winSizes : vector of odd integers, e.g., [5] or [3 5 7]
%           cfg.filter.padMode  : 'replicate' | 'zero' (default 'replicate')
%
% Outputs (added to 'state'):
%   state.imgFilt.default : filtered image using the first win size
%   state.filtSweep       : struct with fields:
%                           .sizes  - list of window sizes used
%                           .images - cell array of filtered results
%                           .paths  - saved image paths for each size
%
% Saved figures:
%   results/figures/step2_filter_<ksize>.png
%   results/figures/step2_filter_sweep_grid.png
%
% Notes:
%   - Window size must be odd to have a centered kernel.
%   - This is a naive O(H*W*k^2) implementation suitable for moderate sizes.

function state = step2_filter(state, cfg)

    % --------- Check inputs ---------
    assert(isfield(state, 'imgEnh') || isfield(state, 'imgGray'), ...
        'Run step1 before step2_filter: missing state.imgEnh or state.imgGray.');

    % Prefer enhanced image (minmax) if available; else fall back to gray
    if isfield(state, 'imgEnh') && isfield(state.imgEnh, 'minmax')
        I = state.imgEnh.minmax;
    else
        I = state.imgGray;
    end
    I = ensure_double01(I);  % make sure in [0,1] double

    % Read config
    if ~isfield(cfg, 'filter'); cfg.filter = struct(); end
    if ~isfield(cfg.filter, 'winSizes'); cfg.filter.winSizes = 5; end
    if ~isfield(cfg.filter, 'padMode');  cfg.filter.padMode  = 'replicate'; end
    winSizes = cfg.filter.winSizes(:)';   % row vector

    % Validate sizes
    for k = winSizes
        assert(mod(k,2)==1 && k>=1, 'winSize must be odd and >=1. Got %d', k);
    end

    % --------- Do filtering for each window size ---------
    images = cell(1, numel(winSizes));
    outPaths = cell(1, numel(winSizes));

    tic;
    for idx = 1:numel(winSizes)
        ksize = winSizes(idx);
        F = local_mean_filter(I, ksize, cfg.filter.padMode);
        images{idx} = F;

        % Save single image
        f = figure('Name', sprintf('Step2 - Mean %dx%d', ksize, ksize), ...
                   'Color','w');
        imshow(F, 'Border', 'tight');
        title(sprintf('Mean Filter %dx%d', ksize, ksize));
        drawnow;
        outName = sprintf('step2_filter_%dx%d.png', ksize, ksize);
        outPath = fullfile(cfg.paths.figures, outName);
        exportgraphics(f, outPath, 'Resolution', 150);
        close(f);
        outPaths{idx} = outPath;
    end
    elapsed = toc;

    % First size as default
    state.imgFilt = struct();
    state.imgFilt.default = images{1};

    % Sweep info
    state.filtSweep = struct();
    state.filtSweep.sizes  = winSizes;
    state.filtSweep.images = images;
    state.filtSweep.paths  = outPaths;
    state.filtSweep.timeSec = elapsed;

    % --------- Comparison grid ---------
    f = figure('Name','Step2 - Filter Sweep','Color','w');
    n = numel(winSizes);
    cols = min(n, 3);
    rows = ceil(n / cols);
    tiledlayout(rows, cols, 'TileSpacing','compact','Padding','compact');

    for idx = 1:n
        nexttile;
        imshow(images{idx}, 'Border','tight');
        title(sprintf('Mean %dx%d', winSizes(idx), winSizes(idx)));
    end
    drawnow;
    gridPath = fullfile(cfg.paths.figures, 'step2_filter_sweep_grid.png');
    exportgraphics(f, gridPath, 'Resolution', 150);
    close(f);

    % --------- Console log ---------
    fprintf('[Step2] Mean filter done. Sizes: %s. Time: %.3fs\n', ...
        mat2str(winSizes), elapsed);
    fprintf('[Step2] Saved: %s\n', gridPath);
end

% ========= Helper: ensure image is double in [0,1] =========
function J = ensure_double01(I)
    I = double(I);
    mn = min(I(:)); mx = max(I(:));
    if mx > mn
        J = (I - mn) / (mx - mn);
    else
        J = zeros(size(I), 'like', I);
    end
end

% ========= Helper: local mean filter (toolbox-free) =========
function F = local_mean_filter(I, ksize, padMode)
% LOCAL_MEAN_FILTER
% Naive mean (box) filter with replicate/zero padding.
% I       : double image in [0,1]
% ksize   : odd integer (e.g., 3,5,7)
% padMode : 'replicate' or 'zero'
    r = floor(ksize/2);
    [H, W] = size(I);

    % Pad
    switch lower(padMode)
        case 'replicate'
            P = pad_replicate(I, r, r);
        case 'zero'
            P = pad_zero(I, r, r);
        otherwise
            error('Unsupported padMode: %s', padMode);
    end

    % Pre-allocate
    F = zeros(H, W);
    area = ksize * ksize;

    % Naive convolution
    for y = 1:H
        yy = y + r; % offset due to padding
        for x = 1:W
            xx = x + r;
            win = P(yy-r:yy+r, xx-r:xx+r);
            F(y,x) = sum(win(:)) / area;
        end
    end
end

% ========= Helper: replicate padding (toolbox-free) =========
function P = pad_replicate(I, pr, pc)
% PAD_REPLICATE
% Replicate border padding by pr rows and pc cols on all sides.
    [H, W] = size(I);
    P = zeros(H + 2*pr, W + 2*pc);
    % Center
    P(pr+1:pr+H, pc+1:pc+W) = I;
    % Top/bottom
    P(1:pr, pc+1:pc+W) = repmat(I(1,:), pr, 1);
    P(pr+H+1:end, pc+1:pc+W) = repmat(I(end,:), pr, 1);
    % Left/right (including corners after top/bottom)
    P(:, 1:pc) = repmat(P(:, pc+1), 1, pc);
    P(:, pc+W+1:end) = repmat(P(:, pc+W), 1, pc);
end

% ========= Helper: zero padding (toolbox-free) =========
function P = pad_zero(I, pr, pc)
% PAD_ZERO
% Zero padding by pr rows and pc cols on all sides.
    [H, W] = size(I);
    P = zeros(H + 2*pr, W + 2*pc);
    P(pr+1:pr+H, pc+1:pc+W) = I;
end
