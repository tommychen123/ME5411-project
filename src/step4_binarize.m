function state = step4_binarize(state, cfg)
% STEP4_BINARIZE - Global Otsu thresholding (no toolbox).
% Goal:
%   Convert the ROI (grayscale in [0,1]) into a binary image using
%   a self-implemented Otsu threshold.
%
% Inputs:
%   state : expects state.roi (double in [0,1])
%   cfg.bin (optional):
%       .polarity : 'bright' (default) | 'dark'
%                   'bright' => foreground (text) = I >= thr
%                   'dark'   => foreground (text) = I <= thr
%       .nbins    : histogram bins for Otsu (default 256)
%
% Outputs (added to 'state'):
%   state.binarize.bw   : logical binary image (ROI-sized)
%   state.binarize.thr  : scalar threshold in [0,1]
%   state.binarize.info : struct with params
%
% Saved files (under cfg.paths.figures):
%   step4_binarize.png : side-by-side visualization (ROI vs Binary)

    assert(isfield(state,'roi'), 'state.roi is missing. Run step3_roi first.');
    I = state.roi;
    I = ensure_double01(I);

    % defaults
    if ~isfield(cfg,'bin'), cfg.bin = struct(); end
    if ~isfield(cfg.bin,'polarity'), cfg.bin.polarity = 'bright'; end
    if ~isfield(cfg.bin,'nbins'),    cfg.bin.nbins    = 256; end

    % compute Otsu threshold (self-implemented)
    thr = otsu_threshold(I, cfg.bin.nbins);

    % apply polarity
    switch lower(cfg.bin.polarity)
        case 'bright'
            BW = I >= thr;
        case 'dark'
            BW = I <= thr;
        otherwise
            error('Unsupported cfg.bin.polarity: %s', cfg.bin.polarity);
    end

    % save to state
    state.binarize = struct();
    state.binarize.bw   = BW;
    state.binarize.thr  = thr;
    state.binarize.info = cfg.bin;

    % visualize & save
    f = figure('Name','Step4 - Binarize (Otsu)','Color','w');
    tiledlayout(1,2,'TileSpacing','compact','Padding','compact');
    nexttile; imshow(I,'Border','tight');  title(sprintf('ROI (thr=%.4f)',thr));
    nexttile; imshow(BW,'Border','tight'); title(sprintf('Binary (%s)', cfg.bin.polarity));
    exportgraphics(f, fullfile(cfg.paths.figures,'step4_binarize.png'), 'Resolution',150);
    close(f);

    fprintf('[Step4] Otsu done. thr=%.4f | polarity=%s\n', thr, cfg.bin.polarity);
end

% ----------------- helpers -----------------
function thr = otsu_threshold(I, nbins)
% OTSU_THRESHOLD - classic Otsu on double [0,1], using 'nbins' bins.
    I = ensure_double01(I);
    edges  = linspace(0,1,nbins+1);
    counts = histcounts(I(:), edges);  % base MATLAB
    p  = counts / max(1,sum(counts));
    w  = cumsum(p);
    mu = cumsum(p .* (0:nbins-1));
    muT = mu(end);
    sigma_b2 = (muT*w - mu).^2 ./ max(w.*(1-w), eps);
    [~, idx] = max(sigma_b2);
    thr = (idx-1 + 0.5) / nbins;  % bin center in [0,1]
end

function J = ensure_double01(I)
    I = double(I);
    mn = min(I(:)); mx = max(I(:));
    if mx > mn, J = (I - mn) / (mx - mn);
    else,       J = zeros(size(I), 'like', I);
    end
end
