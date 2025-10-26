% STEP4_BINARIZE
% Purpose:
%   - Robust global Otsu thresholding (self-implemented) with light
%     denoising and morphology to avoid fat blobs and sticking (plate-like text).
%
% Inputs:
%   state : struct carrying pipeline context (expects state.roi)
%   cfg   : struct with configuration (expects cfg.paths.*, optional cfg.bin.*)
%
% Outputs (added to 'state'):
%   state.binarize.bw    : logical mask of the ROI
%   state.binarize.thr   : final numeric threshold in [0,1]
%   state.binarize.info  : struct of effective parameters (incl. raw Otsu)
%
% Saved figure (when visualize=true):
%   results/figures/step4_binarize.png
function state = step4_binarize(state, cfg)

    %% ---- Preconditions ----
    assert(isfield(state,'roi'), 'state.roi is missing. Run step3_roi first.');
    I0 = state.roi;
    I  = ensure_double01(I0);

    %% ---- Defaults (overridable via cfg.bin.*) ----
    if ~isfield(cfg,'bin'), cfg.bin = struct(); end
    B = cfg.bin;

    % polarity: 'bright' (foreground bright) or 'dark'
    if ~isfield(B,'polarity'),     B.polarity     = 'bright'; end
    if ~isfield(B,'nbins'),        B.nbins        = 256;      end
    % threshold tuning on top of raw Otsu
    if ~isfield(B,'thrScale'),     B.thrScale     = 1.08;     end
    if ~isfield(B,'thrBias'),      B.thrBias      = 0.02;     end
    % pre-denoising
    if ~isfield(B,'preDenoise'),   B.preDenoise   = 'median'; end  % 'median' | 'gaussian' | 'none'
    if ~isfield(B,'preWin'),       B.preWin       = 3;        end
    if ~isfield(B,'preSigma'),     B.preSigma     = 0.6;      end
    % morphology (lightweight defaults)
    if ~isfield(B,'openRadius'),   B.openRadius   = 1;        end
    if ~isfield(B,'closeRadius'),  B.closeRadius  = 0;        end
    if ~isfield(B,'fillHoles'),    B.fillHoles    = false;    end
    if ~isfield(B,'postMinArea'),  B.postMinArea  = [];       end % auto if empty
    if ~isfield(B,'erodeRadius'),  B.erodeRadius  = 0;        end
    % visualization
    if ~isfield(B,'visualize'),    B.visualize    = true;     end

    %% ---- Pre-denoise (optional) ----
    I_pre = I;
    switch lower(B.preDenoise)
        case 'median'
            w = max(1, round(B.preWin)); if mod(w,2)==0, w=w+1; end
            I_pre = medfilt2(I, [w w]);
        case 'gaussian'
            w = max(1, round(B.preWin)); if mod(w,2)==0, w=w+1; end
            g = local_gauss_kernel(w, B.preSigma);
            I_pre = conv2(I, g, 'same');
        otherwise % 'none'
            % no-op
    end

    %% ---- Otsu + tuning ----
    thrRaw = otsu_threshold(I_pre, B.nbins);
    thr    = min(max(thrRaw * B.thrScale + B.thrBias, 0), 1);

    %% ---- Apply polarity ----
    switch lower(B.polarity)
        case 'bright', BW = I_pre >= thr;
        case 'dark',   BW = I_pre <= thr;
        otherwise, error('Unsupported cfg.bin.polarity: %s', B.polarity);
    end

    %% ---- Post-processing ----
    % auto min area ~0.05% of ROI if not specified
    if isempty(B.postMinArea)
        roiArea = numel(BW);
        B.postMinArea = max(10, round(0.0005 * roiArea));
    end

    if B.postMinArea > 0
        BW = bwareaopen(BW, B.postMinArea, 8);
    end
    if B.openRadius > 0
        BW = imopen(BW, strel('disk', B.openRadius));
    end
    if B.closeRadius > 0
        BW = imclose(BW, strel('disk', B.closeRadius));
    end
    if B.erodeRadius > 0
        BW = imerode(BW, strel('disk', B.erodeRadius)); % use when you need thinner strokes
    end
    if B.fillHoles
        BW = imfill(BW, 'holes');
    end

    %% ---- Write state ----
    state.binarize = struct();
    state.binarize.bw   = BW;
    state.binarize.thr  = thr;
    B.otsuRaw           = thrRaw;
    state.binarize.info = B;

    %% ---- Visualize & export (optional) ----
    if B.visualize
        f = figure('Name','Step4 - Binarize (tuned)','Color','w');
        tiledlayout(1,2,'TileSpacing','compact','Padding','compact');
        nexttile; imshow(I,  'Border','tight'); title(sprintf('ROI (thr=%.4f)',thr));
        nexttile; imshow(BW, 'Border','tight'); title(sprintf('Binary (%s)', upper(B.polarity)));
        exportgraphics(f, fullfile(cfg.paths.figures,'step4_binarize.png'), 'Resolution',150);
        close(f);
    end

    fprintf('[Step4] Otsu thrRaw=%.4f -> thr=%.4f | pol=%s | postMinArea=%d | open=%d | close=%d | erode=%d | visualize=%d\n', ...
        thrRaw, thr, lower(B.polarity), B.postMinArea, B.openRadius, B.closeRadius, B.erodeRadius, B.visualize);
end

%% ----------------- helpers -----------------
function thr = otsu_threshold(I, nbins)
    I = ensure_double01(I);
    edges  = linspace(0,1,nbins+1);
    counts = histcounts(I(:), edges);
    p  = counts / max(1,sum(counts));
    w  = cumsum(p);
    mu = cumsum(p .* (0:nbins-1));
    muT = mu(end);
    denom = w.*(1-w); denom(denom==0) = eps;
    sigma_b2 = (muT*w - mu).^2 ./ denom;
    [~, idx] = max(sigma_b2);
    thr = (idx-1 + 0.5) / nbins;
end

function J = ensure_double01(I)
    I = double(I);
    mn = min(I(:)); mx = max(I(:));
    if mx > mn, J = (I - mn) / (mx - mn);
    else,       J = zeros(size(I), 'like', I);
    end
end

function g = local_gauss_kernel(w, sigma)
    if nargin<2 || isempty(sigma), sigma = 0.6; end
    r = (-(w-1)/2:(w-1)/2);
    g1 = exp(-(r.^2)/(2*sigma^2)); g1 = g1/sum(g1);
    g  = g1' * g1; g = g / sum(g(:));
end
