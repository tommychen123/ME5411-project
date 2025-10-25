function state = step4_binarize(state, cfg)
% STEP4_BINARIZE - Robust global Otsu thresholding (self-implemented).
% Tuned defaults to avoid fat blobs & sticking (license-plate style text).

    assert(isfield(state,'roi'), 'state.roi is missing. Run step3_roi first.');
    I0 = state.roi;
    I  = ensure_double01(I0);

    % -------- tuned defaults (可被 cfg.bin.* 覆盖) --------
    if ~isfield(cfg,'bin'), cfg.bin = struct(); end
    B = cfg.bin;
    if ~isfield(B,'polarity'),     B.polarity    = 'bright'; end  % 固定亮字
    if ~isfield(B,'nbins'),        B.nbins       = 256;      end
    if ~isfield(B,'thrScale'),     B.thrScale    = 1.08;     end % 略抬阈值
    if ~isfield(B,'thrBias'),      B.thrBias     = 0.02;     end % 再加微偏置
    if ~isfield(B,'preDenoise'),   B.preDenoise  = 'median'; end % 温和降噪
    if ~isfield(B,'preWin'),       B.preWin      = 3;        end
    if ~isfield(B,'preSigma'),     B.preSigma    = 0.6;      end
    if ~isfield(B,'openRadius'),   B.openRadius  = 1;        end % 轻微开运算
    if ~isfield(B,'closeRadius'),  B.closeRadius = 0;        end % 关闭闭运算，防粘连
    if ~isfield(B,'fillHoles'),    B.fillHoles   = false;    end % 不自动填洞，保轮廓
    if ~isfield(B,'postMinArea'),  B.postMinArea = [];       end % 自适应
    if ~isfield(B,'erodeRadius'),  B.erodeRadius = 0;        end % 可选细化(0=关闭)

    % -------- pre-denoise --------
    I_pre = I;
    switch lower(B.preDenoise)
        case 'median'
            w = max(1, round(B.preWin)); if mod(w,2)==0, w=w+1; end
            I_pre = medfilt2(I, [w w]);
        case 'gaussian'
            w = max(1, round(B.preWin)); if mod(w,2)==0, w=w+1; end
            g = local_gauss_kernel(w, B.preSigma);
            I_pre = conv2(I, g, 'same');
        otherwise
            % 'none' -> no-op
    end

    % -------- Otsu (self) + tuned threshold --------
    thr0 = otsu_threshold(I_pre, B.nbins);
    thr  = min(max(thr0 * B.thrScale + B.thrBias, 0), 1);

    % -------- apply polarity (fixed bright by default) --------
    switch lower(B.polarity)
        case 'bright', BW = I_pre >= thr;
        case 'dark',   BW = I_pre <= thr;
        otherwise, error('Unsupported cfg.bin.polarity: %s', B.polarity);
    end

    % -------- post-processing: 去孤点 + 开/闭 + 可选细化 --------
    if isempty(B.postMinArea)
        roiArea = numel(BW);
        B.postMinArea = max(10, round(0.0005 * roiArea)); % ~0.05% ROI
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
        BW = imerode(BW, strel('disk', B.erodeRadius)); % 需要更瘦时打开
    end
    if B.fillHoles
        BW = imfill(BW, 'holes');
    end

    % -------- write state & visualize --------
    state.binarize = struct();
    state.binarize.bw   = BW;
    state.binarize.thr  = thr;
    B.otsuRaw           = thr0;
    state.binarize.info = B;

    f = figure('Name','Step4 - Binarize (tuned)','Color','w');
    tiledlayout(1,2,'TileSpacing','compact','Padding','compact');
    nexttile; imshow(I,'Border','tight');  title(sprintf('ROI (thr=%.4f)',thr));
    nexttile; imshow(BW,'Border','tight'); title(sprintf('Binary (%s)', upper(B.polarity)));
    exportgraphics(f, fullfile(cfg.paths.figures,'step4_binarize.png'), 'Resolution',150);
    close(f);

    fprintf('[Step4] Otsu thrRaw=%.4f -> thr=%.4f | pol=%s | postMinArea=%d | open=%d | close=%d | erode=%d\n', ...
        thr0, thr, lower(B.polarity), B.postMinArea, B.openRadius, B.closeRadius, B.erodeRadius);
end

% ----------------- helpers -----------------
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
