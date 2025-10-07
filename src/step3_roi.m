function state = step3_roi(state, cfg)
% STEP3_ROI - Fixed-bbox cropping (manual ROI).
% Goal:
%   Crop the grayscale ROI that contains the text line using a fixed bbox.
%
% Inputs:
%   state : expects one of:
%           state.imgFilt.default (preferred) OR state.imgEnh.minmax OR state.imgGray
%   cfg.roi :
%       .bbox   : [x y w h] (required) - the rectangle to crop
%       .margin : integer pixels to expand the final box (default 0)
%
% Outputs (added to 'state'):
%   state.roi     : cropped ROI (double in [0,1])
%   state.roiBBox : [x y w h] after applying margin & clamping
%
% Saved files (under cfg.paths.figures):
%   step3_roi_bbox.png  : overlay of the box on the source image
%   step3_roi.png       : the cropped ROI image

    % ---------- choose a source grayscale image ----------
    if isfield(state,'imgFilt')
        I = state.imgFilt.default;
    elseif isfield(state,'imgEnh') && isfield(state.imgEnh,'minmax')
        I = state.imgEnh.minmax;
    else
        I = state.imgGray;
    end
    I = ensure_double01(I);
    [H,W] = size(I);

    % ---------- config ----------
    assert(isfield(cfg,'roi') && isfield(cfg.roi,'bbox') && numel(cfg.roi.bbox)==4, ...
        'cfg.roi.bbox must be provided as [x y w h].');
    if ~isfield(cfg.roi,'margin'), cfg.roi.margin = 0; end

    % ---------- apply margin & clamp ----------
    bb = round(cfg.roi.bbox);
    m  = round(cfg.roi.margin);
    bbox = [bb(1)-m, bb(2)-m, bb(3)+2*m, bb(4)+2*m];
    bbox = clamp_bbox(bbox, W, H);

    % ---------- crop ----------
    roi = I(bbox(2):bbox(2)+bbox(4)-1, bbox(1):bbox(1)+bbox(3)-1);

    % ---------- save to state ----------
    state.roi     = roi;
    state.roiBBox = bbox;

    % ---------- save figures ----------
    f1 = figure('Name','Step3 - ROI Box','Color','w');
    imshow(I, 'Border','tight'); hold on;
    rectangle('Position', bbox, 'EdgeColor','r', 'LineWidth', 2);
    title(sprintf('ROI bbox = [x=%d, y=%d, w=%d, h=%d]', bbox));
    exportgraphics(f1, fullfile(cfg.paths.figures,'step3_roi_bbox.png'), 'Resolution',150);
    close(f1);

    imwrite(roi, fullfile(cfg.paths.figures,'step3_roi.png'));
end

% ----------------- helpers -----------------
function J = ensure_double01(I)
    I = double(I);
    mn = min(I(:)); mx = max(I(:));
    if mx > mn, J = (I - mn) / (mx - mn);
    else,       J = zeros(size(I), 'like', I);
    end
end

function bbox = clamp_bbox(b, W, H)
% Clamp [x y w h] to image bounds (1..W, 1..H).
    x = max(1, round(b(1)));
    y = max(1, round(b(2)));
    w = max(1, round(b(3)));
    h = max(1, round(b(4)));
    if x + w - 1 > W, w = W - x + 1; end
    if y + h - 1 > H, h = H - y + 1; end
    bbox = [x y w h];
end
