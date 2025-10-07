% STEP1_DISPLAY
% Purpose:
%   - Load the original BMP image
%   - Convert to grayscale (if RGB)
%   - Display and save a snapshot for reporting
%
% Inputs:
%   state : struct carrying pipeline context
%   cfg   : struct with configuration (paths, file names)
%
% Outputs (added to 'state'):
%   state.imgGray   : grayscale double image in [0,1]
%   state.meta.size : [H, W] image size
%   Saved figures:
%     results/figures/step1_original.png
%
% Notes:
%   - Avoids toolbox-only functions; uses built-in MATLAB I/O.

function state = step1_display(state, cfg)

    % ---------- Load image ----------
    imgPath = fullfile(cfg.paths.data, cfg.file.image);
    assert(exist(imgPath, 'file') == 2, ...
        'Image file not found: %s', imgPath);

    img = imread(imgPath);

    % ---------- Convert to grayscale double in [0,1] ----------
    if ndims(img) == 3
        % manual RGB to gray (avoid toolbox dependency)
        img = 0.2989 * double(img(:,:,1)) + ...
              0.5870 * double(img(:,:,2)) + ...
              0.1140 * double(img(:,:,3));
    else
        img = double(img);
    end
    % scale to [0,1] safely
    minv = min(img(:));
    maxv = max(img(:));
    if maxv > minv
        imgGray = (img - minv) / (maxv - minv);
    else
        imgGray = zeros(size(img), 'like', img);
    end

    % ---------- Save to state ----------
    state.imgGray = imgGray;
    state.meta = struct();
    state.meta.size = [size(imgGray,1), size(imgGray,2)];

    % ---------- Display & save ----------
    f = figure('Name','Step1 - Original','Color','w');
    imshow(imgGray, 'Border', 'tight');
    title(sprintf('Original (H=%d, W=%d)', ...
        state.meta.size(1), state.meta.size(2)));
    drawnow;

    outFig = fullfile(cfg.paths.figures, 'step1_original.png');
    exportgraphics(f, outFig, 'Resolution', 150);
    close(f);
end
