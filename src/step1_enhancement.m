% STEP1_ENHANCEMENT
% Purpose:
%   - Apply simple, library-free contrast enhancement variants
%   - Save comparison figure for the report (original vs enhanced)
%
% Inputs:
%   state : struct containing state.imgGray
%   cfg   : struct with output paths
%
% Outputs (added to 'state'):
%   state.imgEnh.minmax   : min-max stretched image in [0,1]
%   state.imgEnh.gamma08  : gamma-corrected (gamma=0.8)
%   state.imgEnh.gamma12  : gamma-corrected (gamma=1.2)
%   Saved figures:
%     results/figures/step1_enhancement_compare.png
%
% Notes:
%   - Implements basic enhancements without relying on Image Toolbox
%   - You can later swap in CLAHE/adapthisteq if toolbox is allowed


function state = step1_enhancement(state, cfg)

    assert(isfield(state, 'imgGray'), ...
        'state.imgGray is missing. Run step1_display first.');
    I = state.imgGray;

    % ---------- Enhancement A: Min-Max Stretch ----------
    % Stretch around data range; guard against degenerate ranges
    imin = min(I(:));
    imax = max(I(:));
    if imax > imin
        A = (I - imin) / (imax - imin);
    else
        A = I;
    end
    % Optional further linear stretch to [0,1] (already is)

    % ---------- Enhancement B: Gamma (gamma < 1 brightens) ----------
    gamma_bright = 0.8;
    B = I .^ gamma_bright;

    % ---------- Enhancement C: Gamma (gamma > 1 darkens) ----------
    gamma_dark = 1.2;
    C = I .^ gamma_dark;

    % ---------- Save to state ----------
    state.imgEnh = struct();
    state.imgEnh.minmax  = A;
    state.imgEnh.gamma08 = B;
    state.imgEnh.gamma12 = C;

    % ---------- Visualization ----------
    f = figure('Name','Step1 - Enhancement Compare','Color','w');
    tiledlayout(2,2, 'TileSpacing','compact','Padding','compact');

    nexttile; imshow(I, 'Border','tight');      title('Original');
    nexttile; imshow(A, 'Border','tight');      title('Min-Max Stretch');
    nexttile; imshow(B, 'Border','tight');      title('Gamma = 0.8 (Brighten)');
    nexttile; imshow(C, 'Border','tight');      title('Gamma = 1.2 (Darken)');

    drawnow;
    outFig = fullfile(cfg.paths.figures, ...
        'step1_enhancement_compare.png');
    exportgraphics(f, outFig, 'Resolution', 150);
    close(f);
end
