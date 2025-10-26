function state = step5_outline(state, cfg)
% STEP5_OUTLINE (fixed) - perimeter via convolution neighbor count (toolbox-free)
% Inputs:  state.roi (gray), state.binarize.bw (logical)
% Outputs: state.outline.mask (logical, same size as ROI)
% Saves:   step5_outline_mask.png, step5_outline_overlay.png

    assert(isfield(state,'binarize') && isfield(state.binarize,'bw'), ...
        'Run step4_binarize first.');
    BW = logical(state.binarize.bw);
    if isfield(state,'roi'), I = state.roi; else, I = double(BW); end

    if ~isfield(cfg,'outline'), cfg.outline = struct(); end
    if ~isfield(cfg.outline,'connectivity'), cfg.outline.connectivity = '8'; end

    if strcmpi(cfg.outline.connectivity,'4')
        K = [0 1 0; 1 1 1; 0 1 0];
        N = conv2(double(BW), K, 'same');
        perim = BW & (N < 5); 
    else
        N = conv2(double(BW), ones(3), 'same');
        perim = BW & (N < 9); 
    end

    state.outline = struct();
    state.outline.mask = perim;

    % ---- save figures ----
    f1 = figure('Name','Step5 - Outline Mask','Color','w');
    imshow(perim, 'Border','tight'); title('Perimeter mask');
    exportgraphics(f1, fullfile(cfg.paths.figures,'step5_outline_mask.png'), 'Resolution',150);
    close(f1);

    f2 = figure('Name','Step5 - Overlay','Color','w');
    imshow(I, 'Border','tight'); hold on;
    [yy,xx] = find(perim);
    if ~isempty(xx)
        plot(xx, yy, '.', 'Color', [1 0 0], 'MarkerSize', 1);
    end
    title('Outline overlay on ROI');
    exportgraphics(f2, fullfile(cfg.paths.figures,'step5_outline_overlay.png'), 'Resolution',150);
    close(f2);
end
