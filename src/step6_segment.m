function state = step6_segment(state, cfg)
% STEP6_SEGMENT - Toolbox-free character segmentation
% - Connected components with size/aspect filters
% - Optional conservative merge
% - Optional splitting of over-wide boxes via vertical projection valleys
%
% Inputs:
%   state:
%     state.roi         : grayscale ROI (double in [0,1])
%     state.binarize.bw : logical ROI binary (foreground = true)
%   cfg.segment (optional):
%     .connectivity : '8' (default) | '4'
%     .preCloseIters: integer >=0, 3x3 closing iters before CC (default 0)
%     .preErodeIters: integer >=0, 3x3 erosion iters before CC (default 1)
%     .minAreaFrac  : min area vs ROI area (default 7e-5)
%     .maxAreaFrac  : max area vs ROI area (default 0.12)
%     .minHFrac     : min height vs ROI height (default 0.22)
%     .maxHFrac     : max height vs ROI height (default 0.95)
%     .minWHRatio   : min width/height (default 0.08)
%     .maxWHRatio   : max width/height (default 3.0)
%     .mergeGaps    : true/false (default false)
%     .xGapPx       : max horizontal gap (px) to merge (default 3)
%     .yOverlapFrac : required vertical overlap [0..1] (default 0.75)
%     .gridCols     : preview grid columns (default 12)
%     .splitWide         : true/false (default true)
%     .splitWidthFactor  : trigger split if w > factor * median_w (default 1.35)
%     .splitMinPartFrac  : each child >= frac * median_w (default 0.45)
%     .splitPadPx        : ignore cut near box edges (px, default 3)
%     .targetK           : desired total boxes; 0/[] to disable (default 10)
%
% Outputs:
%   state.segment.boxes      : Nx4 [x y w h], left→right
%   state.segment.cropsGray  : 1xN gray crops
%   state.segment.cropsBW    : 1xN binary crops (original polarity)
%   state.segment.cropsBin   : 1xN binary crops (inverted: white background, black glyph)
%   state.segment.bwFiltered : final mask keeping only content within boxes
%
% Saves:
%   step6_segment_boxes.png, step6_chars_grid.png, step6_final_bw.png

    assert(isfield(state,'binarize') && isfield(state.binarize,'bw'), ...
        'Missing state.binarize.bw (run step4 first).');
    assert(isfield(state,'roi'), 'Missing state.roi (from step3).');

    I  = state.roi;
    BW = logical(state.binarize.bw);
    [H,W] = size(BW);
    roiArea = H*W;

    % ---------- defaults ----------
    if ~isfield(cfg,'segment'), cfg.segment = struct(); end
    S = cfg.segment;
    S = set_default(S,'connectivity','8');
    S = set_default(S,'preCloseIters',0);
    S = set_default(S,'preErodeIters',1);
    S = set_default(S,'minAreaFrac',7e-5);
    S = set_default(S,'maxAreaFrac',0.12);
    S = set_default(S,'minHFrac',0.22);
    S = set_default(S,'maxHFrac',0.95);
    S = set_default(S,'minWHRatio',0.08);
    S = set_default(S,'maxWHRatio',3.0);
    S = set_default(S,'mergeGaps',false);
    S = set_default(S,'xGapPx',3);
    S = set_default(S,'yOverlapFrac',0.75);
    S = set_default(S,'gridCols',12);
    % splitting
    S = set_default(S,'splitWide',true);
    S = set_default(S,'splitWidthFactor',1.35);
    S = set_default(S,'splitMinPartFrac',0.45);
    S = set_default(S,'splitPadPx',3);
    S = set_default(S,'targetK',10);

    % ---------- light morphology (toolbox-free) ----------
    if S.preCloseIters > 0
        BW = morph_close(BW, S.preCloseIters);   % 3x3: dilate then erode
    end
    if S.preErodeIters > 0
        BW = morph_erode(BW, S.preErodeIters);   % 3x3 erosion
    end

    % ---------- connected components (BFS) ----------
    if strcmpi(S.connectivity,'4')
        neigh = [ -1 0; 1 0; 0 -1; 0 1 ];
    else
        neigh = [ -1 -1; -1 0; -1 1; 0 -1; 0 1; 1 -1; 1 0; 1 1 ];
    end
    [labels, stats] = label_cc(BW, neigh); %#ok<ASGLU>

    % ---------- filter by size / aspect / height ----------
    minA = max(5, round(S.minAreaFrac * roiArea));
    maxA = max(minA+1, round(S.maxAreaFrac * roiArea));
    minH = max(1, round(S.minHFrac * H));
    maxH = max(minH, round(S.maxHFrac * H));

    boxes = zeros(0,4);
    for k = 1:numel(stats)
        bb = stats(k).bbox; a = stats(k).area;
        r = bb(3) / max(1, bb(4));   % w/h
        if a>=minA && a<=maxA && bb(4)>=minH && bb(4)<=maxH && ...
           r>=S.minWHRatio && r<=S.maxWHRatio
            boxes(end+1,:) = bb; %#ok<AGROW>
        end
    end
    if isempty(boxes)
        state.segment = struct('boxes',[],'cropsGray',{{}},'cropsBW',{{}},'cropsBin',{{}},'bwFiltered',[]);
        warn_and_dump_empty(cfg.paths.figures, I, BW);
        return;
    end

    % ---------- split over-wide boxes ----------
    if S.splitWide
        boxes = split_overwide_boxes(BW, boxes, S);
    end

    % ---------- conservative merge (optional) ----------
    if S.mergeGaps
        boxes = merge_horizontal_boxes(boxes, S.xGapPx, S.yOverlapFrac);
    end

    % ---------- reach target K by splitting widest boxes ----------
    if ~isempty(S.targetK) && S.targetK > 0
        boxes = reach_targetK_by_splitting(BW, boxes, S.targetK, S);
    end

    % ---------- left→right & crops ----------
    [~,ord] = sort(boxes(:,1),'ascend'); boxes = boxes(ord,:);
    N = size(boxes,1);

    % final mask keeping only pixels inside boxes
    BWfinal = false(H,W);
    for i = 1:N
        b = boxes(i,:);
        sub = BW(b(2):b(2)+b(4)-1, b(1):b(1)+b(3)-1);
        BWfinal(b(2):b(2)+b(4)-1, b(1):b(1)+b(3)-1) = sub | BWfinal(b(2):b(2)+b(4)-1, b(1):b(1)+b(3)-1);
    end

    % crops: original polarity and inverted (white background, black glyph)
    cropsBW   = cell(1,N);
    cropsBin  = cell(1,N);
    cropsGray = cell(1,N);
    for i = 1:N
        b = boxes(i,:);
        bin = BW(b(2):b(2)+b(4)-1, b(1):b(1)+b(3)-1);
        cropsBW{i}   = bin;
        cropsBin{i}  = ~bin;
        cropsGray{i} = I( b(2):b(2)+b(4)-1, b(1):b(1)+b(3)-1 );
    end

    % ---------- save to state ----------
    state.segment = struct( ...
        'boxes',      boxes, ...
        'cropsGray',  {cropsGray}, ...
        'cropsBW',    {cropsBW}, ...
        'cropsBin',   {cropsBin}, ...
        'bwFiltered',  BWfinal);

    % ---------- visualization ----------
    f1 = figure('Name','Step6 - Boxes','Color','w');
    imshow(I,'Border','tight'); hold on;
    for i=1:N
        b=boxes(i,:);
        rectangle('Position',b,'EdgeColor',[0 1 0],'LineWidth',1.5);
        text(b(1), max(1,b(2)-4), sprintf('%d',i), 'Color','y','FontSize',10,'FontWeight','bold');
    end
    safe_save_fig(f1, fullfile(cfg.paths.figures,'step6_segment_boxes.png'), 150);
    close(f1);

    cols = max(1,S.gridCols); rows = ceil(N/cols);
    f2 = figure('Name','Step6 - Crops Grid','Color','w');
    tiledlayout(rows, cols, 'TileSpacing','compact','Padding','compact');
    for i=1:N
        nexttile; imshow(cropsBW{i},'Border','tight'); title(sprintf('#%d',i),'FontSize',9);
    end
    safe_save_fig(f2, fullfile(cfg.paths.figures,'step6_chars_grid.png'), 150);
    close(f2);

    f3 = figure('Name','Step6 - Final BW','Color','w');
    imshow(BWfinal,'Border','tight'); title('Step6 Final BW (kept boxes only)');
    safe_save_fig(f3, fullfile(cfg.paths.figures,'step6_final_bw.png'), 150);
    close(f3);

    fprintf('[Step6] Done. %d segments kept.\n', N);

    % export per-character inverted crops (white bg / black glyph)
    outDir = fullfile(cfg.paths.figures, 'step6_crops_bin');
    if ~exist(outDir,'dir'), mkdir(outDir); end
    for i=1:N
        imwrite(uint8(cropsBin{i}*255), fullfile(outDir, sprintf('char_%02d.png', i)));
    end
end

% ================= helpers =================
function S = set_default(S, f, v)
    if ~isfield(S, f) || isempty(S.(f)), S.(f) = v; end
end

function B = morph_close(B, iters)
% 3x3 closing: dilate then erode (toolbox-free)
    B = logical(B);
    for t = 1:max(1, round(iters))
        D = conv2(double(B), ones(3), 'same') > 0;
        B = conv2(double(D), ones(3), 'same') == 9;
    end
    B = logical(B);
end

function B = morph_erode(B, iters)
% 3x3 erosion (toolbox-free)
    B = logical(B);
    for t = 1:max(1, round(iters))
        B = conv2(double(B), ones(3), 'same') == 9;
    end
    B = logical(B);
end

function [labels, stats] = label_cc(BW, neigh)
% BFS connected components; returns labels and stats(area, bbox)
    [H,W] = size(BW);
    labels = zeros(H,W,'uint32');
    visited = false(H,W);
    lab = uint32(0);
    stats = struct('area',{},'bbox',{});
    for y = 1:H
        for x = 1:W
            if ~BW(y,x) || visited(y,x), continue; end
            lab = lab + 1;
            [area, bb, visited, labels] = bfs_component(BW, visited, labels, y, x, lab, neigh);
            stats(end+1).area = area; %#ok<AGROW>
            stats(end).bbox = bb;
        end
    end
end

function [area, bbox, visited, labels] = bfs_component(BW, visited, labels, sy, sx, lab, neigh)
    [H,W] = size(BW);
    qy = zeros(H*W,1,'uint32'); qx = zeros(H*W,1,'uint32');
    head = 1; tail = 1; qy(1)=uint32(sy); qx(1)=uint32(sx);
    area = 0; minx=W; maxx=1; miny=H; maxy=1;
    while head <= tail
        y = qy(head); x = qx(head); head = head + 1;
        if visited(y,x) || ~BW(y,x), continue; end
        visited(y,x) = true; labels(y,x) = lab; area = area + 1;
        if x<minx, minx=double(x); end
        if x>maxx, maxx=double(x); end
        if y<miny, miny=double(y); end
        if y>maxy, maxy=double(y); end
        for k = 1:size(neigh,1)
            ny = int32(y) + neigh(k,1);
            nx = int32(x) + neigh(k,2);
            if ny>=1 && ny<=H && nx>=1 && nx<=W
                if ~visited(ny,nx) && BW(ny,nx)
                    tail = tail + 1; qy(tail)=uint32(ny); qx(tail)=uint32(nx);
                end
            end
        end
    end
    bbox = [minx, miny, maxx-minx+1, maxy-miny+1];
end

function boxesOut = merge_horizontal_boxes(B, gapPx, yOverlapFrac)
    if isempty(B), boxesOut = B; return; end
    [~,idx] = sort(B(:,1),'ascend'); B = B(idx,:);
    out = []; cur = B(1,:);
    for i = 2:size(B,1)
        nxt = B(i,:);
        gap = nxt(1) - (cur(1)+cur(3));
        y1 = max(cur(2),nxt(2)); y2 = min(cur(2)+cur(4),nxt(2)+nxt(4));
        vOverlap = max(0,y2-y1) / max(cur(4),nxt(4));
        if gap <= gapPx && vOverlap >= yOverlapFrac
            x1=min(cur(1),nxt(1)); y1=min(cur(2),nxt(2));
            x2=max(cur(1)+cur(3),nxt(1)+nxt(3)); y2=max(cur(2)+cur(4),nxt(2)+nxt(4));
            cur=[x1,y1,x2-x1,y2-y1];
        else
            out=[out;cur]; cur=nxt;
        end
    end
    out=[out;cur]; boxesOut = out;
end

function warn_and_dump_empty(figDir, I, BW)
    f = figure('Name','Step6 - Empty','Color','w');
    tiledlayout(1,2,'TileSpacing','compact','Padding','compact');
    nexttile; imshow(I,'Border','tight');  title('ROI');
    nexttile; imshow(BW,'Border','tight'); title('Binary');
    safe_save_fig(f, fullfile(figDir,'step6_empty.png'), 150);
    close(f);
end

function safe_save_fig(figHandle, outPath, dpi)
    if nargin<3 || isempty(dpi), dpi = 150; end
    if ~ishandle(figHandle) || ~strcmp(get(figHandle,'Type'),'figure'), figHandle = gcf; end
    drawnow;
    try
        if exist('exportgraphics','file')==2
            exportgraphics(figHandle, outPath, 'Resolution', dpi); return;
        end
    catch, end
    try
        set(figHandle,'PaperPositionMode','auto');
        print(figHandle, outPath, '-dpng', ['-r' num2str(dpi)]); return;
    catch, end
    try
        fr = getframe(figHandle); imwrite(fr.cdata, outPath);
    catch ME
        warning('Failed to save figure to %s: %s', outPath, ME.message);
    end
end

% ------------ split helpers ------------
function boxes = split_overwide_boxes(BW, boxes, S)
% Split boxes with width >> median width using vertical projection valleys
    if isempty(boxes), return; end
    wMed = median(boxes(:,3));
    if isempty(wMed) || wMed<=0, return; end

    pad = max(1, round(S.splitPadPx));
    minChildW = max(6, round(S.splitMinPartFrac * wMed));
    maxW = S.splitWidthFactor * wMed;

    out = [];
    for i = 1:size(boxes,1)
        b = boxes(i,:);
        if b(3) <= maxW
            out = [out; b]; %#ok<AGROW>
            continue;
        end
        parts = split_box_once(BW, b, pad, minChildW);
        if isempty(parts)
            cut = round(b(3)/2);
            if cut >= minChildW && (b(3)-cut) >= minChildW
                left  = [b(1), b(2), cut,        b(4)];
                right = [b(1)+cut, b(2), b(3)-cut, b(4)];
                out = [out; left; right]; %#ok<AGROW>
            else
                out = [out; b]; %#ok<AGROW>
            end
        else
            out = [out; parts]; %#ok<AGROW>
        end
    end
    boxes = out;
end

function parts = split_box_once(BW, b, pad, minChildW)
% Split one box at deepest valley of column foreground sum (toolbox-free)
    parts = [];
    sub = BW(b(2):b(2)+b(4)-1, b(1):b(1)+b(3)-1);
    [~, Ws] = size(sub);
    col = sum(sub, 1);
    win = max(3, round(0.06 * Ws)); % ~6% width
    k = ones(1,win)/win;
    colS = conv(col, k, 'same');

    L = max(2, 1+pad);
    R = min(Ws-1, Ws-pad);
    if L >= R, return; end

    [~, idx] = min(colS(L:R));
    cut = L + idx - 1;

    if (cut < minChildW) || (Ws - cut) < minChildW
        return;
    end
    left  = [b(1),      b(2), cut,        b(4)];
    right = [b(1)+cut,  b(2), b(3)-cut,   b(4)];
    parts = [left; right];
end

function boxes = reach_targetK_by_splitting(BW, boxes, K, S)
% Split the widest box until we reach K or cannot split safely
    iters = 0; maxiters = 50;
    while size(boxes,1) < K && iters < maxiters
        iters = iters + 1;
        [~, idx] = max(boxes(:,3));
        b = boxes(idx,:);
        parts = split_box_once(BW, b, max(1,round(S.splitPadPx)), ...
                                   max(6,round(S.splitMinPartFrac*median(boxes(:,3)))));
        if isempty(parts)
            cut = round(b(3)/2);
            if cut >= 6 && (b(3)-cut) >= 6
                parts = [ b(1),      b(2), cut,        b(4);
                          b(1)+cut,  b(2), b(3)-cut,   b(4) ];
            else
                break;
            end
        end
        boxes(idx,:) = [];
        boxes = [boxes; parts];
    end
end
