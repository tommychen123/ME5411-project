function state = step7_task1_apply_cnn(state, cfg)
% Robust CNN application with letterbox + test-time augmentations (TTA)
% Saves:
%   figures/step7_task1_cnn_overlay.png
%   figures/step7_task1_cnn_grid.png
%   results/step7_task1_cnn_preds.csv
%   results/step7_task1_cnn_text.txt / _text_marked.txt

    assert(isfield(state,'step7') && isfield(state.step7,'cnn') && ~isempty(state.step7.cnn.net), ...
        'CNN not found. Run step7_task1_cnn first.');
    assert(isfield(state,'segment') && isfield(state.segment,'cropsGray') && ~isempty(state.segment.cropsGray), ...
        'No crops. Run step6_segment first.');

    if ~isfield(cfg,'task1'), cfg.task1 = struct(); end
    V = cfg.task1;
    V = set_default(V,'confThresh',0.60);
    V = set_default(V,'colorLow',  [1 0.6 0]);     % 橙
    V = set_default(V,'colorHigh', [1 1 0]);       % 黄
    V = set_default(V,'useCLAHE',  true);
    V = set_default(V,'useLetterbox', true);
    V = set_default(V,'tta', true);                % 开/关 TTA

    net       = state.step7.cnn.net;
    inputSize = state.step7.inputSize;  % [H W C]
    crops     = state.segment.cropsGray;
    boxes     = state.segment.boxes;
    Iroi      = state.roi;

    Ht = inputSize(1); Wt = inputSize(2); C = inputSize(3);
    N  = numel(crops);

    labels = cell(N,1); confidence = zeros(N,1); usedAug = strings(N,1);

    for i = 1:N
        I = crops{i};
        if size(I,3)>1, I = rgb2gray(I); end
        I = mat2gray(I);

        % --------- 生成 TTA 变体 ---------
        if V.tta
            variants = make_variants(I, V);
        else
            variants = {I};
        end

        % --------- 预处理到网络输入尺寸 (letterbox + 通道) ---------
        M = numel(variants);
        X = zeros(Ht, Wt, C, M, 'single');
        for m = 1:M
            Im = variants{m};
            Im = preprocess_to_size(Im, [Ht Wt], V.useLetterbox);
            if C==3, Im = repmat(Im,[1 1 3]); else, Im = reshape(Im,[Ht Wt 1]); end
            X(:,:,:,m) = Im;
        end

        % --------- 批量分类，平均 softmax ---------
        [Ytmp, scores] = classify(net, X); %#ok<ASGLU>
        % scores: M x K
        Smean = mean(scores, 1);                 % 1 x K
        [conf, k] = max(Smean, [], 2);
        lbls = net.Layers(end).Classes;          % categories
        labels{i} = char(lbls(k));
        confidence(i) = conf;

        % 记录贡献最大的那个 TTA 变体（谁给的分最高）
        [~, mBest] = max(max(scores,[],2));
        usedAug(i) = string(aug_name(mBest, V.tta));
    end

    % ---- 低置信度标记 ----
    marked = labels;
    for i=1:N
        if confidence(i) < V.confThresh
            marked{i} = [marked{i} '?'];
        end
    end

    % ---- Overlay ----
    f1 = figure('Name','Task1 - CNN Overlay','Color','w');
    imshow(Iroi,'Border','tight'); hold on;
    for i=1:size(boxes,1)
        b = boxes(i,:);
        rectangle('Position',b,'EdgeColor',[0 1 0],'LineWidth',1.6);
        col = V.colorHigh; if confidence(i) < V.confThresh, col = V.colorLow; end
        txt = sprintf('%s (%.2f)', labels{i}, confidence(i));
        if confidence(i) < V.confThresh, txt = [txt ' ?']; end
        text(b(1), max(1,b(2)-6), txt, 'Color', col, ...
             'BackgroundColor',[0 0 0], 'Margin',2, ...
             'FontSize',12,'FontWeight','bold');
    end
    safe_save_fig(f1, fullfile(cfg.paths.figures,'step7_task1_cnn_overlay.png'), 150);
    close(f1);

    % ---- Crops Grid ----
    cols = 12; rows = ceil(N/cols);
    f2 = figure('Name','Task1 - CNN Crops','Color','w');
    tiledlayout(rows, cols, 'TileSpacing','compact','Padding','compact');
    for i=1:N
        nexttile; imshow(crops{i},'Border','tight');
        col = V.colorHigh; if confidence(i) < V.confThresh, col = V.colorLow; end
        t = title(sprintf('#%d %s (%.2f) [%s]', i, labels{i}, confidence(i), usedAug(i)), 'FontSize',9);
        try, t.Color = col; catch, end
    end
    safe_save_fig(f2, fullfile(cfg.paths.figures,'step7_task1_cnn_grid.png'), 150);
    close(f2);

    % ---- Exports ----
    if ~exist(cfg.paths.results,'dir'), mkdir(cfg.paths.results); end
    idx = (1:N).';
    T = table(idx, labels(:), confidence(:), usedAug(:), confidence(:)<V.confThresh, marked(:), ...
        'VariableNames', {'Index','Label','Confidence','BestAug','LowConf','LabelMarked'});
    writetable(T, fullfile(cfg.paths.results,'step7_task1_cnn_preds.csv'));

    textPlain  = strjoin(labels,'');
    textMarked = strjoin(marked,'');
    fid = fopen(fullfile(cfg.paths.results,'step7_task1_cnn_text.txt'),'w');       fprintf(fid,'%s\n',textPlain);  fclose(fid);
    fid = fopen(fullfile(cfg.paths.results,'step7_task1_cnn_text_marked.txt'),'w');fprintf(fid,'%s\n',textMarked); fclose(fid);

    state.step7.task1 = struct('labels',{labels}, 'confidence', confidence, ...
        'labelsMarked',{marked}, 'text', textPlain, 'textMarked', textMarked, ...
        'confThresh', V.confThresh, 'bestAug', usedAug);
end

% ------------- helpers -------------
function Im = preprocess_to_size(I, outHW, useLetterbox)
    Ht = outHW(1); Wt = outHW(2);
    I = im2single(I);
    if useLetterbox
        [h,w] = size(I); s = min(Ht/h, Wt/w);
        newh = max(1, round(h*s)); neww = max(1, round(w*s));
        Ir = imresize(I, [newh neww], 'bilinear');
        Im = zeros(Ht, Wt, 'single');                     % 黑底
        y0 = floor((Ht-newh)/2)+1; x0 = floor((Wt-neww)/2)+1;
        Im(y0:y0+newh-1, x0:x0+neww-1) = Ir;
    else
        Im = imresize(I, [Ht Wt], 'bilinear');
    end
end

function list = make_variants(I, V)
    list = {I}; id = 1; %#ok<NASGU>
    if V.useCLAHE
        try, list{end+1} = adapthisteq(I,'NumTiles',[4 8],'ClipLimit',0.01); end %#ok<AGROW>
    end
    list{end+1} = 1-I;                                                  %#ok<AGROW>
    try, list{end+1} = imsharpen(I,'Radius',1,'Amount',1); catch, end   %#ok<AGROW>
    try, list{end+1} = imgaussfilt(I,1.0); catch, end                   %#ok<AGROW>
    try, list{end+1} = imadjust(I,[],[],0.75); catch, end               %#ok<AGROW>
    try, list{end+1} = imadjust(I,[],[],1.30); catch, end               %#ok<AGROW>
end

function name = aug_name(idx, useTTA)
    if ~useTTA, name = "orig"; return; end
    names = ["orig","clahe","invert","sharpen","gauss","gamma0.75","gamma1.3"];
    idx = max(1,min(idx,numel(names))); name = names(idx);
end

function S = set_default(S,f,v), if ~isfield(S,f), S.(f)=v; end, end

function safe_save_fig(figHandle, outPath, dpi)
    if nargin<3||isempty(dpi), dpi=150; end
    if ~ishandle(figHandle)||~strcmp(get(figHandle,'Type'),'figure'), figHandle=gcf; end
    drawnow;
    try, if exist('exportgraphics','file')==2, exportgraphics(figHandle,outPath,'Resolution',dpi); return; end
    catch, end
    try, set(figHandle,'PaperPositionMode','auto'); print(figHandle,outPath,'-dpng',['-r' num2str(dpi)]); return;
    catch, end
    try, fr=getframe(figHandle); imwrite(fr.cdata,outPath); catch, end
end
