function state = step7_task2_apply_noncnn(state, cfg)
% Apply HOG+SVM to segmented chars; produce overlay/grid/CSV and timing.

    assert(isfield(state,'step7') && isfield(state.step7,'hogsvm'), 'Train HOG+SVM first.');
    assert(isfield(state,'segment') && ~isempty(state.segment.cropsGray), 'Run step6 first.');

    model=state.step7.hogsvm.model; cellSize=state.step7.hogsvm.cellSize;
    inputSize=state.step7.inputSize; crops=state.segment.cropsGray; boxes=state.segment.boxes; Iroi=state.roi;

    Ht=inputSize(1); Wt=inputSize(2); N=numel(crops);
    X=zeros(N, numel(extractHOGFeatures(zeros(Ht,Wt,'single'),'CellSize',cellSize)), 'single');

    for i=1:N
        I=crops{i}; if size(I,3)>1, I=rgb2gray(I); end
        I=im2single(imresize(I,[Ht Wt]));                         % 可换成 letterbox
        X(i,:)=single(extractHOGFeatures(I,'CellSize',cellSize));
    end

    t0=tic; [Ypred, ~]=predict(model,X); inferTime=toc(t0);
    labels=cellstr(Ypred);

    % overlay
    f1=figure('Name','Task2 - HOG+SVM Overlay','Color','w'); imshow(Iroi,'Border','tight'); hold on;
    for i=1:size(boxes,1)
        b=boxes(i,:); rectangle('Position',b,'EdgeColor',[0 1 0],'LineWidth',1.5);
        text(b(1),max(1,b(2)-6),labels{i},'Color',[1 1 0],'BackgroundColor',[0 0 0],...
             'Margin',2,'FontSize',12,'FontWeight','bold');
    end
    safe_save_fig(f1, fullfile(cfg.paths.figures,'step7_task2_hogsvm_overlay.png'),150); close(f1);

    % grid
    cols=12; rows=ceil(N/cols);
    f2=figure('Name','Task2 - HOG+SVM Crops','Color','w'); tiledlayout(rows,cols,'TileSpacing','compact','Padding','compact');
    for i=1:N, nexttile; imshow(crops{i},'Border','tight'); title(sprintf('#%d %s',i,labels{i}),'FontSize',9); end
    safe_save_fig(f2, fullfile(cfg.paths.figures,'step7_task2_hogsvm_grid.png'),150); close(f2);

    % exports
    if ~exist(cfg.paths.results,'dir'), mkdir(cfg.paths.results); end
    T=table((1:N).',labels(:),'VariableNames',{'Index','Label'});
    writetable(T, fullfile(cfg.paths.results,'step7_task2_hogsvm_preds.csv'));
    fid=fopen(fullfile(cfg.paths.results,'step7_task2_hogsvm_text.txt'),'w'); fprintf(fid,'%s\n',strjoin(labels,'')); fclose(fid);

    state.step7.task2 = struct('labels',{labels},'text',strjoin(labels,''),'inferTimeSec',inferTime);
end

function safe_save_fig(figHandle,outPath,dpi)
    if nargin<3||isempty(dpi), dpi=150; end
    if ~ishandle(figHandle)||~strcmp(get(figHandle,'Type'),'figure'), figHandle=gcf; end
    drawnow;
    try, if exist('exportgraphics','file')==2, exportgraphics(figHandle,outPath,'Resolution',dpi); return; end, catch, end
    try, set(figHandle,'PaperPositionMode','auto'); print(figHandle,outPath,'-dpng',['-r' num2str(dpi)]); return; catch, end
    try, fr=getframe(figHandle); imwrite(fr.cdata,outPath); catch, end
end
