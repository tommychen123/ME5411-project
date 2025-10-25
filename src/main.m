% MAIN ENTRY FOR ME5411
clear; clc;

%% ---------------- Config ----------------
cfg = struct();
cfg.seed = 5411;

% paths
cfg.paths = struct();
cfg.paths.data     = fullfile('..','data');
cfg.paths.results  = fullfile('..','results');
cfg.paths.figures  = fullfile(cfg.paths.results, 'figures');
cfg.paths.models   = fullfile(cfg.paths.results, 'models');
cfg.paths.logs     = fullfile(cfg.paths.results, 'logs');

% data file
cfg.file = struct();
cfg.file.image = 'charact2.bmp';
cfg.file.datasetRoot = fullfile(cfg.paths.data,'dataset_2025');
cfg.file.datasetZip  = fullfile(cfg.paths.data,'dataset_2025.zip');

% switches
cfg.flags = struct();
cfg.flags.runStep1 = true;
cfg.flags.runStep2 = true;
cfg.flags.runStep3 = true;
cfg.flags.runStep4 = true;
cfg.flags.runStep5 = true;
cfg.flags.runStep6 = true;
cfg.flags.runStep7 = true;
cfg.flags.trainCNN = false;        % 训练一次后可改为 false 复用缓存
cfg.flags.applyCNN = true;
cfg.flags.resumeModels = true;

% ensure dirs
ensure_dir(cfg.paths.results);
ensure_dir(cfg.paths.figures);
ensure_dir(cfg.paths.models);
ensure_dir(cfg.paths.logs);

% simple log
logfile = fullfile(cfg.paths.logs, [datestr(now,'yyyymmdd_HHMMSS') '_run.log']);
ensure_dir(fileparts(logfile)); logline(logfile, '=== ME5411 pipeline started ===');
rng(cfg.seed);

%% ---------------- Step 1 ----------------
state = struct();
if cfg.flags.runStep1
    t=tic;
    state = step1_display(state, cfg);
    state = step1_enhancement(state, cfg);
    state.timing.step1 = toc(t);
    logline(logfile, sprintf('[Step1] done in %.3fs', state.timing.step1));
end

%% ---------------- Step 2 ----------------
if cfg.flags.runStep2
    t=tic;
    cfg.filter = struct('winSizes',5,'padMode','replicate');
    state = step2_filter(state, cfg);
    state.timing.step2 = toc(t);
    logline(logfile, sprintf('[Step2] %.3fs', state.timing.step2));
end

%% ---------------- Step 3 ----------------
if cfg.flags.runStep3
    t=tic;
    cfg.roi = struct('bbox',[20 200 950 150],'margin',0);
    state = step3_roi(state, cfg);
    state.timing.step3 = toc(t);
    logline(logfile, sprintf('[Step3] %.3fs', state.timing.step3));
end

%% ---------------- Step 4 ----------------
if cfg.flags.runStep4
    t=tic;
    cfg.bin = struct('polarity','bright','nbins',256);
    state = step4_binarize(state, cfg);
    state.timing.step4 = toc(t);
    logline(logfile, sprintf('[Step4] %.3fs', state.timing.step4));
end

%% ---------------- Step 5 ----------------
if cfg.flags.runStep5
    t=tic;
    cfg.outline = struct('minLength',15,'connectivity','8');
    state = step5_outline(state, cfg);
    state.timing.step5 = toc(t);
    logline(logfile, sprintf('[Step5] %.3fs', state.timing.step5));
end

%% ---------------- Step 6 ----------------
if cfg.flags.runStep6
    t=tic;
    cfg.segment = struct( ...
        'connectivity','8','preCloseIters',0,'preErodeIters',1, ...
        'minAreaFrac',7e-5,'maxAreaFrac',0.12, ...
        'minHFrac',0.22,'maxHFrac',0.95, ...
        'minWHRatio',0.08,'maxWHRatio',3.0, ...
        'mergeGaps',false,'xGapPx',3,'yOverlapFrac',0.75, ...
        'gridCols',12,'splitWide',true,'targetK',10, ...
        'splitWidthFactor',1.35,'splitMinPartFrac',0.45);
    state = step6_segment(state, cfg);
    state.timing.step6 = toc(t);
    logline(logfile, sprintf('[Step6] %.3fs', state.timing.step6));
end

%% ---------------- Step 7: CNN (use your cnn.m) ----------------
if cfg.flags.runStep7
    % 7b) CNN 训练或加载（由你的 cnn.m 决定，模型保存到 ../results/models/CNN_latest.mat）
    t = tic;
    state = step7_task1_cnn(state, cfg);     % <- 用上面给你的“适配器”
    state.timing.step7_cnn = toc(t);
    logline(logfile, sprintf('[Step7/CNN] ready in %.3fs (ValAcc=%.3f)', ...
        state.timing.step7_cnn, state.step7.cnn.valAcc));

    % 7c) Apply CNN
    if cfg.flags.applyCNN
        state = step7_task1_apply_cnn(state, cfg);
    end

    if 0
        state = step7_task2_mlp(state, cfg);
    end
    state = step7_task2_apply_mlp(state, cfg);

    % 如不再需要 SVM，对 7d/7e/7f 可直接注释掉
end


%% ---------------- Save final state ----------------
save(fullfile(cfg.paths.results,'run_state.mat'), 'state','cfg');
disp('All steps finished. Check ../results and ../results/figures .');

%% -------------- Helpers --------------
function ensure_dir(p), if ~exist(p, 'dir'), mkdir(p); end, end
function logline(f, s)
    fprintf('%s\n', s);
    try
        fid = fopen(f,'a'); fprintf(fid,'%s %s\n', datestr(now,'yyyy-mm-dd HH:MM:SS'), s); fclose(fid);
    catch, end
end
