% MAIN ENTRY FOR ME5411: Steps 1–7 end-to-end (with switches & caching)
% Author: <your name>
% Purpose:
%   - Prepare config and folders
%   - Step1: display + enhancement
%   - Step2: mean filter (adjustable)
%   - Step3: fixed ROI crop (your bbox)
%   - Step4: self-implemented Otsu binarization
%   - Step5: character outlines
%   - Step6: segmentation (CC + optional split-wide to reach ~10 chars)
%   - Step7: dataset split, train/apply CNN and HOG+SVM, and compare
%
% Outputs:
%   - Figures under ../results/figures
%   - Models under  ../results/models
%   - CSV/overlays under ../results
%   - state saved to ../results/run_state.mat

clear; clc;

%% ---------------- Config ----------------
cfg = struct();
cfg.seed = 5411;

% paths (run from /src; data/results are parent siblings)
cfg.paths = struct();
cfg.paths.data     = fullfile('..','data');
cfg.paths.results  = fullfile('..','results');
cfg.paths.figures  = fullfile(cfg.paths.results, 'figures');
cfg.paths.models   = fullfile(cfg.paths.results, 'models');
cfg.paths.logs     = fullfile(cfg.paths.results, 'logs');

% data file
cfg.file = struct();
cfg.file.image = 'charact2.bmp';                  % expected under ../data/
cfg.file.datasetRoot = fullfile(cfg.paths.data,'dataset_2025'); % unzipped root
cfg.file.datasetZip  = fullfile(cfg.paths.data,'dataset_2025.zip'); % optional zip

% switches (turn heavy steps on/off quickly)
cfg.flags = struct();
cfg.flags.runStep1 = true;
cfg.flags.runStep2 = true;
cfg.flags.runStep3 = true;
cfg.flags.runStep4 = true;
cfg.flags.runStep5 = true;
cfg.flags.runStep6 = true;
cfg.flags.runStep7 = true;           % all Task1/2/3 below
cfg.flags.trainCNN = true;           % set false to reuse cached model
cfg.flags.trainSVM = true;           % set false to reuse cached model
cfg.flags.applyCNN = true;
cfg.flags.applySVM = true;
cfg.flags.resumeModels = true;       % load cnn/svm if cache exists

% ensure output dirs
ensure_dir(cfg.paths.results);
ensure_dir(cfg.paths.figures);
ensure_dir(cfg.paths.models);
ensure_dir(cfg.paths.logs);

% simple log file
logfile = fullfile(cfg.paths.logs, datestr(now,'yyyymmdd_HHMMSS'), '_run.log');
ensure_dir(fileparts(logfile)); logline(logfile, '=== ME5411 pipeline started ===');

rng(cfg.seed);

%% ---------------- Step 1 ----------------
state = struct();
if cfg.flags.runStep1
    t=tic;
    try
        state = step1_display(state, cfg);
        state = step1_enhancement(state, cfg);
        state.timing.step1 = toc(t);
        logline(logfile, sprintf('[Step1] done in %.3fs', state.timing.step1));
    catch ME
        logline(logfile, ['[Step1] ERROR: ' ME.message]);
        rethrow(ME);
    end
end

%% ---------------- Step 2 (Mean Filter) ----------------
if cfg.flags.runStep2
    t=tic;
    cfg.filter = struct();
    cfg.filter.winSizes = 5;            % e.g., 5 or [3 5 7]
    cfg.filter.padMode  = 'replicate';  % 'replicate' | 'zero'
    state = step2_filter(state, cfg);
    state.timing.step2 = toc(t); logline(logfile, sprintf('[Step2] %.3fs', state.timing.step2));
end

%% ---------------- Step 3 (Fixed ROI crop) ----------------
if cfg.flags.runStep3
    t=tic;
    cfg.roi = struct();
    cfg.roi.bbox   = [20 200 950 150];  % your [x y w h]
    cfg.roi.margin = 0;                 % keep exact
    state = step3_roi(state, cfg);
    state.timing.step3 = toc(t); logline(logfile, sprintf('[Step3] %.3fs', state.timing.step3));
end

%% ---------------- Step 4 (Binarize - self Otsu) ----------------
if cfg.flags.runStep4
    t=tic;
    cfg.bin = struct();
    cfg.bin.polarity = 'bright';        % 'bright' or 'dark'
    cfg.bin.nbins    = 256;
    state = step4_binarize(state, cfg);
    state.timing.step4 = toc(t); logline(logfile, sprintf('[Step4] %.3fs', state.timing.step4));
end

%% ---------------- Step 5 (Outlines) ----------------
if cfg.flags.runStep5
    t=tic;
    cfg.outline = struct();
    cfg.outline.minLength    = 15;   % 10~30
    cfg.outline.connectivity = '8';
    state = step5_outline(state, cfg);
    state.timing.step5 = toc(t); logline(logfile, sprintf('[Step5] %.3fs', state.timing.step5));
end

%% ---------------- Step 6 (Segmentation) ----------------
if cfg.flags.runStep6
    t=tic;
    cfg.segment = struct();
    cfg.segment.connectivity  = '8';
    cfg.segment.preCloseIters = 0;
    cfg.segment.preErodeIters = 1;
    cfg.segment.minAreaFrac   = 7e-5;
    cfg.segment.maxAreaFrac   = 0.12;
    cfg.segment.minHFrac      = 0.22;
    cfg.segment.maxHFrac      = 0.95;
    cfg.segment.minWHRatio    = 0.08;
    cfg.segment.maxWHRatio    = 3.0;
    cfg.segment.mergeGaps     = false;
    cfg.segment.xGapPx        = 3;
    cfg.segment.yOverlapFrac  = 0.75;
    cfg.segment.gridCols      = 12;

    % splitting large boxes (works well for the '00' issue)
    cfg.segment.splitWide        = true;
    cfg.segment.targetK          = 10;
    cfg.segment.splitWidthFactor = 1.35;
    cfg.segment.splitMinPartFrac = 0.45;

    state = step6_segment(state, cfg);
    state.timing.step6 = toc(t); logline(logfile, sprintf('[Step6] %.3fs', state.timing.step6));
end

%% ---------------- Step 7: dataset split + models + report ----------------
if cfg.flags.runStep7
    % ensure dataset (auto unzip if needed)
    if ~exist(cfg.file.datasetRoot,'dir')
        if exist(cfg.file.datasetZip,'file')
            logline(logfile, '[Step7] Unzipping dataset_2025.zip ...');
            unzip(cfg.file.datasetZip, cfg.paths.data);
        else
            error('Dataset not found: %s (zip: %s)', cfg.file.datasetRoot, cfg.file.datasetZip);
        end
    end

    % 7a) dataset split
    t=tic;
    cfg.step7 = struct();
    cfg.step7.datasetRoot = cfg.file.datasetRoot;
    cfg.step7.inputSize   = [32 32 1];     % change if you want RGB: [32 32 3]
    cfg.step7.seed        = cfg.seed;
    state = step7_dataset(state, cfg);
    state.timing.step7_dataset = toc(t); logline(logfile, sprintf('[Step7/Dataset] %.3fs', state.timing.step7_dataset));

    % paths to cached models
    fCNN = fullfile(cfg.paths.models,'cnn.mat');
    fSVM = fullfile(cfg.paths.models,'hogsvm.mat');

    % 7b) CNN train or load
    if cfg.flags.trainCNN || ~cfg.flags.resumeModels || ~exist(fCNN,'file')
        t=tic; state = step7_task1_cnn(state, cfg); state.timing.step7_cnn = toc(t);
        logline(logfile, sprintf('[Step7/CNN] trained in %.3fs (ValAcc=%.3f)', state.timing.step7_cnn, state.step7.cnn.valAcc));
        try S=struct('net',state.step7.cnn.net,'inputSize',state.step7.inputSize,'classes',state.step7.classes);
            save(fCNN,'-struct','S'); logline(logfile, ['[Step7/CNN] saved: ' fCNN]); end
    else
        tmp = load(fCNN);
        state.step7.cnn = struct('net',tmp.net,'valAcc',NaN,'trainTime',NaN);
        state.step7.inputSize = tmp.inputSize;
        logline(logfile, ['[Step7/CNN] loaded cached model: ' fCNN]);
    end

    % 7c) Apply CNN (simple framework; teammate may refine inside the function)
    if cfg.flags.applyCNN
        state = step7_task1_apply_cnn(state, cfg);
    end

    % 7d) HOG+SVM train or load
    if cfg.flags.trainSVM || ~cfg.flags.resumeModels || ~exist(fSVM,'file')
        t=tic; state = step7_task2_hogsvm(state, cfg); state.timing.step7_svm = toc(t);
        logline(logfile, sprintf('[Step7/HOG+SVM] trained in %.3fs (ValAcc=%.3f)', state.timing.step7_svm, state.step7.hogsvm.valAcc));
        try S=struct('model',state.step7.hogsvm.model,'cellSize',state.step7.hogsvm.cellSize,'classes',state.step7.classes);
            save(fSVM,'-struct','S'); logline(logfile, ['[Step7/SVM] saved: ' fSVM]); end
    else
        tmp = load(fSVM);
        state.step7.hogsvm = struct('model',tmp.model,'valAcc',NaN,'trainTime',NaN,'cellSize',tmp.cellSize);
        logline(logfile, ['[Step7/SVM] loaded cached model: ' fSVM]);
    end

    % 7e) Apply HOG+SVM
    if cfg.flags.applySVM
        state = step7_task2_apply_noncnn(state, cfg);
    end

    % 7f) Compare
    state = step7_task3_report(state, cfg);
end

%% ---------------- Save final state ----------------
save(fullfile(cfg.paths.results,'run_state.mat'), 'state','cfg');
disp('All steps finished. Check ../results and ../results/figures .');


%% -------------- Helpers --------------
function ensure_dir(p)
    if ~exist(p, 'dir'), mkdir(p); end
end
function logline(f, s)
    fprintf('%s\n', s);
    try
        fid = fopen(f,'a'); fprintf(fid,'%s %s\n', datestr(now,'yyyy-mm-dd HH:MM:SS'), s); fclose(fid);
    catch
    end
end
