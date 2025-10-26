% MAIN ENTRY FOR ME5411
% A clean, switch-driven pipeline to run Steps 1–7:
%   - Image display/enhancement, filtering, ROI, binarization, outline, segmentation
%   - CNN training/apply (via your cnn.m + adapter step)
%   - MLP training/apply (controlled by flags)
% All outputs/logs go to ../results, figures to ../results/figures, models to ../results/models.

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
cfg.file.image       = 'charact2.bmp';
cfg.file.datasetRoot = fullfile(cfg.paths.data,'dataset_2025');
cfg.file.datasetZip  = fullfile(cfg.paths.data,'dataset_2025.zip');

% master switches
cfg.flags = struct();
cfg.flags.runStep1     = true;
cfg.flags.runStep2     = true;
cfg.flags.runStep3     = true;
cfg.flags.runStep4     = true;
cfg.flags.runStep5     = true;
cfg.flags.runStep6     = true;
cfg.flags.runStep7     = true;

% model/training switches
cfg.flags.trainCNN     = false;      % train once; set false later to reuse cached model
cfg.flags.applyCNN     = true;       % apply CNN predictions
cfg.flags.resumeModels = true;       % allow loading existing models if available

% NEW: MLP switches
cfg.flags.trainMLP     = false;      % set true to train/update MLP
cfg.flags.applyMLP     = true;       % set true to apply MLP after CNN

% ensure dirs
ensure_dir(cfg.paths.results);
ensure_dir(cfg.paths.figures);
ensure_dir(cfg.paths.models);
ensure_dir(cfg.paths.logs);

% simple log
logfile = fullfile(cfg.paths.logs, [datestr(now,'yyyymmdd_HHMMSS') '_run.log']);
ensure_dir(fileparts(logfile));
logline(logfile, '=== ME5411 pipeline started ===');
rng(cfg.seed);

%% ---------------- Step 1: display + enhancement ----------------
state = struct();
if cfg.flags.runStep1
    t=tic;
    state = step1_display(state, cfg);
    state = step1_enhancement(state, cfg);
    state.timing.step1 = toc(t);
    logline(logfile, sprintf('[Step1] done in %.3fs', state.timing.step1));
end

%% ---------------- Step 2: filtering ----------------
if cfg.flags.runStep2
    t=tic;
    cfg.filter = struct('winSizes',5,'padMode','replicate');
    state = step2_filter(state, cfg);
    state.timing.step2 = toc(t);
    logline(logfile, sprintf('[Step2] %.3fs', state.timing.step2));
end

%% ---------------- Step 3: ROI ----------------
if cfg.flags.runStep3
    t=tic;
    % bbox = [x y w h]; margin is optional expansion around ROI
    cfg.roi = struct('bbox',[20 200 950 150],'margin',0);
    state = step3_roi(state, cfg);
    state.timing.step3 = toc(t);
    logline(logfile, sprintf('[Step3] %.3fs', state.timing.step3));
end

%% ---------------- Step 4: binarization ----------------
if cfg.flags.runStep4
    t=tic;
    % polarity: 'bright' means white foreground on black, adapt as needed
    cfg.bin = struct('polarity','bright','nbins',256);
    state = step4_binarize(state, cfg);
    state.timing.step4 = toc(t);
    logline(logfile, sprintf('[Step4] %.3fs', state.timing.step4));
end

%% ---------------- Step 5: outline ----------------
if cfg.flags.runStep5
    t=tic;
    cfg.outline = struct('minLength',15,'connectivity','8');
    state = step5_outline(state, cfg);
    state.timing.step5 = toc(t);
    logline(logfile, sprintf('[Step5] %.3fs', state.timing.step5));
end

%% ---------------- Step 6: segmentation ----------------
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

%% ---------------- Step 7: CNN/MLP ----------------
if cfg.flags.runStep7
    % 7a/b) CNN train or load (handled inside your cnn.m via adapter)
    t = tic;
    state = step7_task1_cnn(state, cfg);   % prepares/loads state.step7.cnn.*
    state.timing.step7_cnn = toc(t);
    % Defensive logging in case valAcc is absent
    valAccStr = '(n/a)';
    try
        if isfield(state,'step7') && isfield(state.step7,'cnn') && isfield(state.step7.cnn,'valAcc')
            valAccStr = sprintf('%.3f', state.step7.cnn.valAcc);
        end
    catch
    end
    logline(logfile, sprintf('[Step7/CNN] ready in %.3fs (ValAcc=%s)', ...
        state.timing.step7_cnn, valAccStr));

    % 7c) Apply CNN
    if cfg.flags.applyCNN
        state = step7_task1_apply_cnn(state, cfg);
        logline(logfile, '[Step7/CNN] apply done.');
    end

    % 7d) MLP train
    if cfg.flags.trainMLP
        t = tic;
        state = step7_task2_mlp(state, cfg);
        state.timing.step7_mlp_train = toc(t);
        logline(logfile, sprintf('[Step7/MLP] trained in %.3fs', state.timing.step7_mlp_train));
    end

    % 7e) MLP apply (optional)
    if cfg.flags.applyMLP
        t = tic;
        state = step7_task2_apply_mlp(state, cfg);
        state.timing.step7_mlp_apply = toc(t);
        logline(logfile, sprintf('[Step7/MLP] apply in %.3fs', state.timing.step7_mlp_apply));
    end

    % If SVM or other models are not needed, keep them commented out.
end

%% ---------------- Save final state ----------------
save(fullfile(cfg.paths.results,'run_state.mat'), 'state','cfg');
disp('All steps finished. Check ../results and ../results/figures .');

%% -------------- Helpers --------------
function ensure_dir(p)
    % Create directory if it does not exist
    if ~exist(p, 'dir'), mkdir(p); end
end

function logline(f, s)
    % Mirror log to console and append to a file safely
    fprintf('%s\n', s);
    try
        fid = fopen(f,'a');
        fprintf(fid,'%s %s\n', datestr(now,'yyyy-mm-dd HH:MM:SS'), s);
        fclose(fid);
    catch
        % ignore file I/O errors to not break the run
    end
end
