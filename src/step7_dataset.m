function state = step7_dataset(state, cfg)
% STEP7_DATASET - Load dataset and make a stratified 75/25 split.
% Produces imageDatastore for train/val and stores meta into state.step7.
%
% cfg.step7:
%   .datasetRoot : root folder (default fullfile(cfg.paths.data,'dataset_2025'))
%   .inputSize   : [H W 1] target size for models (default [32 32 1])
%   .seed        : RNG seed (default 5411)

    if ~isfield(cfg,'step7'), cfg.step7 = struct(); end
    S = cfg.step7;
    if ~isfield(S,'datasetRoot') || isempty(S.datasetRoot)
        S.datasetRoot = fullfile(cfg.paths.data,'dataset_2025');
    end
    if ~isfield(S,'inputSize') || isempty(S.inputSize)
        S.inputSize = [32 32 1];
    end
    if ~isfield(S,'seed') || isempty(S.seed)
        S.seed = 5411;
    end

    assert(exist(S.datasetRoot,'dir')==7, 'Dataset root not found: %s', S.datasetRoot);

    imds = imageDatastore(S.datasetRoot, ...
        'IncludeSubfolders', true, 'LabelSource','foldernames');

    % stratified split 75/25
    rng(S.seed);
    [imdsTrain, imdsVal] = splitEachLabel(imds, 0.75, 'randomized');

    classes = categories(imds.Labels);
    trainTbl = countEachLabel(imdsTrain); valTbl = countEachLabel(imdsVal);

    % stash to state
    state.step7 = struct();
    state.step7.imdsTrain = imdsTrain;
    state.step7.imdsVal   = imdsVal;
    state.step7.classes   = classes;
    state.step7.inputSize = S.inputSize;
    state.step7.seed      = S.seed;

    % save a small summary
    if isfield(cfg,'paths') && isfield(cfg.paths,'results')
        outdir = cfg.paths.results;
        if ~exist(outdir,'dir'), mkdir(outdir); end
        writetable(trainTbl, fullfile(outdir,'step7_train_counts.csv'));
        writetable(valTbl,   fullfile(outdir,'step7_val_counts.csv'));
    end

    fprintf('[Step7/Dataset] %d classes. Train=%d, Val=%d\n', numel(classes), ...
        numel(imdsTrain.Files), numel(imdsVal.Files));
end
