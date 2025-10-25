function state = step7_task2_apply_mlp(state, cfg)
% STEP7_TASK2_APPLY_MLP - 使用 MLP 模型进行推理（自动低置信度重试 + 网格可视化）
%
% 自动加载:
%   ..\results\models\MLP_latest.mat
%   若不存在，则选取最新 MLP*.mat
%
% 特性:
%   - 低置信度 (<0.7) 自动调整 padScale 重试一次
%   - 控制台输出各类概率
%   - 生成网格可视化 png（用最终采用的“入网图像”）

    %% ====== 路径 ======
    if nargin<2, cfg = struct(); end
    if isfield(cfg,'paths') && isfield(cfg.paths,'results')
        resultsDir = cfg.paths.results;
    else
        resultsDir = fullfile('..','results');
    end
    if isfield(cfg,'paths') && isfield(cfg.paths,'figures')
        figDir = cfg.paths.figures;
    else
        figDir = fullfile('..','results','figures');
    end
    if ~exist(resultsDir,'dir'), mkdir(resultsDir); end
    if ~exist(figDir,'dir'),   mkdir(figDir);   end

    %% ====== 模型加载 ======
    modelsDir = fullfile(resultsDir,'models');
    if ~exist(modelsDir,'dir')
        modelsDir = fullfile('..','rerults','models'); % 容错
    end
    modelFile = fullfile(modelsDir,'MLP_latest.mat');
    if exist(modelFile,'file')~=2
        dd = dir(fullfile(modelsDir,'MLP*.mat'));
        assert(~isempty(dd),'未在 %s 找到任何 MLP 模型',modelsDir);
        [~,ix]=max([dd.datenum]);
        modelFile=fullfile(dd(ix).folder,dd(ix).name);
    end
    fprintf('[apply-mlp] 加载模型: %s\n',modelFile);
    S = load(modelFile);
    params=S.params; classes=S.classes; inSize=S.inputSize;
    if iscategorical(classes), classes = cellstr(classes); end
    if numel(inSize)==2, inSize(3)=1; end
    useCLAHE = true; if isfield(S,'useCLAHE'), useCLAHE=S.useCLAHE; end
    mu=[]; sigma=[];
    if isfield(S,'mu'), mu=S.mu; end
    if isfield(S,'sigma'), sigma=S.sigma; end

    %% ====== 输入数据 ======
    assert(isfield(state,'segment') && isfield(state.segment,'cropsBin'), ...
        '缺少 Step6 的裁剪结果 state.segment.cropsBin');
    crops = state.segment.cropsBin;
    N = numel(crops);
    fprintf('[apply-mlp] 识别 %d 个字符块\n',N);

    %% ====== 参数 ======
    padScale_default = 1.35;   % 初始白边
    padScale_retry    = 1.2;  % 重试白边
    D = prod(inSize(1:2));

    %% ====== 存放最终采用的入网图（用于网格可视化） ======
    Ximg = zeros(inSize(1), inSize(2), inSize(3), N, 'single');  % 记录最终采用的 Iproc
    labels = strings(N,1);
    scores = zeros(N,1);

    %% ====== 逐个识别并判断置信度 ======
    fprintf('\n=== MLP 识别结果（含低置信度重试） ===\n');
    for i=1:N
        Ci = crops{i};
        if ~isfloat(Ci), Ci=single(Ci); end
        Ci=max(0,min(1,Ci));

        % 第一次推理
        [label1, conf1, probs1, Iproc1] = run_once(Ci, params, classes, inSize, useCLAHE, padScale_default, mu, sigma);

        chosenI = Iproc1; chosenLabel = label1; chosenConf = conf1; chosenProbs = probs1;
        retried = false;

        % 若置信度低则重试一次
        if conf1 < 0.7
            [label2, conf2, probs2, Iproc2] = run_once(Ci, params, classes, inSize, useCLAHE, padScale_retry, mu, sigma);
            retried = true;
            if conf2 > conf1
                chosenI = Iproc2; chosenLabel = label2; chosenConf = conf2; chosenProbs = probs2;
                fprintf('#%02d → Pred: %-3s  Conf: %.2f%%  ↻重新猜测(%.2f%%)\n', ...
                    i, chosenLabel, chosenConf*100, conf1*100);
            else
                fprintf('#%02d → Pred: %-3s  Conf: %.2f%%  (重试无提升)\n', ...
                    i, chosenLabel, chosenConf*100);
            end
        else
            fprintf('#%02d → Pred: %-3s  Conf: %.2f%%\n', i, chosenLabel, chosenConf*100);
        end

        % 打印概率分布
        fprintf('   概率分布: ');
        for c=1:numel(classes)
            fprintf('%s=%.2f ', classes{c}, chosenProbs(c)*100);
        end
        fprintf('\n');

        % 记录最终结果 + 最终使用图像
        labels(i) = chosenLabel;
        scores(i) = chosenConf;
        Ximg(:,:,:,i) = chosenI;
    end
    fprintf('===========================================\n');

    %% ====== 网格可视化 ======
    try
        cols = 12; rows = ceil(N/cols);
        f = figure('Color','w','Name','MLP inputs (final used)');
        tiledlayout(rows,cols,'TileSpacing','compact','Padding','compact');
        for i=1:N
            nexttile;
            imshow(Ximg(:,:,:,i),'Border','tight');
            title(sprintf('%s(%.2f)',labels(i),scores(i)),'FontSize',8);
        end
        exportgraphics(f, fullfile(figDir,'step7_task2_mlp_inputs_grid.png'), 'Resolution',150);
        close(f);
        fprintf('[apply-mlp] 网格图已保存: %s\n', fullfile(figDir,'step7_task2_mlp_inputs_grid.png'));
    catch
        % 忽略可视化错误
    end

    %% ====== 回写 state ======
    state.step7.task2_apply = struct('labels',labels,'scores',scores,'N',N, ...
        'useCLAHE',useCLAHE,'padScale',[padScale_default padScale_retry], ...
        'modelFile',modelFile);
end

% ================= 内部函数 =================
function [label, conf, probs, Iproc] = run_once(Ci, params, classes, inSize, useCLAHE, padScale, mu, sigma)
    Iproc = preprocess_for_cnn_pad(Ci,inSize(1:2),useCLAHE,inSize(3),padScale);
    x = reshape(Iproc(:,:,1),[],1);
    if ~isempty(mu)&&~isempty(sigma)
        x=(x-mu)./sigma;
    end
    probs = mlp_forward_predict(x, params);
    [conf,idx] = max(probs);
    label = string(classes(idx));
end

function Iout = preprocess_for_cnn_pad(I,outSz,useCLAHE,outC,padScale)
    if size(I,3)>1, I=rgb2gray(I); end
    I=im2single(I);
    if useCLAHE
        I=adapthisteq(I,'NumTiles',[8 8],'ClipLimit',0.02);
    end
    S=outSz(1); T=outSz(2);
    s=min(S/size(I,1),T/size(I,2))/padScale;
    nh=max(1,min(S,round(size(I,1)*s)));
    nw=max(1,min(T,round(size(I,2)*s)));
    Ir=imresize(I,[nh nw],'nearest');
    canvas=ones(S,T,'single');
    y0=max(1,floor((S-nh)/2)+1);
    x0=max(1,floor((T-nw)/2)+1);
    y1=min(S,y0+nh-1);
    x1=min(T,x0+nw-1);
    Ir=Ir(1:(y1-y0+1),1:(x1-x0+1));
    canvas(y0:y1,x0:x1)=Ir;
    if nargin<4||outC==1
        Iout=canvas;
    else
        Iout=repmat(canvas,[1 1 3]);
    end
end

function P = mlp_forward_predict(X, params)
    % X: D x 1
    L=numel(params.W);
    A=X;
    for l=1:L-1
        Z=params.W{l}*A+params.b{l};
        A=max(0,Z);
    end
    ZL=params.W{L}*A+params.b{L};
    ZL=ZL-max(ZL,[],1);
    P=exp(ZL); P=P./sum(P,1);
end
