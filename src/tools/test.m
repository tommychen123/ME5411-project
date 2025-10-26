%% test.m - 单图推理（输出所有类别的概率，按从高到低排序）
% 需要先运行 cnn.m 生成 ../results/models/CNN_latest.mat

clc; clear; close all;

%% ====== 路径与加载 ======
modelFile = fullfile('..','results','models','CNN_latest.mat');
assert(isfile(modelFile), '找不到模型文件: %s', modelFile);
S = load(modelFile);
net = S.net;

% 读取类名 & 输入尺寸 & CLAHE 开关
if isfield(S,'classes'), classes = S.classes; else, classes = net.Layers(end).Classes; end
if iscategorical(classes), classes = cellstr(classes); end
if isfield(S,'inputSizeSave'), inputSize = S.inputSizeSave; else, inputSize = net.Layers(1).InputSize; end
useCLAHE = true; if isfield(S,'useCLAHE'), useCLAHE = S.useCLAHE; end

%% ====== 指定要预测的图片 ======
imgFile = '..\results\figures\dbg_apply_cnn\cmp\input_04.png';   % TODO: 换成你的图片
assert(isfile(imgFile), '找不到测试图片: %s', imgFile);

%% ====== 预处理（与训练一致） ======
I0 = imread(imgFile);
if size(I0,3)>1, I0 = rgb2gray(I0); end
I0 = im2single(I0);
Iproc = preprocess_for_cnn(I0, inputSize(1:2), useCLAHE, inputSize(3));  % 灰度→通道匹配
Xin = reshape(Iproc, [inputSize 1]);   % H x W x C x 1

%% ====== 预测 ======
[scores] = predict(net, Xin);   % 1 x numClasses (有时是 N x C，取第1样本即可)
scores = squeeze(scores);       % 列向量
[sortedScores, idx] = sort(scores, 'descend');
sortedClasses = classes(idx);

% Top-1
predLabel = string(sortedClasses{1});
conf      = sortedScores(1);
fprintf('Predicted: %s (%.2f%%)\n', predLabel, conf*100);

%% ====== 打印所有类别概率 ======
T = table(string(classes(:)), scores(:), 'VariableNames', {'Class','Prob'});
T = sortrows(T, 'Prob', 'descend');
disp(T);

%% ====== 可视化 ======
figure('Color','w','Position',[100,100,900,350]);
subplot(1,2,1);
imshow(Iproc(:,:,min(1,size(Iproc,3))), 'Border', 'tight'); title('Preprocessed Input');

subplot(1,2,2);
barh(sortedScores);
set(gca,'YTick',1:numel(sortedClasses),'YTickLabel',sortedClasses);
xlim([0 1]); xlabel('Probability'); title('Class Probabilities (sorted)');
set(gca,'YDir','reverse'); grid on;

%% ====== 可选：保存概率到CSV ======
% writetable(T, 'probs.csv');

%% ====== Helper：与训练一致的预处理 ======
function Iout = preprocess_for_cnn(I, outSz, useCLAHE, outC)
    if useCLAHE
        I = adapthisteq(I,'NumTiles',[8 8],'ClipLimit',0.02);
    end
    S = outSz(1); T = outSz(2);
    s  = min(S/size(I,1), T/size(I,2));
    nh = max(1, min(S, round(size(I,1)*s)));
    nw = max(1, min(T, round(size(I,2)*s)));
    Ir = imresize(I, [nh nw], 'nearest');
    y0 = max(1, floor((S - nh)/2) + 1);
    x0 = max(1, floor((T - nw)/2) + 1);
    y1 = min(S, y0 + nh - 1);
    x1 = min(T, x0 + nw - 1);
    Ir = Ir(1:(y1-y0+1), 1:(x1-x0+1));
    canvas = ones(S,T,'single');
    canvas(y0:y1, x0:x1) = Ir;
    if nargin<4 || outC==1
        Iout = canvas;
    else
        Iout = repmat(canvas, [1 1 3]);
    end
end
