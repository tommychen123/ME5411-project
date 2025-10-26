%% augment_dataset_safe.m - 离线扩充数据（安全增广，不翻转、不剪切）
% 输入:  inRoot  = '../data/dataset_2025'        % 原始数据，每类一个子文件夹
% 输出:  outRoot = '../data/dataset_2025_aug'    % 扩充后的数据集（含原图 + 增广图）
% 做法:  每类最少扩充到 targetPerClass（可修改）；不会改变语义的形变和光照扰动

clc; clear; close all;

%% ====== 路径 & 目标数量 ======
inRoot  = fullfile('..','data','dataset_2025');       % TODO: 改成你的原始目录
outRoot = fullfile('..','data','dataset_2025_aug');   % 输出目录
targetPerClass = 3000;                                % 每类目标张数，可改

if ~exist(inRoot,'dir'), error('原始数据集不存在: %s', inRoot); end
if ~exist(outRoot,'dir'), mkdir(outRoot); end

imds = imageDatastore(inRoot, 'IncludeSubfolders', true, 'LabelSource','foldernames');
classes = categories(imds.Labels);
fprintf('发现 %d 个类别：\n', numel(classes)); disp(classes');

rng(42);  % 复现

for ci = 1:numel(classes)
    cname = classes{ci};
    idx = find(imds.Labels == cname);
    srcFiles = imds.Files(idx);

    if isempty(srcFiles)
        warning('[%s] 没有原始样本，跳过。', cname);
        continue;
    end

    outDir = fullfile(outRoot, cname);
    if ~exist(outDir,'dir'), mkdir(outDir); end

    % 先把原图拷贝（仅第一次）
    if isempty(dir(fullfile(outDir, '*.png')))
        for k = 1:numel(srcFiles)
            I = imread(srcFiles{k});
            [~, base, ~] = fileparts(srcFiles{k});
            imwrite(I, fullfile(outDir, sprintf('%s_orig.png', base)));
        end
    end

    nNow = numel(dir(fullfile(outDir, '*.png')));
    if nNow >= targetPerClass
        fprintf('[%s] 已有 %d >= %d，跳过增广。\n', cname, nNow, targetPerClass);
        continue;
    end

    need = targetPerClass - nNow;
    fprintf('[%s] 需要增广 %d 张（当前 %d / 目标 %d）。\n', cname, need, nNow, targetPerClass);

    gen = 0; sN = numel(srcFiles);
    while gen < need
        I = imread(srcFiles{randi(sN)});
        J = safe_augment_once(I);                 % —— 安全增广 —— %
        imwrite(J, fullfile(outDir, sprintf('aug_%06d.png', nNow+gen+1)));
        gen = gen + 1;
        if mod(gen, 200) == 0
            fprintf('   [%s] 已增广 %d/%d\n', cname, gen, need);
        end
    end
    fprintf('[%s] 完成，总数≈%d\n', cname, targetPerClass);
end
fprintf('\n全部完成：%s\n', outRoot);

%% ---------- 单次安全增广（不翻转、不剪切、不拉伸变形） ----------
function J = safe_augment_once(I)
    % 支持灰度或RGB；返回同通道数
    I = im2single(I);

    % 形变参数（温和）—— 不改变语义
    rotDeg = (rand()*6 - 3);          % -3° ~ 3° 轻量旋转
    scale  = 0.99 + rand()*0.03;      % 0.99 ~ 1.02 等比缩放
    tx     = randi([-2, 2]);          % 平移 x
    ty     = randi([-2, 2]);          % 平移 y

    % 仿射矩阵（旋转 + 等比缩放）
    tRot  = [cosd(rotDeg) -sind(rotDeg) 0; sind(rotDeg) cosd(rotDeg) 0; 0 0 1];
    tScl  = [scale 0 0; 0 scale 0; 0 0 1];
    T = tRot * tScl;
    tform = affine2d(T);

    % 尺寸与填充
    Rout = imref2d(size(I(:,:,1)));
    fillVal = ones(1, 1, size(I,3), 'like', I);  % 自动匹配通道数和数据类型
    fillVal(:) = 1;  % 填充为白色

    J = imwarp(I, tform, 'OutputView', Rout, 'FillValues', fillVal);
    J = imtranslate(J, [tx ty], 'FillValues', fillVal);

    % 光照扰动（微弱）
    alpha = 0.95 + rand()*0.10;       % 0.95~1.05 亮度缩放
    beta  = (rand()*0.06 - 0.03);     % -0.03~0.03 偏置
    J = J*alpha + beta;
    J = min(max(J,0),1);

    % 小概率加极轻噪声
    if rand() < 0.10
        J = imnoise(J, 'gaussian', 0, 0.0015);
    end

    J = im2uint8(J);
end
