function state = step7_task3_report(state, cfg)
% Summarize Task1(CNN) vs Task2(HOG+SVM): val accuracy, train time, inference time, Image1 strings.

    hasCNN = isfield(state,'step7') && isfield(state.step7,'cnn');
    hasSVM = isfield(state,'step7') && isfield(state.step7,'hogsvm');

    M = {}; Acc=[]; TrainT=[]; InferT=[]; Text={};
    if hasCNN
        M{end+1}='CNN';  Acc(end+1)=state.step7.cnn.valAcc;
        TrainT(end+1)=state.step7.cnn.trainTime;
        InferT(end+1)=getfield_safe(state.step7,'task1','inferTimeSec',NaN);
        Text{end+1}=getfield_safe(state.step7,'task1','text','');
    end
    if hasSVM
        M{end+1}='HOG+SVM'; Acc(end+1)=state.step7.hogsvm.valAcc;
        TrainT(end+1)=state.step7.hogsvm.trainTime;
        InferT(end+1)=getfield_safe(state.step7,'task2','inferTimeSec',NaN);
        Text{end+1}=getfield_safe(state.step7,'task2','text','');
    end

    T=table(M',Acc',TrainT',InferT',Text','VariableNames',...
        {'Method','ValAccuracy','TrainTime_sec','Image1_InferTime_sec','Image1_String'});

    if ~exist(cfg.paths.results,'dir'), mkdir(cfg.paths.results); end
    writetable(T, fullfile(cfg.paths.results,'step7_compare_summary.csv'));
    disp(T);
end

function v=getfield_safe(s, f1, f2, def)
    v=def; if isfield(s,f1) && isfield(s.(f1),f2), v=s.(f1).(f2); end
end
