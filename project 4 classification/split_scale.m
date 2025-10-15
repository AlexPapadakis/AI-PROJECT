%% Split - Preprocess Data

function [trnData,chkData,tstData] = split_scale(data,preproc)

    % Stratified sampling to preserve class proportions
    classes = unique(data(:,end));
    trnIdx = [];
    chkIdx = [];
    tstIdx = [];
    
    % For each class, split proportionally
    for i = 1:length(classes)
        classData = find(data(:,end) == classes(i));
        classIdx = classData(randperm(length(classData)));
        
        % 60% training, 20% validation, 20% test
        nTrn = round(length(classIdx) * 0.6);
        nChk = round(length(classIdx) * 0.2);
        nTst = length(classIdx) - nTrn - nChk;  % Remaining samples
        
        trnIdx = [trnIdx; classIdx(1:nTrn)];
        chkIdx = [chkIdx; classIdx(nTrn+1:nTrn+nChk)];
        tstIdx = [tstIdx; classIdx(nTrn+nChk+1:end)];
    end
    
    % Shuffle the final indices to avoid class ordering
    trnIdx = trnIdx(randperm(length(trnIdx)));
    chkIdx = chkIdx(randperm(length(chkIdx)));
    tstIdx = tstIdx(randperm(length(tstIdx)));
    trnX=data(trnIdx,1:end-1);
    chkX=data(chkIdx,1:end-1);
    tstX=data(tstIdx,1:end-1);
    switch preproc
        case 1                      %Normalization to unit hypercube
            xmin=min(trnX,[],1);
            xmax=max(trnX,[],1);
            trnX=(trnX-repmat(xmin,[length(trnX) 1]))./(repmat(xmax,[length(trnX) 1])-repmat(xmin,[length(trnX) 1]));
            chkX=(chkX-repmat(xmin,[length(chkX) 1]))./(repmat(xmax,[length(chkX) 1])-repmat(xmin,[length(chkX) 1]));
            tstX=(tstX-repmat(xmin,[length(tstX) 1]))./(repmat(xmax,[length(tstX) 1])-repmat(xmin,[length(tstX) 1]));
        case 2                      %Standardization to zero mean - unit variance
            mu=mean(data,1);
            sig=std(data,1);
            trnX=(trnX-repmat(mu,[length(trnX) 1]))./repmat(sig,[length(trnX) 1]);
            chkX=(trnX-repmat(mu,[length(chkX) 1]))./repmat(sig,[length(chkX) 1]);
            tstX=(trnX-repmat(mu,[length(tstX) 1]))./repmat(sig,[length(tstX) 1]);
        otherwise
            disp('Not appropriate choice.')
    end
    trnData=[trnX data(trnIdx,end)];
    chkData=[chkX data(chkIdx,end)];
    tstData=[tstX data(tstIdx,end)];

end