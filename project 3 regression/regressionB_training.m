%% =================== ANFIS Training with Relief ===================

clc; clear; close all;

%% ----------- Load Dataset -----------
data = readtable('train.csv'); 
X = table2array(data(:,1:end-1));
Y = table2array(data(:,end));

%% ----------- 60-20-20 Split -----------
N = size(X,1);
idx = randperm(N);
nTrn = round(0.6*N); nVal = round(0.2*N);

Xtrn = X(idx(1:nTrn),:);   Ytrn = Y(idx(1:nTrn),:);
Xval = X(idx(nTrn+1:nTrn+nVal),:); Yval = Y(idx(nTrn+1:nTrn+nVal),:);
Xchk = X(idx(nTrn+nVal+1:end),:); Ychk = Y(idx(nTrn+nVal+1:end),:);

%% ----------- Feature Ranking with Relief -----------
numFeaturesSet = [5 10 20 40];
[ranks,weights] = relieff(Xtrn,Ytrn,10);

%% ----------- Grid Search over Features & Cluster Radius -----------
raSet = [0.3 0.5 0.7];
nFolds = 5;

bestRMSE = inf;
results = [];
counter = 0;
totalComb = length(numFeaturesSet)*length(raSet)*nFolds;


for nf = numFeaturesSet
    featIdx = ranks(1:nf);
    Xtrn_sel = Xtrn(:,featIdx);

    for ra = raSet
        cv = cvpartition(size(Xtrn_sel,1),'KFold',nFolds);
        cvErr = zeros(nFolds,1);
    
        for k = 1:nFolds
            trainIdx = training(cv,k);
            valIdx = test(cv,k);
            Xtrain_cv = Xtrn_sel(trainIdx,:); Ytrain_cv = Ytrn(trainIdx,:);
            Xval_cv = Xtrn_sel(valIdx,:);    Yval_cv = Ytrn(valIdx,:);
    
            options = genfisOptions('SubtractiveClustering','ClusterInfluenceRange',ra);
            fis = genfis(Xtrain_cv,Ytrain_cv,options);
    
            anfisOpt = anfisOptions('InitialFIS',fis,'EpochNumber',50,...
                'DisplayANFISInformation',0,'DisplayErrorValues',0,'DisplayStepSize',0);
            trnFisCV = anfis([Xtrain_cv Ytrain_cv], anfisOpt);
    
            Ypred = evalfis(Xval_cv,trnFisCV);
            cvErr(k) = sqrt(mean((Ypred - Yval_cv).^2));
    
            counter = counter + 1;
            fprintf('NF=%d | RA=%.2f | Fold=%d/%d | [%d/%d]\n', nf, ra, k, nFolds, counter, totalComb);
        end
    
        meanErr = mean(cvErr);
        meanRules = length(fis.Rules);
        results = [results; nf ra meanErr meanRules];
    
        if meanErr < bestRMSE
            bestRMSE = meanErr;
            bestNF = nf; bestRA = ra;
        end
    end

end

%% ----------- Train Final Model with Best Parameters -----------
featIdx = ranks(1:bestNF);
Xtrn_sel = Xtrn(:,featIdx);

options = genfisOptions('SubtractiveClustering','ClusterInfluenceRange',bestRA);
fis = genfis(Xtrn_sel,Ytrn,options);

anfisOpt = anfisOptions('InitialFIS',fis,'EpochNumber',100,...
    'DisplayANFISInformation',1,'DisplayErrorValues',1,'DisplayStepSize',1);
[trnFis, trnError] = anfis([Xtrn_sel Ytrn], anfisOpt);

%% ----------- Evaluate Final Model on Check Set -----------
Xchk_sel = Xchk(:,featIdx);
Ypred_chk = evalfis(Xchk_sel,trnFis);

RMSE = sqrt(mean((Ypred_chk - Ychk).^2));
NMSE = mean((Ypred_chk - Ychk).^2)/var(Ychk);
NDEI = sqrt(NMSE);
R2 = 1 - sum((Ychk-Ypred_chk).^2)/sum((Ychk-mean(Ychk)).^2);

fprintf('\nBest features: %d, Cluster radius: %.2f\n',bestNF,bestRA);
fprintf('RMSE: %.4f, NMSE: %.4f, NDEI: %.4f, R2: %.4f\n',RMSE,NMSE,NDEI,R2);

T = table(RMSE,NMSE,NDEI,R2);
disp('Performance metrics:'); disp(T);
