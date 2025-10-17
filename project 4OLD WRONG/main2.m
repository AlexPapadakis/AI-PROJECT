clc; clear; close all;

disp("Start of script");

%% Read data and normalise
data = readmatrix('Epileptic Seizure Recognition.csv');              % modern replacement for csvread
data = data(2:end, 2:end);                  % skip first row & col if needed
data = normaliseData(data);

%% Grid search setup
a = 5;          
b = 7;

% First param is num of features, second param is clusters ra
gridSearchParams = zeros(a,b,2);
gridSearchParams(:,:,1) = repmat([5;10;15;20;25],1,b);
gridSearchParams(:,1,2) = 0.2;
gridSearchParams(:,2,2) = 0.3;
gridSearchParams(:,3,2) = 0.4;
gridSearchParams(:,4,2) = 0.5;
gridSearchParams(:,5,2) = 0.6;
gridSearchParams(:,6,2) = 0.7;
gridSearchParams(:,7,2) = 0.8;

errors = zeros(a,b);
k = 5;   % k-fold cross validation

classes = unique(data(:,end));
nClasses = numel(classes);

%% Grid search
tic
for w = 1:a
    for z = 1:b
        numOfFeatures = gridSearchParams(w,z,1);
        ra = gridSearchParams(w,z,2);

        crossValOA = zeros(k,1);
        cvPart = cvpartition(data(:,end),"KFold",k,"Stratify",true);

        for iteration = 1:k
            trnDataTemp = data(training(cvPart,iteration),:);
            tstData     = data(test(cvPart,iteration),:);

            % secondary CV split for training/validation
            cvPartitionTrn = cvpartition(trnDataTemp(:,end),"KFold",4,"Stratify",true);
            trnData = trnDataTemp(training(cvPartitionTrn,1), :);
            chkData = trnDataTemp(test(cvPartitionTrn,1), :);

            %% Feature selection
            [idx,~] = relieff(trnData(:,1:end-1), trnData(:,end), 5);
            feats = idx(1:numOfFeatures);

            trnDataFS = [trnData(:,feats) trnData(:,end)];
            chkDataFS = [chkData(:,feats) chkData(:,end)];
            tstDataFS = [tstData(:,feats) tstData(:,end)];

            %% Clustering per class
            clusterCenters = cell(nClasses,1);
            clusterSigma   = cell(nClasses,1);
            nRulesPerClass = zeros(nClasses,1);
            for ci = 1:nClasses
                classLabel = classes(ci);
                [c,s] = subclust(trnDataFS(trnDataFS(:,end)==classLabel,:), ra);
                clusterCenters{ci} = c;
                clusterSigma{ci}   = s;
                nRulesPerClass(ci) = size(c,1);
            end
            num_rules = sum(nRulesPerClass);

            %% Build FIS
            fis = sugfis('Name',"FIS_SC");    

            % Add input variables
            nInputs = size(trnDataFS,2)-1;
            for i = 1:nInputs
                fis = addInput(fis,[0 1],'Name',"input"+i);
            end
            fis = addOutput(fis,[0 1],'Name',"out1");

            % Add input membership functions
            for i = 1:nInputs
                varName = "input" + i;
                for ci = 1:nClasses
                    c = clusterCenters{ci};
                    s = clusterSigma{ci};
                    for j = 1:size(c,1)
                        fis = addMF(fis,varName,"gaussmf",[s(i) c(j,i)]);
                    end
                end
            end

            % Add output membership functions (constants per class)
            params = [];
            for ci = 1:nClasses
                constVal = (ci-1)/(nClasses-1); % map classes to [0,1]
                params = [params repmat(constVal,1,nRulesPerClass(ci))];
            end
            for r = 1:num_rules
                fis = addMF(fis,"out1","constant",params(r));
            end

            % Build rule base
            m = nInputs;
            n = 1;
            ruleLength = m + n + 2;
            ruleList = zeros(num_rules, ruleLength);

            mfCounter = 0; % tracks MF indices across classes
            for ci = 1:nClasses
                for j = 1:nRulesPerClass(ci)
                    mfCounter = mfCounter + 1;

                    % Each input uses this MF index
                    ruleList(mfCounter,1:m) = mfCounter;

                    % Output MF index
                    ruleList(mfCounter,m+1) = mfCounter;

                    % Weight + AND connection
                    ruleList(mfCounter,m+2) = 1;
                    ruleList(mfCounter,m+3) = 1;
                end
            end
            fis = addRule(fis,ruleList);

            %% Train ANFIS
            opt = anfisOptions("InitialFIS",fis, ...
                               "EpochNumber",100, ...
                               "ErrorGoal",0, ...
                               "InitialStepSize",0.01, ...
                               "StepSizeDecreaseRate",0.9, ...
                               "StepSizeIncreaseRate",1.1, ...
                               "ValidationData",chkDataFS);
            [~,trnError,~,valFis,valError] = anfis(trnDataFS,opt);

            figure();
            plot([trnError valError],'LineWidth',2); grid on;
            legend('Training Error','Validation Error');
            xlabel('# of Epochs'); ylabel('Error');

            %% Evaluate
            Y = evalfis(valFis, tstDataFS(:,1:end-1));
            Y = round(Y*(nClasses-1)+1);   % map back to class labels 1..nClasses
            diff = tstDataFS(:,end)-Y;
            Acc  = (length(diff)-nnz(diff))/length(Y)*100;

            % Confusion/error matrix
            errorMatrix = zeros(nClasses);
            for i = 1:nClasses
                for j = 1:nClasses
                    errorMatrix(i,j) = nnz( Y==classes(i) & tstData(:,end)==classes(j) );
                end
            end

            sumCorrect = trace(errorMatrix);
            OA = sumCorrect/size(tstDataFS,1);

            crossValOA(iteration) = OA;
        end

        errors(w,z) = mean(crossValOA);
    end
end
toc

disp(errors);
disp("End of script");
