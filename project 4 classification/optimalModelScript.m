%clc;clear;
disp("Start of script");

% Read data and normalise - Using Epileptic Seizure Recognition dataset
data = csvread('Epileptic Seizure Recognition.csv',1,1);
data = normaliseData(data);

% Get dataset info
fprintf('Dataset Info:\n');
fprintf('Samples: %d\n', size(data,1));
fprintf('Features: %d\n', size(data,2)-1);
fprintf('Classes: %d\n', length(unique(data(:,end))));

% Optimal Model - Epileptic dataset has 179 features, 5 classes
numOfFeatures = 15;  % Changed to reasonable number for 179 features
ra = 0.5;

% k-fold cross validaiton
k = 5;

% Get number of classes dynamically
numClasses = length(unique(data(:,end)));
fprintf('Number of classes: %d\n', numClasses);

% Crossvalidate R2 and RMSE
crossValErrors = zeros(k,2);

% k-fold cross validation
crossValOA = zeros(k,1);
crossValPA = zeros(numClasses,k);
crossValUA = zeros(numClasses,k);
crossValk  = zeros(k,1);
crossValErrorMatrix = zeros(numClasses,numClasses,k);
cvPart = cvpartition(data(:,end),'KFold',5,'Stratify',true);

% k-fold cross validation
for iteration = 1:k
    trnDataTemp = data(training(cvPart,iteration),:);
    tstData = data(test(cvPart,iteration),:);
    % cv partition is used to split trnDataTemp into trnData and
    % chkData using stratification
    cvPartitionTrn = cvpartition(trnDataTemp(:,end),'KFold',4,'Stratify',true);
    trnData = trnDataTemp(training(cvPartitionTrn,1), :);
    chkData = trnDataTemp(test(cvPartitionTrn,1), :);
    
    % Feature selection
    [idx,weights] = relieff( trnData(:,1:end-1), trnData(:,end),5);
    
    trnDataFS = trnData( :, idx(1:numOfFeatures) );
    trnDataFS = [ trnDataFS trnData( :, end)];
    
    chkDataFS = chkData( :, idx(1:numOfFeatures) );
    chkDataFS = [ chkDataFS chkData( :, end) ];
    
    tstDataFS = tstData( :, idx(1:numOfFeatures) );
    tstDataFS = [ tstDataFS tstData( :, end) ];
    
    % Clustering Per Class - Dynamic for any number of classes
    clusters = cell(numClasses, 1);
    sigmas = cell(numClasses, 1);
    num_rules = 0;
    
    for classIdx = 1:numClasses
        classData = trnDataFS(trnDataFS(:,end) == classIdx, :);
        if size(classData, 1) > 0  % Check if class has data
            [clusters{classIdx}, sigmas{classIdx}] = subclust(classData, ra);
            num_rules = num_rules + size(clusters{classIdx}, 1);
        else
            clusters{classIdx} = [];
            sigmas{classIdx} = [];
        end
    end
    
    fprintf('Total rules for iteration %d: %d\n', iteration, num_rules);
    
    % Build FIS From Scratch using modern syntax
    fis2 = sugfis('Name','FIS_SC');
    
    % Add Input-Output Variables using modern syntax
    names_in = {};
    for i = 1:size(trnDataFS,2)-1
        num = int2str(i);
        name = 'input';
        name = strcat(name,num);
        names_in = [names_in name];
    end
    for i = 1:size(trnDataFS,2)-1
        fis2 = addInput(fis2,[0 1],'Name',names_in{i});
    end
    fis2 = addOutput(fis2,[0 1],'Name','out1');
    
    % Add Input Membership Functions using modern syntax - Dynamic
    for i = 1:size(trnDataFS,2)-1
        for classIdx = 1:numClasses
            if ~isempty(clusters{classIdx})
                for j = 1:size(clusters{classIdx}, 1)
                    fis2 = addMF(fis2, names_in{i}, 'gaussmf', ...
                        [sigmas{classIdx}(i), clusters{classIdx}(j,i)]);
                end
            end
        end
    end
    
    % Add Output Membership Functions - Dynamic for any number of classes
    params = [];
    for classIdx = 1:numClasses
        if ~isempty(clusters{classIdx})
            % Normalize class values to [0,1] range
            classValue = (classIdx - 1) / (numClasses - 1);
            classParams = repmat(classValue, 1, size(clusters{classIdx}, 1));
            params = [params, classParams];
        end
    end
    for i = 1:num_rules
        fis2 = addMF(fis2,'out1','constant',params(i));
    end
    
    % Add FIS Rule Base using modern syntax
    ruleList = zeros(num_rules,size(trnDataFS,2));
    for i = 1:size(ruleList,1)
        ruleList(i,:)=i;
    end
    ruleList = [ruleList ones(num_rules,2)];
    fis2 = addRule(fis2,ruleList);
    
    % Plot mf before training using original syntax (still valid)
    titleBefore = "Optimal TSK model membership functions for input ";
    figure(1);
    plotmf(fis2,'input',1);
    title1 = strcat(titleBefore,'1');
    title1 = strcat(title1,' before training');
    title(title1);
    
    figure(2);
    plotmf(fis2,'input',size(trnDataFS,2)-1);
    num = int2str(size(trnDataFS,2)-1);
    title2 = strcat(titleBefore,num);
    title2 = strcat(title2,' before training');
    title(title2);
    
    % Train & Evaluate ANFIS
    [trnFis,trnError,~,valFis,valError]=anfis(trnDataFS,fis2,[150 0 0.01 0.9 1.1],[],chkDataFS);
    
    % Learning curve plot
    figure(1000);
    plot([trnError valError],'LineWidth',2); grid on;
    legend('Training Error','Validation Error');
    xlabel('# of Epochs');
    ylabel('Error');
    Y=evalfis(valFis,tstDataFS(:,1:end-1));
    
    % Round predictions to nearest class and ensure valid range
    Y = round(Y * (numClasses - 1)) + 1;  % Convert from [0,1] to class labels
    Y = max(1, min(numClasses, Y));  % Clamp to valid class range
    
    diff=tstDataFS(:,end)-Y;
    Acc=(length(diff)-nnz(diff))/length(Y)*100;
    
    % Plot mf after training using original syntax (still valid)
    titleAfter = "Optimal TSK model membership functions for input ";
    figure(4);
    plotmf(valFis,'input',1);
    title1 = strcat(titleAfter,'1');
    title1 = strcat(title1,' after training');
    title(title1);
    
    figure(5);
    plotmf(valFis,'input',size(trnDataFS,2)-1);
    num = int2str(size(trnDataFS,2)-1);
    title2 = strcat(titleAfter,num);
    title2 = strcat(title2,' after training');
    title(title2);
    
    % Predictions plot
    figure(7)
    scatter(1:size(Y,1),Y,'.'); grid on;
    xlabel('input');
    legend('Prediction');
    title('Model predictions');
    
    % Ground truth
    figure(8)
    scatter(1:size(tstDataFS,1),tstDataFS(:,end),'.');grid on;
    xlabel('input');
    legend('Ground truth');
    title('Ground truth');
    
    % Prediction Error plot
    predictionError = tstDataFS(:,end) - Y;
    figure(9);
    scatter(1:size(predictionError,1),predictionError,'.'); grid on;
    xlabel('input');ylabel('Error');
    legend('Prediction Error');
    title('Optimal TSK model prediction error')
    
    classes = 1:numClasses;
    errorMatrix = zeros(numClasses);
    N = size(tstDataFS,1);
    % Error matrix - Dynamic
    for i = 1:numClasses
        for j = 1:numClasses
            errorMatrix(i,j) = size( intersect( find( Y == classes(i) ) , find(tstDataFS(:,end) == classes(j) ) ) ,1);
        end
    end
    
    % OA
    sumCorrect = trace(errorMatrix);
    OA = 1/N*sumCorrect;
    
    % Producers accuracy and users accuracy - Dynamic
    sumRows = zeros(numClasses,1);
    sumColumns = zeros(numClasses,1);
    PA = zeros(numClasses,1);
    UA = zeros(numClasses,1);
    for i = 1:numClasses
        sumRows(i) = sum( errorMatrix(i,:) );
        sumColumns(i) = sum( errorMatrix(:,i) );
    end
    
    for i = 1:numClasses
        if sumColumns(i) > 0
            PA(i) = errorMatrix(i,i)/sumColumns(i);
        else
            PA(i) = 0;
        end
        if sumRows(i) > 0
            UA(i) = errorMatrix(i,i)/sumRows(i);
        else
            UA(i) = 0;
        end
    end

    khat = (N*sumCorrect - sum(sumRows.*sumColumns ) ) / (N^2 - sum(sumRows.*sumColumns) );
    
    % Save errors for cross validation
    crossValOA(iteration) = OA;
    crossValPA(:,iteration) = PA;
    crossValUA(:,iteration) = UA;
    crossValk(iteration) = khat;
    crossValErrorMatrix(:,:,iteration) = errorMatrix;
end
% Find average of cross validation errors and save it - Dynamic
averageErrorMatrix = zeros(numClasses,numClasses);
averageOA = sum( crossValOA(:) ) / k;
averagek = sum( crossValk(:)) / k;
for i = 1:numClasses
    averagePA(i) = sum( crossValPA(i,:) ) /k;
    averageUA(i) = sum( crossValUA(i,:) ) /k;
end
for foldIdx = 1:k
    averageErrorMatrix = averageErrorMatrix + crossValErrorMatrix(:,:,foldIdx);
end
averageErrorMatrix = averageErrorMatrix/k;

% Display results
fprintf('\n=== OPTIMAL MODEL RESULTS ===\n');
fprintf('Average Overall Accuracy: %.4f (%.2f%%)\n', averageOA, averageOA*100);
fprintf('Average Kappa: %.4f\n', averagek);
fprintf('Classes: %d\n', numClasses);
fprintf('Features used: %d\n', numOfFeatures);
fprintf('Radius: %.1f\n', ra);

disp("End of script");