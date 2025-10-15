%% QUICK TEST# Test with single parameter combination
numOfFeatures = 15;
ra = 0.5;
k = 5; % 5-fold CV as requestedingle parameter combination to verify the code works
% Run this first before the full grid search

clc; clear;

fprintf('=== QUICK TEST OF TSK FUZZY SYSTEM ===\n');

% Load and preprocess data
data = csvread('Epileptic Seizure Recognition.csv',1,1);
data = normaliseData(data);

fprintf('Dataset loaded: %d samples, %d features\n', size(data,1), size(data,2)-1);

% Test with single parameter combination
numOfFeatures = 15;
ra = 0.5;
k = 5; % 5-fold CV as requested

fprintf('Testing: %d features, radius %.1f, %d-fold CV\n', numOfFeatures, ra, k);

% Get number of classes
numClasses = length(unique(data(:,end)));
fprintf('Number of classes: %d\n', numClasses);

tic;
crossValOA = zeros(k,1);
cvPart = cvpartition(data(:,end),'KFold',k,'Stratify',true);

% Quick cross-validation test
for iteration = 1:k
    fprintf('  Fold %d/%d...', iteration, k);
    
    trnDataTemp = data(training(cvPart,iteration),:);
    tstData = data(test(cvPart,iteration),:);
    
    % Simple split for speed
    cvPartitionTrn = cvpartition(trnDataTemp(:,end),'KFold',4,'Stratify',true);
    trnData = trnDataTemp(training(cvPartitionTrn,1), :);
    chkData = trnDataTemp(test(cvPartitionTrn,1), :);
    
    % Feature selection
    [idx,weights] = relieff(trnData(:,1:end-1), trnData(:,end), 5);
    
    trnDataFS = [trnData(:, idx(1:numOfFeatures)), trnData(:, end)];
    chkDataFS = [chkData(:, idx(1:numOfFeatures)), chkData(:, end)];
    tstDataFS = [tstData(:, idx(1:numOfFeatures)), tstData(:, end)];
    
    % Class-dependent clustering
    clusters = cell(numClasses, 1);
    sigmas = cell(numClasses, 1);
    num_rules = 0;
    
    for classIdx = 1:numClasses
        classData = trnDataFS(trnDataFS(:,end) == classIdx, :);
        if size(classData, 1) > 0
            [clusters{classIdx}, sigmas{classIdx}] = subclust(classData, ra);
            num_rules = num_rules + size(clusters{classIdx}, 1);
        end
    end
    
    % Build FIS using modern syntax
    fis2 = sugfis('Name', 'TestTSK');
    
    % Add inputs
    names_in = {};
    for i = 1:size(trnDataFS,2)-1
        name = sprintf('input%d', i);
        names_in{i} = name;
        fis2 = addInput(fis2, [0 1], 'Name', name);
    end
    
    % Add output
    fis2 = addOutput(fis2, [0 1], 'Name', 'output');
    
    % Add membership functions for inputs
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
    
    % Add output membership functions
    params = [];
    for classIdx = 1:numClasses
        if ~isempty(clusters{classIdx})
            classValue = (classIdx - 1) / (numClasses - 1);
            classParams = repmat(classValue, 1, size(clusters{classIdx}, 1));
            params = [params, classParams];
        end
    end
    
    for i = 1:num_rules
        fis2 = addMF(fis2, 'output', 'constant', params(i));
    end
    
    % Add rules
    ruleList = zeros(num_rules, size(trnDataFS,2));
    for i = 1:num_rules
        ruleList(i,:) = i;
    end
    ruleList = [ruleList, ones(num_rules, 2)];
    fis2 = addRule(fis2, ruleList);
    
    % Train ANFIS with reduced epochs
    [trnFis,trnError,~,valFis,valError]=anfis(trnDataFS,fis2,[30 0 0.01 0.9 1.1],[],chkDataFS);
    
    % Test
    Y = evalfis(valFis, tstDataFS(:,1:end-1));
    Y = round(Y * (numClasses - 1)) + 1;
    Y = max(1, min(numClasses, Y));
    
    % Calculate accuracy
    diff = tstDataFS(:,end) - Y;
    Acc = (length(diff) - nnz(diff)) / length(Y);
    crossValOA(iteration) = Acc;
    
    fprintf(' Accuracy: %.3f\n', Acc);
end

avgAccuracy = mean(crossValOA);
testTime = toc;

fprintf('\n=== TEST RESULTS ===\n');
fprintf('Average accuracy: %.4f (%.2f%%)\n', avgAccuracy, avgAccuracy*100);
fprintf('Total time: %.1f seconds\n', testTime);
fprintf('Time per fold: %.1f seconds\n', testTime/k);

% Estimate full grid search time
gridCombinations = 3 * 3; % 3x3 grid
estimatedTime = testTime * gridCombinations;
fprintf('\nEstimated time for 3x3 grid: %.1f minutes\n', estimatedTime/60);

fprintf('\n✓ Quick test completed successfully!\n');
fprintf('Now you can run main2ndpart.m with confidence.\n');