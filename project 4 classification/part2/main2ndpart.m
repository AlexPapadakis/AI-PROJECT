clc;clear;

% Part 2: Grid Search for TSK Classification
disp("Start of script");

% Load dataset
data = csvread('Epileptic Seizure Recognition.csv',1,1);
dataTarget = data(:, end);

fprintf('Dataset Info:\n');
fprintf('  Samples: %d\n', size(data,1));
fprintf('  Features: %d\n', size(data,2)-1);
fprintf('  Classes: %d\n', length(unique(dataTarget)));

% Split dataset (60-20-20)
fprintf('\nSplitting dataset...\n');
[trainData, validationData, testData] = split_scale(data, 1);
trainTarget = trainData(:, end);
validationTarget = validationData(:, end);
testTarget = testData(:, end);

% Clamp to [0,1]
trainData = max(min(trainData, 1), 0);
validationData = max(min(validationData, 1), 0);
testData = max(min(testData, 1), 0);

fprintf('  Training set: %d samples\n', size(trainData,1));
fprintf('  Validation set: %d samples\n', size(validationData,1));
fprintf('  Test set: %d samples\n', size(testData,1));

numClasses = length(unique(dataTarget));
fprintf('  Number of classes: %d\n', numClasses);

% Grid search parameters
fprintf('\n=== GRID SEARCH ===\n');
numOfFeatures = [4 8 12 16];
clusterRadius = [0.3 0.4 0.5 0.6 0.9];
numOfFolds = 5;

% Feature selection
fprintf('Running ReliefF...\n');
[importanceIndexes, importanceWeights] = relieff(trainData(:,1:end-1), trainTarget, 10, 'method', 'classification');
fprintf('  Done\n');

outputMembershipFunctionType = 'constant';

% Initialize results
grid_OAs = zeros(length(numOfFeatures), length(clusterRadius));
grid_numOfRules = zeros(length(numOfFeatures), length(clusterRadius));
grid_MSEs = zeros(length(numOfFeatures), length(clusterRadius));

fprintf('\nStarting grid search over %d combinations...\n', length(numOfFeatures) * length(clusterRadius));
tic;

combination = 0;
for i = numOfFeatures
    % Select top features
    temp_train = [trainData(:, importanceIndexes(1:i)) trainTarget];
    temp_val = [validationData(:, importanceIndexes(1:i)) validationTarget];
    temp_test = [testData(:, importanceIndexes(1:i)) testTarget];
    
    for j = clusterRadius
        combination = combination + 1;
        fprintf('\n--- Combination %d/%d: Features=%d, Radius=%.1f ---\n', ...
            combination, length(numOfFeatures)*length(clusterRadius), i, j);
        
        cvObj = cvpartition(temp_train(:, end), 'KFold', numOfFolds);
        
        OAs = zeros(numOfFolds, 1);
        cvMSE = zeros(numOfFolds, 1);
        rulesNum_k = zeros(numOfFolds, 1);
        
        % 5-fold cross-validation loop
        for k = 1:numOfFolds
            fprintf('  CV Fold %d/%d... ', k, numOfFolds);
            
            % Get training and test indices for this fold
            trainIdx = training(cvObj, k);
            testIdx = test(cvObj, k);
            
            % Split: 80% for training (including ANFIS validation), 20% for testing
            cv_trainValData = temp_train(trainIdx, :);
            cv_testData = temp_train(testIdx, :);
            cv_testTarget = cv_testData(:, end);
            
            % Further split training data: 80% train, 20% validation (for ANFIS early stopping)
            nTrainVal = size(cv_trainValData, 1);
            nTrain = round(0.8 * nTrainVal);
            
            % Stratified split for ANFIS validation
            cv_partition = cvpartition(cv_trainValData(:, end), 'HoldOut', 0.2);
            cv_trainData = cv_trainValData(training(cv_partition), :);
            cv_validationData = cv_trainValData(test(cv_partition), :);
            
            % ================================================================
            % Clustering Per Class (as per ΜΕΡΟΣ2.txt note 2)
            % ================================================================
            clusters = cell(numClasses, 1);
            sigmas = cell(numClasses, 1);
            num_rules = 0;
            
            for classIdx = 1:numClasses
                classData = cv_trainData(cv_trainData(:, end) == classIdx, :);
                if size(classData, 1) > 1
                    [clusters{classIdx}, sigmas{classIdx}] = subclust(classData, j);
                    num_rules = num_rules + size(clusters{classIdx}, 1);
                else
                    clusters{classIdx} = [];
                    sigmas{classIdx} = [];
                end
            end
            
            if num_rules == 0
                error('No clusters generated with radius %.1f', j);
            end
            
            if k == 1
                fprintf('[%d rules] ', num_rules);
            end
            
            % Build FIS
            initialFIS = sugfis('Name', 'TSK_GridSearch');
            
            % Add inputs
            for n = 1:size(cv_trainData, 2) - 1
                initialFIS = addInput(initialFIS, [0, 1], 'Name', sprintf("in%d", n));
                
                for classIdx = 1:numClasses
                    if ~isempty(clusters{classIdx})
                        for m = 1:size(clusters{classIdx}, 1)
                            initialFIS = addMF(initialFIS, sprintf("in%d", n), 'gaussmf', ...
                                [sigmas{classIdx}(n), clusters{classIdx}(m, n)]);
                        end
                    end
                end
            end
            
            initialFIS = addOutput(initialFIS, [1, numClasses], 'Name', 'out1');
            
            % Add output MFs
            params = [];
            for classIdx = 1:numClasses
                if ~isempty(clusters{classIdx})
                    classParams = repmat(classIdx, 1, size(clusters{classIdx}, 1));
                    params = [params, classParams];
                end
            end
            
            for n = 1:num_rules
                initialFIS = addMF(initialFIS, 'out1', outputMembershipFunctionType, params(n));
            end
            
            % Add rules
            rulesList = zeros(num_rules, size(cv_trainData, 2));
            for n = 1:num_rules
                rulesList(n, :) = n;
            end
            rulesList = [rulesList, ones(num_rules, 2)];
            initialFIS = addrule(initialFIS, rulesList);
            
            % Train with ANFIS
            ANFISoptions = anfisOptions;
            ANFISoptions.InitialFIS = initialFIS;
            ANFISoptions.EpochNumber = 100;
            ANFISoptions.DisplayANFISInformation = 0;
            ANFISoptions.DisplayErrorValues = 0;
            ANFISoptions.DisplayStepSize = 0;
            ANFISoptions.DisplayFinalResults = 0;
            ANFISoptions.ValidationData = cv_validationData;
            ANFISoptions.OptimizationMethod = 1;
            
            [~, ~, ~, validationFIS, ~] = anfis(cv_trainData, ANFISoptions);
            
            % Evaluate on test fold
            y_hat = evalfis(validationFIS, cv_testData(:, 1:end-1));
            y_hat = round(y_hat);
            y_hat = max(min(y_hat, numClasses), 1);
            
            % Calculate metrics
            N = size(cv_testData, 1);
            OA = sum(y_hat == cv_testTarget) / N;
            OAs(k) = OA;
            rulesNum_k(k) = length(validationFIS.Rules);
            cvMSE(k) = mse(y_hat, cv_testTarget);
            
            fprintf('OA=%.4f\n', OA);
        end
        
        % Store results
        grid_numOfRules(find(numOfFeatures == i), find(clusterRadius == j)) = mean(rulesNum_k);
        grid_MSEs(find(numOfFeatures == i), find(clusterRadius == j)) = mean(cvMSE);
        grid_OAs(find(numOfFeatures == i), find(clusterRadius == j)) = mean(OAs);
        
        fprintf('  Average OA: %.4f (%.1f%%), MSE: %.4f, Rules: %.0f\n', ...
            mean(OAs), mean(OAs)*100, mean(cvMSE), mean(rulesNum_k));
    end
end

elapsed_time = toc;
fprintf('\nGrid search completed in %.1f minutes\n', elapsed_time/60);

% Save results
fprintf('\n=== Saving results ===\n');

[minMSE, minIdx] = min(grid_MSEs(:));
[bestFeatIdx, bestRadIdx] = ind2sub(size(grid_MSEs), minIdx);
bestFeatures = numOfFeatures(bestFeatIdx);
bestRadius = clusterRadius(bestRadIdx);
bestOA = grid_OAs(bestFeatIdx, bestRadIdx);
bestRules = grid_numOfRules(bestFeatIdx, bestRadIdx);

fprintf('Best combination:\n');
fprintf('  Features: %d\n', bestFeatures);
fprintf('  Radius: %.1f\n', bestRadius);
fprintf('  Average OA: %.4f (%.2f%%)\n', bestOA, bestOA*100);
fprintf('  Average MSE: %.4f\n', minMSE);
fprintf('  Average Rules: %.0f\n', bestRules);

save('grid_search_results.mat', ...
    'grid_OAs', 'grid_MSEs', 'grid_numOfRules', ...
    'numOfFeatures', 'clusterRadius', ...
    'bestFeatures', 'bestRadius', 'bestOA', 'bestRules', ...
    'importanceIndexes', 'importanceWeights', ...
    'trainData', 'validationData', 'testData', ...
    'trainTarget', 'validationTarget', 'testTarget', ...
    'numClasses', 'elapsed_time');

fprintf('\nResults saved to grid_search_results.mat\n');
fprintf('Next: Run analyze_grid_search.m\n');
disp("End of script");