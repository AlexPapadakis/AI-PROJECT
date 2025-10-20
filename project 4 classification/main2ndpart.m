clc;clear;

% ============================================================================
% PART 2: HIGH DIMENSIONAL CLASSIFICATION WITH GRID SEARCH
% Based on ΜΕΡΟΣ2.txt instructions and reference implementation
% ============================================================================
% Requirements followed:
% - 60-20-20 train-validation-test split (done once at start)
% - 5-fold cross-validation for parameter selection
% - ReliefF feature selection
% - Subtractive clustering per class (class-dependent)
% - Grid search over features and radius
% - Constant output membership functions
% ============================================================================

disp("Start of script");

% Import dataset
data = csvread('Epileptic Seizure Recognition.csv',1,1);
dataTarget = data(:, end);

fprintf('Dataset Info:\n');
fprintf('  Samples: %d\n', size(data,1));
fprintf('  Features: %d\n', size(data,2)-1);
fprintf('  Classes: %d\n', length(unique(dataTarget)));

% ============================================================================
% STEP 1: Dataset split 60-20-20 and pre-processing (as per ΜΕΡΟΣ2.txt)
% ============================================================================
fprintf('\nSplitting dataset (60%% train, 20%% validation, 20%% test)...\n');

% Use split_scale function or manual stratified split
[trainData, validationData, testData] = split_scale(data, 1);
trainTarget = trainData(:, end);
validationTarget = validationData(:, end);
testTarget = testData(:, end);

% Ensure data is in [0,1] range
trainData = max(min(trainData, 1), 0);
validationData = max(min(validationData, 1), 0);
testData = max(min(testData, 1), 0);

fprintf('  Training set: %d samples\n', size(trainData,1));
fprintf('  Validation set: %d samples\n', size(validationData,1));
fprintf('  Test set: %d samples\n', size(testData,1));

% Get number of classes
numClasses = length(unique(dataTarget));
fprintf('  Number of classes: %d\n', numClasses);

% ============================================================================
% STEP 2: Grid Search Algorithm for optimal parameters
% ============================================================================
fprintf('\n=== GRID SEARCH FOR OPTIMAL PARAMETERS ===\n');

% Define grid search parameters
numOfFeatures = [4 7 10 12];
clusterRadius = [0.3 0.5 0.7];

% Number of folds for cross-validation (as per ΜΕΡΟΣ2.txt: 5-fold)
numOfFolds = 5;

% Feature selection using ReliefF with k nearest neighbors
fprintf('Running ReliefF feature selection on training data...\n');
numOfNearestNeighbors = 10;
[importanceIndexes, importanceWeights] = relieff(trainData(:,1:end-1), trainTarget, numOfNearestNeighbors, 'method', 'classification');
fprintf('  Feature selection completed\n');

% Output type: constant (as per ΜΕΡΟΣ2.txt note 1)
outputMembershipFunctionType = 'constant';

% Initialize result matrices
grid_OAs = zeros(length(numOfFeatures), length(clusterRadius));
grid_numOfRules = zeros(length(numOfFeatures), length(clusterRadius));
grid_MSEs = zeros(length(numOfFeatures), length(clusterRadius));

fprintf('\nStarting grid search over %d combinations...\n', length(numOfFeatures) * length(clusterRadius));
tic;

fprintf('\nStarting grid search over %d combinations...\n', length(numOfFeatures) * length(clusterRadius));
tic;

% Grid search loops
combination = 0;
for i = numOfFeatures
    % Select features for this iteration
    temp_train = [trainData(:, importanceIndexes(1:i)) trainTarget];
    temp_val = [validationData(:, importanceIndexes(1:i)) validationTarget];
    temp_test = [testData(:, importanceIndexes(1:i)) testTarget];
    
    for j = clusterRadius
        combination = combination + 1;
        fprintf('\n--- Combination %d/%d: Features=%d, Radius=%.1f ---\n', ...
            combination, length(numOfFeatures)*length(clusterRadius), i, j);
        
        % Define cvpartition for 5-fold CV (stratified as per ΜΕΡΟΣ2.txt)
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
                if size(classData, 1) > 1  % Need at least 2 samples for clustering
                    [clusters{classIdx}, sigmas{classIdx}] = subclust(classData, j);
                    num_rules = num_rules + size(clusters{classIdx}, 1);
                else
                    clusters{classIdx} = [];
                    sigmas{classIdx} = [];
                end
            end
            
            % Safety check: ensure we have at least some rules
            if num_rules == 0
                error('No clusters generated! Radius %.1f might be too small.', j);
            end
            
            % Debug: print rules per class for first fold only
            if k == 1
                fprintf('[%d rules: ', num_rules);
                for classIdx = 1:numClasses
                    if ~isempty(clusters{classIdx})
                        fprintf('C%d=%d ', classIdx, size(clusters{classIdx}, 1));
                    end
                end
                fprintf('] ');
            end
            
            % ================================================================
            % Build TSK FIS from scratch
            % ================================================================
            initialFIS = sugfis('Name', 'TSK_GridSearch');
            
            % Add inputs
            for n = 1:size(cv_trainData, 2) - 1
                initialFIS = addInput(initialFIS, [0, 1], 'Name', sprintf("in%d", n));
                
                % Add input membership functions from all class clusters
                for classIdx = 1:numClasses
                    if ~isempty(clusters{classIdx})
                        for m = 1:size(clusters{classIdx}, 1)
                            initialFIS = addMF(initialFIS, sprintf("in%d", n), 'gaussmf', ...
                                [sigmas{classIdx}(n), clusters{classIdx}(m, n)]);
                        end
                    end
                end
            end
            
            % Add output (using class labels 1-5 directly, not normalized)
            initialFIS = addOutput(initialFIS, [1, numClasses], 'Name', 'out1');
            
            % Add output membership functions (constant type)
            % Use actual class values: 1, 2, 3, 4, 5
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
            
            % Add rule base
            rulesList = zeros(num_rules, size(cv_trainData, 2));
            for n = 1:num_rules
                rulesList(n, :) = n;
            end
            rulesList = [rulesList, ones(num_rules, 2)];
            initialFIS = addrule(initialFIS, rulesList);
            
            % ================================================================
            % Train with ANFIS (increased epochs for better convergence)
            % ================================================================
            ANFISoptions = anfisOptions;
            ANFISoptions.InitialFIS = initialFIS;
            ANFISoptions.EpochNumber = 100;
            ANFISoptions.DisplayANFISInformation = 0;
            ANFISoptions.DisplayErrorValues = 0;
            ANFISoptions.DisplayStepSize = 0;
            ANFISoptions.DisplayFinalResults = 0;
            ANFISoptions.ValidationData = cv_validationData;
            ANFISoptions.OptimizationMethod = 1;  % Hybrid method (backprop + LSE)
            
            [~, ~, ~, validationFIS, ~] = anfis(cv_trainData, ANFISoptions);
            
            % Evaluate on the held-out TEST fold (NOT the validation data used for early stopping)
            y_hat = evalfis(validationFIS, cv_testData(:, 1:end-1));
            
            % Round to nearest integer class and clip to valid range [1, numClasses]
            y_hat = round(y_hat);
            y_hat = max(min(y_hat, numClasses), 1);
            
            % Calculate OA on test fold
            N = size(cv_testData, 1);
            OA = sum(y_hat == cv_testTarget) / N;
            OAs(k) = OA;
            
            % Save number of rules and MSE (on test fold)
            rulesNum_k(k) = length(validationFIS.Rules);
            cvMSE(k) = mse(y_hat, cv_testTarget);
            
            fprintf('OA=%.4f\n', OA);
        end
        
        % Store average results for this combination
        grid_numOfRules(find(numOfFeatures == i), find(clusterRadius == j)) = mean(rulesNum_k);
        grid_MSEs(find(numOfFeatures == i), find(clusterRadius == j)) = mean(cvMSE);
        grid_OAs(find(numOfFeatures == i), find(clusterRadius == j)) = mean(OAs);
        
        fprintf('  Average OA: %.4f (%.1f%%), MSE: %.4f, Rules: %.0f\n', ...
            mean(OAs), mean(OAs)*100, mean(cvMSE), mean(rulesNum_k));
    end
end

elapsed_time = toc;
fprintf('\nGrid search completed in %.1f minutes\n', elapsed_time/60);

elapsed_time = toc;
fprintf('\nGrid search completed in %.1f minutes\n', elapsed_time/60);

% ============================================================================
% STEP 3: Save Results
% ============================================================================
fprintf('\n=== SAVING GRID SEARCH RESULTS ===\n');

% Find best parameters
[minMSE, minIdx] = min(grid_MSEs(:));
[bestFeatIdx, bestRadIdx] = ind2sub(size(grid_MSEs), minIdx);
bestFeatures = numOfFeatures(bestFeatIdx);
bestRadius = clusterRadius(bestRadIdx);
bestOA = grid_OAs(bestFeatIdx, bestRadIdx);
bestRules = grid_numOfRules(bestFeatIdx, bestRadIdx);

fprintf('Best combination found:\n');
fprintf('  Features: %d\n', bestFeatures);
fprintf('  Radius: %.1f\n', bestRadius);
fprintf('  Average OA: %.4f (%.2f%%)\n', bestOA, bestOA*100);
fprintf('  Average MSE: %.4f\n', minMSE);
fprintf('  Average Rules: %.0f\n', bestRules);

% Save all results and data splits for later use
save('grid_search_results.mat', ...
    'grid_OAs', 'grid_MSEs', 'grid_numOfRules', ...
    'numOfFeatures', 'clusterRadius', ...
    'bestFeatures', 'bestRadius', 'bestOA', 'bestRules', ...
    'importanceIndexes', 'importanceWeights', ...
    'trainData', 'validationData', 'testData', ...
    'trainTarget', 'validationTarget', 'testTarget', ...
    'numClasses', 'elapsed_time');

fprintf('\n=================================================================\n');
fprintf('Grid search completed successfully!\n');
fprintf('Results saved to: grid_search_results.mat\n');
fprintf('\nNext steps:\n');
fprintf('  1. Run analyze_grid_search.m to visualize results\n');
fprintf('  2. Run train_final_model.m to train with best parameters\n');
fprintf('  3. Run evaluate_final_model.m to see final model performance\n');
fprintf('=================================================================\n');

disp("End of script");