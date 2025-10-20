%% TRAIN FINAL MODEL
% This script trains the final TSK model using the best parameters
% from grid search and saves the trained model

clc; clear;

fprintf('=================================================================\n');
fprintf('       TRAINING FINAL TSK MODEL WITH OPTIMAL PARAMETERS\n');
fprintf('=================================================================\n\n');

%% Load Grid Search Results
if ~exist('grid_search_results.mat', 'file')
    error('Grid search results not found! Run main2ndpart.m first.');
end

fprintf('Loading optimal parameters from grid search...\n');
load('grid_search_results.mat');

fprintf('\n=== DATASET INFO ===\n');
fprintf('  Training samples: %d\n', size(trainData, 1));
fprintf('  Validation samples: %d\n', size(validationData, 1));
fprintf('  Test samples: %d\n', size(testData, 1));
fprintf('  Features (total): %d\n', size(trainData, 2) - 1);
fprintf('  Classes: %d\n', numClasses);

fprintf('\n=== OPTIMAL PARAMETERS (from grid search) ===\n');
fprintf('  Number of features: %d\n', bestFeatures);
fprintf('  Cluster radius: %.1f\n', bestRadius);
fprintf('  Expected rules: ~%.0f\n', bestRules);
fprintf('  Expected OA: %.2f%%\n', bestOA * 100);

%% Apply Feature Selection
fprintf('\n=== FEATURE SELECTION ===\n');
selectedFeatures = importanceIndexes(1:bestFeatures);
fprintf('Selected features: [%s]\n', sprintf('%d ', selectedFeatures));

% Apply to all datasets
trainDataFS = [trainData(:, selectedFeatures), trainTarget];
valDataFS = [validationData(:, selectedFeatures), validationTarget];
testDataFS = [testData(:, selectedFeatures), testTarget];

%% Subtractive Clustering (Class-Dependent)
fprintf('\n=== SUBTRACTIVE CLUSTERING ===\n');
fprintf('Applying class-dependent clustering with radius %.1f...\n', bestRadius);

clusters = cell(numClasses, 1);
sigmas = cell(numClasses, 1);
num_rules = 0;

for classIdx = 1:numClasses
    classData = trainDataFS(trainDataFS(:, end) == classIdx, :);
    if size(classData, 1) > 1
        [clusters{classIdx}, sigmas{classIdx}] = subclust(classData, bestRadius);
        num_rules = num_rules + size(clusters{classIdx}, 1);
        fprintf('  Class %d: %d clusters\n', classIdx, size(clusters{classIdx}, 1));
    else
        clusters{classIdx} = [];
        sigmas{classIdx} = [];
        fprintf('  Class %d: No clusters (insufficient samples)\n', classIdx);
    end
end

fprintf('  Total rules: %d\n', num_rules);

if num_rules == 0
    error('No clusters generated! Cannot build FIS.');
end

%% Build TSK Fuzzy Inference System
fprintf('\n=== BUILDING TSK FIS ===\n');

fis = sugfis('Name', 'TSK_Final_Model');

% Add inputs
names_in = cell(1, bestFeatures);
for i = 1:bestFeatures
    names_in{i} = sprintf('in%d', i);
    fis = addInput(fis, [0 1], 'Name', names_in{i});
end

% Add output (using actual class labels 1 to numClasses)
fis = addOutput(fis, [1 numClasses], 'Name', 'output');

% Add input membership functions (Gaussian from clustering)
for i = 1:bestFeatures
    for classIdx = 1:numClasses
        if ~isempty(clusters{classIdx})
            for j = 1:size(clusters{classIdx}, 1)
                fis = addMF(fis, names_in{i}, 'gaussmf', ...
                    [sigmas{classIdx}(i), clusters{classIdx}(j, i)]);
            end
        end
    end
end

% Add output membership functions (constant type with actual class values)
params = [];
for classIdx = 1:numClasses
    if ~isempty(clusters{classIdx})
        classParams = repmat(classIdx, 1, size(clusters{classIdx}, 1));
        params = [params, classParams];
    end
end

for i = 1:num_rules
    fis = addMF(fis, 'output', 'constant', params(i));
end

% Add rules
ruleList = zeros(num_rules, bestFeatures + 1);
for i = 1:num_rules
    ruleList(i, :) = i;
end
ruleList = [ruleList, ones(num_rules, 2)];
fis = addRule(fis, ruleList);

fprintf('  FIS structure created:\n');
fprintf('    Inputs: %d\n', length(fis.Inputs));
fprintf('    Rules: %d\n', length(fis.Rules));
fprintf('    Output type: %s\n', fis.Outputs.MembershipFunctions(1).Type);

%% Train with ANFIS
fprintf('\n=== ANFIS TRAINING ===\n');
fprintf('Training with 200 epochs (this may take several minutes)...\n\n');

tic;
anfisOpt = anfisOptions;
anfisOpt.InitialFIS = fis;
anfisOpt.EpochNumber = 200;
anfisOpt.ValidationData = valDataFS;
anfisOpt.OptimizationMethod = 1;  % Hybrid method

[trainedFIS, trainError, ~, valFIS, valError] = anfis(trainDataFS, anfisOpt);
trainingTime = toc;

fprintf('\n=== TRAINING COMPLETE ===\n');
fprintf('  Training time: %.1f seconds\n', trainingTime);
fprintf('  Final training RMSE: %.6f\n', trainError(end));
fprintf('  Best validation RMSE: %.6f\n', min(valError));
fprintf('  Final validation RMSE: %.6f\n', valError(end));

%% Save Trained Model
fprintf('\n=== SAVING MODEL ===\n');

% Save everything needed for evaluation
save('trained_final_model.mat', ...
    'valFIS', 'trainedFIS', ...
    'selectedFeatures', 'bestFeatures', 'bestRadius', ...
    'trainDataFS', 'valDataFS', 'testDataFS', ...
    'trainTarget', 'validationTarget', 'testTarget', ...
    'trainError', 'valError', 'trainingTime', ...
    'num_rules', 'numClasses', ...
    'clusters', 'sigmas');

fprintf('Model saved to: trained_final_model.mat\n');

% Quick validation check
fprintf('\n=== QUICK VALIDATION CHECK ===\n');
Y_pred_val = evalfis(valFIS, valDataFS(:, 1:end-1));
Y_pred_val = round(Y_pred_val);
Y_pred_val = max(1, min(numClasses, Y_pred_val));
valOA = sum(Y_pred_val == validationTarget) / length(validationTarget);
fprintf('  Validation set OA: %.4f (%.2f%%)\n', valOA, valOA * 100);

fprintf('\n=================================================================\n');
fprintf('Final model training complete!\n');
fprintf('Model saved successfully\n');
fprintf('\nNext step: Run evaluate_final_model.m to evaluate on test set\n');
fprintf('=================================================================\n');
