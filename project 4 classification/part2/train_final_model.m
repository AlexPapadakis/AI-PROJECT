%% Train Final Model
clc; clear;

fprintf('Loading optimal parameters...\n');
if ~exist('grid_search_results.mat', 'file')
    error('Run main2ndpart.m first');
end
load('grid_search_results.mat');

fprintf('\n=== Dataset ===\n');
fprintf('Training: %d samples\n', size(trainData, 1));
fprintf('Validation: %d samples\n', size(validationData, 1));
fprintf('Test: %d samples\n', size(testData, 1));
fprintf('Features: %d\n', size(trainData, 2) - 1);
fprintf('Classes: %d\n', numClasses);

fprintf('\n=== Optimal Parameters ===\n');
fprintf('Features: %d\n', bestFeatures);
fprintf('Radius: %.1f\n', bestRadius);
fprintf('Expected rules: ~%.0f\n', bestRules);
fprintf('Expected OA: %.2f%%\n', bestOA * 100);

% Feature selection
fprintf('\n=== Feature Selection ===\n');
selectedFeatures = importanceIndexes(1:bestFeatures);
fprintf('Selected: [%s]\n', sprintf('%d ', selectedFeatures));

trainDataFS = [trainData(:, selectedFeatures), trainTarget];
valDataFS = [validationData(:, selectedFeatures), validationTarget];
testDataFS = [testData(:, selectedFeatures), testTarget];

% Clustering
fprintf('\n=== Clustering ===\n');
fprintf('Applying subtractive clustering...\n');

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
    end
end

fprintf('Total rules: %d\n', num_rules);

if num_rules == 0
    error('No clusters generated');
end

% Build FIS
fprintf('\n=== Building FIS ===\n');
fis = sugfis('Name', 'TSK_Final');

for i = 1:bestFeatures
    fis = addInput(fis, [0 1], 'Name', sprintf('in%d', i));
end

fis = addOutput(fis, [1 numClasses], 'Name', 'output');

for i = 1:bestFeatures
    for classIdx = 1:numClasses
        if ~isempty(clusters{classIdx})
            for j = 1:size(clusters{classIdx}, 1)
                fis = addMF(fis, sprintf('in%d', i), 'gaussmf', ...
                    [sigmas{classIdx}(i), clusters{classIdx}(j, i)]);
            end
        end
    end
end

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

ruleList = zeros(num_rules, bestFeatures + 1);
for i = 1:num_rules
    ruleList(i, :) = i;
end
ruleList = [ruleList, ones(num_rules, 2)];
fis = addRule(fis, ruleList);

fprintf('FIS created with %d rules\n', num_rules);

% Train
fprintf('\n=== Training ===\n');
fprintf('Training with 200 epochs...\n\n');

tic;
anfisOpt = anfisOptions;
anfisOpt.InitialFIS = fis;
anfisOpt.EpochNumber = 200;
anfisOpt.ValidationData = valDataFS;
anfisOpt.OptimizationMethod = 1;

[trainedFIS, trainError, ~, valFIS, valError] = anfis(trainDataFS, anfisOpt);
trainingTime = toc;

fprintf('\n=== Complete ===\n');
fprintf('Training time: %.1f seconds\n', trainingTime);
fprintf('Final training RMSE: %.6f\n', trainError(end));
fprintf('Best validation RMSE: %.6f\n', min(valError));
fprintf('Final validation RMSE: %.6f\n', valError(end));

% Save
fprintf('\n=== Saving ===\n');
save('trained_final_model.mat', ...
    'valFIS', 'trainedFIS', ...
    'selectedFeatures', 'bestFeatures', 'bestRadius', ...
    'trainDataFS', 'valDataFS', 'testDataFS', ...
    'trainTarget', 'validationTarget', 'testTarget', ...
    'trainError', 'valError', 'trainingTime', ...
    'num_rules', 'numClasses', ...
    'clusters', 'sigmas');

fprintf('Model saved\n');

% Quick check
Y_pred_val = evalfis(valFIS, valDataFS(:, 1:end-1));
Y_pred_val = round(Y_pred_val);
Y_pred_val = max(1, min(numClasses, Y_pred_val));
valOA = sum(Y_pred_val == validationTarget) / length(validationTarget);
fprintf('Validation OA: %.4f (%.2f%%)\n', valOA, valOA * 100);

fprintf('\nNext: Run evaluate_final_model.m\n');
