%% EVALUATE FINAL MODEL
% This script loads the trained model and evaluates it on the test set
% Generates all required visualizations and performance metrics

clc; clear;

fprintf('=================================================================\n');
fprintf('         FINAL MODEL EVALUATION AND VISUALIZATION\n');
fprintf('=================================================================\n\n');

%% Load Trained Model
if ~exist('trained_final_model.mat', 'file')
    error('Trained model not found! Run train_final_model.m first.');
end

fprintf('Loading trained model...\n');
load('trained_final_model.mat');

fprintf('\n=== MODEL INFO ===\n');
fprintf('  Number of features: %d\n', bestFeatures);
fprintf('  Cluster radius: %.1f\n', bestRadius);
fprintf('  Number of rules: %d\n', num_rules);
fprintf('  Training time: %.1f seconds\n', trainingTime);

%% Evaluate on Test Set
fprintf('\n=== TEST SET EVALUATION ===\n');

% Clamp inputs to [0,1] to avoid warnings
testInput = max(0, min(1, testDataFS(:, 1:end-1)));

% Predict
Y_pred = evalfis(valFIS, testInput);

% Convert to class labels (round to nearest integer class)
Y_pred = round(Y_pred);
Y_pred = max(1, min(numClasses, Y_pred));
Y_true = testTarget;

%% Calculate Performance Metrics

% Confusion Matrix
N = length(Y_true);
confusionMat = zeros(numClasses);

for i = 1:N
    confusionMat(Y_true(i), Y_pred(i)) = confusionMat(Y_true(i), Y_pred(i)) + 1;
end

% Overall Accuracy (OA)
OA = trace(confusionMat) / N;

% Producer's Accuracy (PA) and User's Accuracy (UA)
PA = zeros(numClasses, 1);
UA = zeros(numClasses, 1);

for i = 1:numClasses
    if sum(confusionMat(i, :)) > 0
        PA(i) = confusionMat(i, i) / sum(confusionMat(i, :));
    end
    if sum(confusionMat(:, i)) > 0
        UA(i) = confusionMat(i, i) / sum(confusionMat(:, i));
    end
end

% Kappa Statistic
Po = OA;
Pe = sum(sum(confusionMat, 1) .* sum(confusionMat, 2)') / N^2;
if Pe < 1
    kappa = (Po - Pe) / (1 - Pe);
else
    kappa = 0;
end

% Mean Squared Error
MSE = mse(Y_pred, Y_true);

%% Display Results
fprintf('\n=================================================================\n');
fprintf('                    PERFORMANCE METRICS\n');
fprintf('=================================================================\n\n');

fprintf('CONFUSION MATRIX:\n');
fprintf('%-12s', 'True\\Pred');
for i = 1:numClasses
    fprintf('%8s%d', 'Class', i);
end
fprintf('%10s\n', 'Total');

for i = 1:numClasses
    fprintf('Class %-6d', i);
    for j = 1:numClasses
        fprintf('%9.0f', confusionMat(i, j));
    end
    fprintf('%10.0f\n', sum(confusionMat(i, :)));
end

fprintf('%-12s', 'Total');
for j = 1:numClasses
    fprintf('%9.0f', sum(confusionMat(:, j)));
end
fprintf('%10.0f\n', N);

fprintf('\nOVERALL METRICS:\n');
fprintf('  Overall Accuracy (OA):  %.4f (%.2f%%)\n', OA, OA * 100);
fprintf('  Kappa Statistic:        %.4f\n', kappa);
fprintf('  Mean Squared Error:     %.4f\n', MSE);

fprintf('\nPER-CLASS METRICS:\n');
fprintf('%-10s %15s %15s\n', 'Class', 'PA (Recall)', 'UA (Precision)');
for i = 1:numClasses
    fprintf('Class %-4d  %8.4f (%5.1f%%)  %8.4f (%5.1f%%)\n', ...
        i, PA(i), PA(i)*100, UA(i), UA(i)*100);
end

%% Visualizations

% Figure 1: Confusion Matrix Heatmap
figure(1);
clf;
imagesc(confusionMat);
colormap('parula');
colorbar;
xlabel('Predicted Class', 'FontSize', 12);
ylabel('True Class', 'FontSize', 12);
title(sprintf('Confusion Matrix (OA = %.2f%%)', OA*100), 'FontSize', 14);
set(gca, 'XTick', 1:numClasses, 'YTick', 1:numClasses);

% Add text annotations
for i = 1:numClasses
    for j = 1:numClasses
        text(j, i, sprintf('%d', round(confusionMat(i, j))), ...
            'HorizontalAlignment', 'center', 'Color', 'white', ...
            'FontWeight', 'bold', 'FontSize', 10);
    end
end

% Figure 2: Training Learning Curves
figure(2);
clf;
epochs = 1:length(trainError);
plot(epochs, trainError, 'b-', 'LineWidth', 2, 'DisplayName', 'Training Error');
hold on;
plot(epochs, valError, 'r-', 'LineWidth', 2, 'DisplayName', 'Validation Error');
[minValError, minIdx] = min(valError);
plot(minIdx, minValError, 'r*', 'MarkerSize', 12, 'LineWidth', 2, ...
    'DisplayName', sprintf('Best Val Error (epoch %d)', minIdx));
xlabel('Epoch', 'FontSize', 12);
ylabel('RMSE', 'FontSize', 12);
title('ANFIS Learning Curves', 'FontSize', 14);
legend('Location', 'best');
grid on;
hold off;

% Figure 3: Predictions vs True Values
figure(3);
clf;
sampleIndices = 1:min(500, N);  % Plot first 500 samples
plot(sampleIndices, Y_true(sampleIndices), 'bo-', 'LineWidth', 1.5, 'DisplayName', 'True Class');
hold on;
plot(sampleIndices, Y_pred(sampleIndices), 'rx--', 'LineWidth', 1.5, 'DisplayName', 'Predicted Class');
xlabel('Sample Index', 'FontSize', 12);
ylabel('Class Label', 'FontSize', 12);
title('Predictions vs True Values (First 500 Samples)', 'FontSize', 14);
legend('Location', 'best');
grid on;
ylim([0.5, numClasses + 0.5]);
set(gca, 'YTick', 1:numClasses);
hold off;

% Figure 4: Per-Class Performance
figure(4);
clf;
x = 1:numClasses;
bar(x, [PA, UA]);
xlabel('Class', 'FontSize', 12);
ylabel('Accuracy', 'FontSize', 12);
title('Per-Class Performance Metrics', 'FontSize', 14);
legend({'Producer''s Accuracy (Recall)', 'User''s Accuracy (Precision)'}, 'Location', 'best');
grid on;
set(gca, 'XTick', 1:numClasses);

% Figure 5: Sample Membership Functions (First 3 inputs)
figure(5);
clf;
numInputsToShow = min(3, bestFeatures);
for i = 1:numInputsToShow
    subplot(numInputsToShow, 1, i);
    plotmf(valFIS, 'input', i);
    title(sprintf('Input %d Membership Functions', selectedFeatures(i)));
end

% Figure 6: Error Distribution
figure(6);
clf;
errors = Y_pred - Y_true;
histogram(errors, 'Normalization', 'probability', 'BinWidth', 1);
xlabel('Prediction Error', 'FontSize', 12);
ylabel('Probability', 'FontSize', 12);
title('Error Distribution', 'FontSize', 14);
grid on;
xline(0, 'r--', 'LineWidth', 2, 'Label', 'Perfect Prediction');

%% Rule Explosion Analysis
fprintf('\n=== RULE EXPLOSION ANALYSIS ===\n');

% Calculate theoretical rules for grid partitioning
rulesGrid2 = 2^bestFeatures;
rulesGrid3 = 3^bestFeatures;

fprintf('Theoretical rules for grid partitioning:\n');
fprintf('  2 MFs per input: %d rules\n', rulesGrid2);
fprintf('  3 MFs per input: %d rules\n', rulesGrid3);
fprintf('\nOur TSK model: %d rules\n', num_rules);
fprintf('Reduction factors:\n');
fprintf('  vs 2 MFs/input: %.1fx\n', rulesGrid2 / num_rules);
fprintf('  vs 3 MFs/input: %.1fx\n', rulesGrid3 / num_rules);

%% Active Rules Analysis
fprintf('\n=== ACTIVE RULES ANALYSIS ===\n');

% Sample a subset of test data and check rule activation
sampleSize = min(100, N);
sampleIdx = randperm(N, sampleSize);
sampleData = testInput(sampleIdx, :);

% Evaluate FIS and check which rules fire
firingStrengths = zeros(sampleSize, num_rules);
for i = 1:sampleSize
    % Get firing strength for each rule
    [~, ~, ~, firingStrength] = evalfis(valFIS, sampleData(i, :));
    if ~isempty(firingStrength)
        firingStrengths(i, :) = firingStrength;
    end
end

% Calculate average activation per rule
avgActivation = mean(firingStrengths, 1);
activeRules = sum(avgActivation > 0.01);  % Rules with >1% average activation

fprintf('Rule activation statistics:\n');
fprintf('  Total rules: %d\n', num_rules);
fprintf('  Active rules (>1%% avg): %d (%.1f%%)\n', activeRules, activeRules/num_rules*100);
fprintf('  Average activation: %.4f\n', mean(avgActivation));

%% Save Results
fprintf('\n=== SAVING RESULTS ===\n');

save('final_model_results.mat', ...
    'Y_pred', 'Y_true', 'confusionMat', ...
    'OA', 'PA', 'UA', 'kappa', 'MSE', ...
    'trainError', 'valError', ...
    'bestFeatures', 'bestRadius', 'num_rules', ...
    'selectedFeatures');

fprintf('Results saved to: final_model_results.mat\n');

%% Summary
fprintf('\n=================================================================\n');
fprintf('                    EVALUATION SUMMARY\n');
fprintf('=================================================================\n');
fprintf('Model Parameters:\n');
fprintf('  Features: %d (selected from %d)\n', bestFeatures, size(testData, 2) - 1);
fprintf('  Radius: %.1f\n', bestRadius);
fprintf('  Rules: %d\n', num_rules);
fprintf('\nPerformance:\n');
fprintf('  Overall Accuracy: %.2f%%\n', OA * 100);
fprintf('  Kappa: %.4f\n', kappa);
fprintf('  MSE: %.4f\n', MSE);
fprintf('\nEfficiency:\n');
fprintf('  Rule reduction (vs 2 MFs): %.1fx\n', rulesGrid2 / num_rules);
fprintf('  Active rules: %.1f%%\n', activeRules/num_rules*100);
fprintf('\n=================================================================\n');
fprintf('✓ All visualizations generated (Figures 1-6)\n');
fprintf('✓ Results saved\n');
fprintf('✓ Evaluation complete!\n');
fprintf('=================================================================\n');
