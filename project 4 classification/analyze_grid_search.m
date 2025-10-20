%% ANALYZE GRID SEARCH RESULTS
% This script loads and visualizes the grid search results from main2ndpart.m
% Creates all required plots as per ΜΕΡΟΣ2.txt

clc; clear;

fprintf('=================================================================\n');
fprintf('          GRID SEARCH RESULTS ANALYSIS\n');
fprintf('=================================================================\n\n');

%% Load Results
if ~exist('grid_search_results.mat', 'file')
    error('Grid search results not found! Run main2ndpart.m first.');
end

fprintf('Loading grid search results...\n');
load('grid_search_results.mat');

%% Display Summary
fprintf('\n=== GRID SEARCH SUMMARY ===\n');
fprintf('Search space:\n');
fprintf('  Features tested: [%s]\n', sprintf('%d ', numOfFeatures));
fprintf('  Radius values: [%s]\n', sprintf('%.1f ', clusterRadius));
fprintf('  Total combinations: %d\n', length(numOfFeatures) * length(clusterRadius));
fprintf('  Computation time: %.1f minutes\n', elapsed_time/60);

fprintf('\n=== BEST PARAMETERS ===\n');
fprintf('  Features: %d\n', bestFeatures);
fprintf('  Radius: %.1f\n', bestRadius);
fprintf('  Average OA: %.4f (%.2f%%)\n', bestOA, bestOA*100);
fprintf('  Average MSE: %.4f\n', min(grid_MSEs(:)));
fprintf('  Average Rules: %.0f\n', bestRules);

%% Display Full Results Matrices

fprintf('\n=== FULL RESULTS MATRIX (MSE) ===\n');
fprintf('%-10s', 'Feat\\Rad');
for rad = clusterRadius
    fprintf('%10.1f', rad);
end
fprintf('\n');
for idx = 1:length(numOfFeatures)
    fprintf('%-10d', numOfFeatures(idx));
    for jdx = 1:length(clusterRadius)
        fprintf('%10.4f', grid_MSEs(idx, jdx));
    end
    fprintf('\n');
end

fprintf('\n=== FULL RESULTS MATRIX (Overall Accuracy) ===\n');
fprintf('%-10s', 'Feat\\Rad');
for rad = clusterRadius
    fprintf('%10.1f', rad);
end
fprintf('\n');
for idx = 1:length(numOfFeatures)
    fprintf('%-10d', numOfFeatures(idx));
    for jdx = 1:length(clusterRadius)
        fprintf('%10.4f', grid_OAs(idx, jdx));
    end
    fprintf('\n');
end

fprintf('\n=== FULL RESULTS MATRIX (Number of Rules) ===\n');
fprintf('%-10s', 'Feat\\Rad');
for rad = clusterRadius
    fprintf('%10.1f', rad);
end
fprintf('\n');
for idx = 1:length(numOfFeatures)
    fprintf('%-10d', numOfFeatures(idx));
    for jdx = 1:length(clusterRadius)
        fprintf('%10.0f', grid_numOfRules(idx, jdx));
    end
    fprintf('\n');
end

%% Visualizations (as per ΜΕΡΟΣ2.txt requirements)

% Plot 1: MSE vs Number of Rules
figure(1);
clf;
scatter(reshape(grid_numOfRules, 1, []), reshape(grid_MSEs, 1, []), 100, 'red', 'filled');
hold on;
minMSE = min(grid_MSEs(:));
yline(minMSE, 'LineStyle', '--', 'Label', sprintf('Best MSE = %.4f', minMSE), ...
    'LabelHorizontalAlignment', 'center', 'LineWidth', 2, 'Color', 'blue');
xline(bestRules, 'LineStyle', '--', 'Label', sprintf('Best Rules = %.0f', bestRules), ...
    'LabelHorizontalAlignment', 'left', 'LineWidth', 2, 'Color', 'green');
xlabel('Number of Rules', 'FontSize', 12);
ylabel('MSE', 'FontSize', 12);
title('Grid Search: MSE vs Number of Rules', 'FontSize', 14);
grid on;
hold off;

% Plot 2: MSE vs Number of Features
figure(2);
clf;
meanMSEByFeatures = mean(grid_MSEs, 2);
plot(numOfFeatures, meanMSEByFeatures, 'o-', 'LineWidth', 2, 'MarkerSize', 10, 'Color', [0.8 0.2 0.2]);
xlabel('Number of Features', 'FontSize', 12);
ylabel('Mean MSE', 'FontSize', 12);
title('Grid Search: Error vs Number of Features', 'FontSize', 14);
grid on;

% Plot 3: MSE vs Radius  
figure(3);
clf;
meanMSEByRadius = mean(grid_MSEs, 1);
plot(clusterRadius, meanMSEByRadius, 's-', 'LineWidth', 2, 'MarkerSize', 10, 'Color', [0.2 0.2 0.8]);
xlabel('Cluster Radius', 'FontSize', 12);
ylabel('Mean MSE', 'FontSize', 12);
title('Grid Search: Error vs Radius', 'FontSize', 14);
grid on;

% Plot 4: 3D Surface
figure(4);
clf;
[Xgrid, Ygrid] = meshgrid(clusterRadius, numOfFeatures);
surf(Xgrid, Ygrid, grid_MSEs);
xlabel('Cluster Radius', 'FontSize', 12);
ylabel('Number of Features', 'FontSize', 12);
zlabel('MSE', 'FontSize', 12);
title('Grid Search: 3D Error Surface', 'FontSize', 14);
colorbar;
shading interp;

% Plot 5: Heatmap (MSE)
figure(5);
clf;
imagesc(clusterRadius, numOfFeatures, grid_MSEs);
colorbar;
xlabel('Cluster Radius', 'FontSize', 12);
ylabel('Number of Features', 'FontSize', 12);
title('Grid Search: MSE Heatmap', 'FontSize', 14);
set(gca, 'YDir', 'normal');
colormap('hot');

% Add text annotations with values
for i = 1:length(numOfFeatures)
    for j = 1:length(clusterRadius)
        text(clusterRadius(j), numOfFeatures(i), sprintf('%.3f', grid_MSEs(i, j)), ...
            'HorizontalAlignment', 'center', 'Color', 'white', 'FontWeight', 'bold');
    end
end

% Plot 6: OA Heatmap
figure(6);
clf;
imagesc(clusterRadius, numOfFeatures, grid_OAs);
colorbar;
xlabel('Cluster Radius', 'FontSize', 12);
ylabel('Number of Features', 'FontSize', 12);
title('Grid Search: Overall Accuracy Heatmap', 'FontSize', 14);
set(gca, 'YDir', 'normal');
colormap('jet');

% Add text annotations with percentages
for i = 1:length(numOfFeatures)
    for j = 1:length(clusterRadius)
        text(clusterRadius(j), numOfFeatures(i), sprintf('%.1f%%', grid_OAs(i, j)*100), ...
            'HorizontalAlignment', 'center', 'Color', 'white', 'FontWeight', 'bold');
    end
end

%% Summary and Recommendations
fprintf('\n=== ANALYSIS AND RECOMMENDATIONS ===\n');

% Find worst and best
[maxMSE, maxIdx] = max(grid_MSEs(:));
[maxFeatIdx, maxRadIdx] = ind2sub(size(grid_MSEs), maxIdx);
worstFeatures = numOfFeatures(maxFeatIdx);
worstRadius = clusterRadius(maxRadIdx);

fprintf('\nWorst combination:\n');
fprintf('  Features: %d, Radius: %.1f\n', worstFeatures, worstRadius);
fprintf('  MSE: %.4f, OA: %.4f (%.2f%%)\n', maxMSE, grid_OAs(maxFeatIdx, maxRadIdx), grid_OAs(maxFeatIdx, maxRadIdx)*100);

fprintf('\nPerformance improvement:\n');
fprintf('  MSE reduction: %.2f%% (from %.4f to %.4f)\n', ...
    (maxMSE - minMSE)/maxMSE*100, maxMSE, minMSE);
fprintf('  OA improvement: %.2f%% points\n', ...
    (bestOA - grid_OAs(maxFeatIdx, maxRadIdx))*100);

fprintf('\n=================================================================\n');
fprintf('Analysis complete!\n');
fprintf('All plots generated (Figures 1-6)\n');
fprintf('\nNext step: Run train_final_model.m\n');
fprintf('=================================================================\n');
