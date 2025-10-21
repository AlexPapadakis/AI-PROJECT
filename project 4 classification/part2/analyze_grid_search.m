%% Analyze Grid Search Results
clc; clear;

fprintf('Loading grid search results...\n');
if ~exist('grid_search_results.mat', 'file')
    error('Run main2ndpart.m first');
end
load('grid_search_results.mat');

fprintf('\n=== Grid Search Summary ===\n');
fprintf('Features tested: [%s]\n', sprintf('%d ', numOfFeatures));
fprintf('Radius values: [%s]\n', sprintf('%.1f ', clusterRadius));
fprintf('Total combinations: %d\n', length(numOfFeatures) * length(clusterRadius));
fprintf('Time: %.1f minutes\n', elapsed_time/60);

fprintf('\n=== Best Parameters ===\n');
fprintf('Features: %d\n', bestFeatures);
fprintf('Radius: %.1f\n', bestRadius);
fprintf('OA: %.4f (%.2f%%)\n', bestOA, bestOA*100);
fprintf('MSE: %.4f\n', min(grid_MSEs(:)));
fprintf('Rules: %.0f\n', bestRules);

fprintf('\n=== Results Matrix (MSE) ===\n');
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

fprintf('\n=== Results Matrix (OA) ===\n');
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

fprintf('\n=== Results Matrix (Rules) ===\n');
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

%% Plots

% Plot 1: MSE vs Rules
figure(1);
clf;
scatter(reshape(grid_numOfRules, 1, []), reshape(grid_MSEs, 1, []), 100, 'red', 'filled');
hold on;
minMSE = min(grid_MSEs(:));
yline(minMSE, '--', sprintf('Best MSE = %.4f', minMSE), 'LineWidth', 2, 'Color', 'blue');
xline(bestRules, '--', sprintf('Best Rules = %.0f', bestRules), 'LineWidth', 2, 'Color', 'green');
xlabel('Number of Rules');
ylabel('MSE');
title('MSE vs Number of Rules');
grid on;
hold off;

% Plot 2: MSE vs Features
figure(2);
clf;
meanMSEByFeatures = mean(grid_MSEs, 2);
plot(numOfFeatures, meanMSEByFeatures, 'o-', 'LineWidth', 2, 'MarkerSize', 10);
xlabel('Number of Features');
ylabel('Mean MSE');
title('Error vs Number of Features');
grid on;

% Plot 3: MSE vs Radius  
figure(3);
clf;
meanMSEByRadius = mean(grid_MSEs, 1);
plot(clusterRadius, meanMSEByRadius, 's-', 'LineWidth', 2, 'MarkerSize', 10);
xlabel('Cluster Radius');
ylabel('Mean MSE');
title('Error vs Radius');
grid on;

% Plot 4: 3D Surface
figure(4);
clf;
[Xgrid, Ygrid] = meshgrid(clusterRadius, numOfFeatures);
surf(Xgrid, Ygrid, grid_MSEs);
xlabel('Cluster Radius');
ylabel('Number of Features');
zlabel('MSE');
title('3D Error Surface');
colorbar;
shading interp;

% Plot 5: MSE Heatmap
figure(5);
clf;
imagesc(clusterRadius, numOfFeatures, grid_MSEs);
colorbar;
xlabel('Cluster Radius');
ylabel('Number of Features');
title('MSE Heatmap');
set(gca, 'YDir', 'normal');
colormap('hot');

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
xlabel('Cluster Radius');
ylabel('Number of Features');
title('Overall Accuracy Heatmap');
set(gca, 'YDir', 'normal');
colormap('jet');

for i = 1:length(numOfFeatures)
    for j = 1:length(clusterRadius)
        text(clusterRadius(j), numOfFeatures(i), sprintf('%.1f%%', grid_OAs(i, j)*100), ...
            'HorizontalAlignment', 'center', 'Color', 'white', 'FontWeight', 'bold');
    end
end

fprintf('\nPlots generated (Figures 1-6)\n');
