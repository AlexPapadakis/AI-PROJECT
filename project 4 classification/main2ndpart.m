clc;clear;

disp("Start of script");
% Read data and normalise - Epileptic Seizure Recognition dataset
data = csvread('Epileptic Seizure Recognition.csv',1,1);
data = normaliseData(data);

% Dataset info
fprintf('Dataset Info:\n');
fprintf('Samples: %d\n', size(data,1));
fprintf('Features: %d\n', size(data,2)-1);
fprintf('Classes: %d\n', length(unique(data(:,end))));

% Grid search parameters - REDUCED FOR SPEED
a = 3;          % Reduced from 5 to 3
b = 3;          % Reduced from 7 to 3

% First param is num of features, second param is clusters ra
gridSearchParams = zeros(a,b,2);

gridSearchParams(1,:,1) = 15;   % Fewer combinations but still meaningful
gridSearchParams(2,:,1) = 25;
gridSearchParams(3,:,1) = 35;
gridSearchParams(:,1,2) = 0.3;  % Focus on middle range values
gridSearchParams(:,2,2) = 0.5;
gridSearchParams(:,3,2) = 0.7;

errors = zeros(a,b);

% Add progress saving
results_file = 'grid_search_progress.mat';
if exist(results_file, 'file')
    fprintf('Loading previous progress...\n');
    load(results_file, 'errors', 'completed_grid');
    start_w = find(any(completed_grid == 0, 2), 1);
    if isempty(start_w), start_w = a+1; end
else
    completed_grid = zeros(a,b);
    start_w = 1;
end

% Get number of classes dynamically
numClasses = length(unique(data(:,end)));
fprintf('Number of classes: %d\n', numClasses);

% k-fold cross validation is used - KEEPING 5-fold as requested
k = 5;

tic
% Grid search with progress tracking
total_combinations = a * b;
current_combination = 0;

for w = start_w:a
    for z = 1:b
        if completed_grid(w,z) == 1
            continue; % Skip if already completed
        end
        
        current_combination = current_combination + 1;
        fprintf('\n=== GRID SEARCH PROGRESS ===\n');
        fprintf('Combination %d/%d: Features=%d, Radius=%.1f\n', ...
            (w-1)*b + z, total_combinations, gridSearchParams(w,z,1), gridSearchParams(w,z,2));
        fprintf('Progress: %.1f%% complete\n', ((w-1)*b + z)/total_combinations * 100);
        
        tic_combination = tic;
        numOfFeatures = gridSearchParams(w,z,1);
        ra = gridSearchParams(w,z,2);
        
        crossValOA = zeros(k,1);
        cvPart = cvpartition(data(:,end),'KFold',k,'Stratify',true);

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
            
            % Train & Evaluate ANFIS
            [trnFis,trnError,~,valFis,valError]=anfis(trnDataFS,fis2,[50 0 0.01 0.9 1.1],[],chkDataFS);  % Reduced epochs from 100 to 50
            figure(1000);
            plot([trnError valError],'LineWidth',2); grid on;
            legend('Training Error','Validation Error');
            xlabel('# of Epochs');
            ylabel('Error');
            Y=evalfis(tstDataFS(:,1:end-1),valFis);
            
            % Convert output back to class labels
            Y = round(Y * (numClasses - 1)) + 1;  % Convert from [0,1] to class labels
            Y = max(1, min(numClasses, Y));  % Clamp to valid class range
            
            diff=tstDataFS(:,end)-Y;
            Acc=(length(diff)-nnz(diff))/length(Y)*100;
            
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
            
            % Save error for cross validation
            crossValOA(iteration) = OA
            
            
        end
        % Find average of cross validation errors and save it
        tempErrorOA = sum( crossValOA(:) ) / k;
        errors(w,z) = tempErrorOA;
        completed_grid(w,z) = 1;
        
        % Save progress after each combination
        save(results_file, 'errors', 'completed_grid', 'gridSearchParams');
        
        combination_time = toc(tic_combination);
        fprintf('Combination completed in %.1f minutes\n', combination_time/60);
        fprintf('Current error: %.4f\n', errors(w,z));
    end
end
toc

%% GRID SEARCH RESULTS VISUALIZATION
fprintf('\n=== GRID SEARCH RESULTS ===\n');
fprintf('Best error: %.4f\n', min(errors(:)));
[minError, idx] = min(errors(:));
[bestFeatIdx, bestRadIdx] = ind2sub(size(errors), idx);
bestFeatures = gridSearchParams(bestFeatIdx, bestRadIdx, 1);
bestRadius = gridSearchParams(bestFeatIdx, bestRadIdx, 2);
fprintf('Best parameters: %d features, radius %.1f\n', bestFeatures, bestRadius);

% Error vs Features plot
figure(1);
meanErrorsByFeatures = mean(errors, 2);
plot(gridSearchParams(:,1,1), meanErrorsByFeatures, 'o-', 'LineWidth', 2);
xlabel('Number of Features');
ylabel('Mean Cross-Validation Error');
title('Grid Search: Error vs Number of Features');
grid on;

% Error vs Radius plot  
figure(2);
meanErrorsByRadius = mean(errors, 1);
plot(gridSearchParams(1,:,2), meanErrorsByRadius, 's-', 'LineWidth', 2);
xlabel('Radius');
ylabel('Mean Cross-Validation Error');
title('Grid Search: Error vs Radius');
grid on;

% 3D Surface plot
figure(3);
[X, Y] = meshgrid(gridSearchParams(1,:,2), gridSearchParams(:,1,1));
surf(X, Y, errors);
xlabel('Radius');
ylabel('Number of Features');
zlabel('Cross-Validation Error');
title('Grid Search: 3D Error Surface');
colorbar;

% Heatmap
figure(4);
imagesc(gridSearchParams(1,:,2), gridSearchParams(:,1,1), errors);
colorbar;
xlabel('Radius');
ylabel('Number of Features');
title('Grid Search: Error Heatmap');
set(gca, 'YDir', 'normal');

errors
disp("End of script");