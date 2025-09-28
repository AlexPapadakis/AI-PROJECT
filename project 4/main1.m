clc; clear;

% Load data - Split data
data = load('haberman.data');
preproc = 1;
[trnData,chkData,tstData] = split_scale(data,preproc);

% Radius
radius = [0.2 0.8];

% Performance metrics
OAMatrix = zeros(4,1);
PAMatrix = zeros(4,2);
UAMatrix = zeros(4,2);
kMatrix  = zeros(4,1);
errorMatrices = zeros(2,2,4);

% Input names
names_in = ["in1","in2","in3"];

for r = 1:2
    % --- Class-dependent clustering ---
    [c1,sig1] = subclust(trnData(trnData(:,end) == 1,:),radius(r));
    [c2,sig2] = subclust(trnData(trnData(:,end) == 2,:),radius(r));
    num_rules = size(c1,1) + size(c2,1);

    % Build Sugeno FIS
    fis2 = sugfis("Name","FIS_SC");

    % Add inputs
    for i = 1:size(trnData,2)-1
        fis2 = addInput(fis2,[0 1],"Name",names_in(i));
    end

    % Add output
    fis2 = addOutput(fis2,[0 1],"Name","out1");

    % Add input membership functions
    for i = 1:size(trnData,2)-1
        for j = 1:size(c1,1)
            fis2 = addMF(fis2,names_in(i),"gaussmf", ...
                [sig1(i) c1(j,i)],"Name",sprintf("c1_in%d_%d",i,j));
        end
        for j = 1:size(c2,1)
            fis2 = addMF(fis2,names_in(i),"gaussmf", ...
                [sig2(i) c2(j,i)],"Name",sprintf("c2_in%d_%d",i,j));
        end
    end

    % Add output membership functions (constants for Sugeno)
    params = [zeros(1,size(c1,1)) ones(1,size(c2,1))];
    for i = 1:num_rules
        fis2 = addMF(fis2,"out1","constant",params(i), ...
            "Name",sprintf("outMF_%d",i));
    end

    % Construct rules (indices must align with MF order)
    ruleList = zeros(num_rules,size(trnData,2));
    for i = 1:size(ruleList,1)
        ruleList(i,:) = i;  % sequential assignment
    end
    ruleList = [ruleList ones(num_rules,2)];  % append weights + AND=1
    fis2 = addRule(fis2,ruleList);

    % --- Train & Evaluate ANFIS ---
    [trnFis,trnError,~,valFis,valError] = anfis(trnData,fis2,[100 0 0.01 0.9 1.1],[],chkData);

    figure(2*(r-1)+1);
    plot([trnError valError],'LineWidth',2); grid on;
    legend("Training Error","Validation Error");
    xlabel("# of Epochs"); ylabel("Error");
    title(sprintf("Class dependent subtractive clustering training error, radius = %.2f",radius(r)));

    Y = evalfis(valFis,tstData(:,1:end-1));
    Y = round(Y);

    % Membership function plots
    for i = 1:size(trnData,2)-1
        figure(1000*r+i);
        plotmf(valFis,"input",i);
        title(sprintf("TSK class dependent r=%.2f, MFs after training for input %d",radius(r),i));
    end

    % Error/confusion matrix
    classes = [1 2];
    errorMatrix = zeros(2,2);
    N = size(tstData,1);
    for i = 1:2
        for j = 1:2
            errorMatrix(i,j) = numel(intersect(find(Y==classes(i)),find(tstData(:,end)==classes(j))));
        end
    end

    % Metrics
    sumCorrect = trace(errorMatrix);
    OA = sumCorrect/N;
    sumRows = sum(errorMatrix,2);
    sumColumns = sum(errorMatrix,1)';
    PA = diag(errorMatrix)./sumColumns;
    UA = diag(errorMatrix)./sumRows;
    k = (N*sumCorrect - sum(sumRows.*sumColumns))/(N^2 - sum(sumRows.*sumColumns));

    % Save metrics
    OAMatrix(2*r-1) = OA;
    PAMatrix(2*r-1,:) = PA;
    UAMatrix(2*r-1,:) = UA;
    kMatrix(2*r-1) = k;
    errorMatrices(:,:,2*r-1) = errorMatrix;

    % --- Class-independent clustering ---
    fis1 = genfis2(trnData(:,1:end-1),trnData(:,end),radius(r));
    [trnFis,trnError,~,valFis,valError] = anfis(trnData,fis1,[100 0 0.01 0.9 1.1],[],chkData);

    figure(2*(r-1)+2);
    plot([trnError valError],'LineWidth',2); grid on;
    legend("Training Error","Validation Error");
    xlabel("# of Epochs"); ylabel("Error");
    title(sprintf("Class independent subtractive clustering training error, radius = %.2f",radius(r)));

    Y = evalfis(valFis,tstData(:,1:end-1));
    Y = round(Y);

    for i = 1:size(trnData,2)-1
        figure(1000*r+100+i);
        plotmf(valFis,"input",i);
        title(sprintf("TSK class independent r=%.2f, MFs after training for input %d",radius(r),i));
    end

    errorMatrix = zeros(2);
    for i = 1:2
        for j = 1:2
            errorMatrix(i,j) = numel(intersect(find(Y==classes(i)),find(tstData(:,end)==classes(j))));
        end
    end

    sumCorrect = trace(errorMatrix);
    OA = sumCorrect/size(tstData,1);
    sumRows = sum(errorMatrix,2);
    sumColumns = sum(errorMatrix,1)';
    PA = diag(errorMatrix)./sumColumns;
    UA = diag(errorMatrix)./sumRows;
    k = (N*sumCorrect - sum(sumRows.*sumColumns))/(N^2 - sum(sumRows.*sumColumns));

    OAMatrix(2*r) = OA;
    PAMatrix(2*r,:) = PA;
    UAMatrix(2*r,:) = UA;
    kMatrix(2*r) = k;
    errorMatrices(:,:,2*r) = errorMatrix;
end

% Display results
disp(OAMatrix);
disp(PAMatrix);
disp(UAMatrix);
disp(kMatrix);
for i = 1:4
    disp(errorMatrices(:,:,i));
end
