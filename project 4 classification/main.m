clc;clear;

% Load data - Split data
data=load('haberman.data');
preproc=1;
[trnData,chkData,tstData]=split_scale(data,preproc);

% Radius
radius = zeros(2,1);
radius = [0.2 0.8];

% Performance metrics for every model
OAMatrix = zeros(4,1);
PAMatrix = zeros(4,2);
UAMatrix = zeros(4,2);
kMatrix  = zeros(4,1);
errorMatrices = zeros(2,2,4);

% r is the index of the radius array
for r = 1:2
    % ANFIS - Scatter Partition - Clustering Per Class
    % Clustering Per Class
    [c1,sig1] = subclust(trnData(trnData(:,end) == 1,:),radius(r));
    [c2,sig2] = subclust(trnData(trnData(:,end) == 2,:),radius(r));
    num_rules = size(c1,1) + size(c2,1);
    
    % Build FIS From Scratch using modern syntax
    fis2 = sugfis('Name','FIS_SC');
    
    % Add Input-Output Variables using modern syntax
    names_in = {'in1','in2','in3'};
    for i = 1:size(trnData,2)-1
        fis2 = addInput(fis2,[0 1],'Name',names_in{i});
    end
    fis2 = addOutput(fis2,[0 1],'Name','out1');
    
    % Add Input Membership Functions using modern syntax
    for i = 1:size(trnData,2)-1
        for j=1:size(c1,1)
            fis2 = addMF(fis2,names_in{i},'gaussmf',[sig1(i) c1(j,i)]);
        end
        for j=1:size(c2,1)
            fis2 = addMF(fis2,names_in{i},'gaussmf',[sig2(i) c2(j,i)]);
        end
    end
    
    % Add Output Membership Functions using modern syntax
    params = [zeros(1,size(c1,1)) ones(1,size(c2,1))];
    for i = 1:num_rules
        fis2 = addMF(fis2,'out1','constant',params(i));
    end
    
    % Add FIS Rule Base using modern syntax
    ruleList = zeros(num_rules,size(trnData,2));
    for i = 1:size(ruleList,1)
        ruleList(i,:)=i;
    end
    ruleList = [ruleList ones(num_rules,2)];
    fis2 = addRule(fis2,ruleList);
    
    % Train & Evaluate ANFIS
    [trnFis,trnError,~,valFis,valError]=anfis(trnData,fis2,[100 0 0.01 0.9 1.1],[],chkData);
    figure(2*(r-1)+1);
    plot([trnError valError],'LineWidth',2); grid on;
    legend('Training Error','Validation Error');
    xlabel('# of Epochs');
    ylabel('Error');
    string = 'Class dependent subtractive clustering training error, radius =  ';
    num = num2str(radius(r));
    title1 = strcat(string,num);
    title(title1);
    Y=evalfis(valFis,tstData(:,1:end-1));
    Y=round(Y);
    diff=tstData(:,end)-Y;
    
    % Membership function plots using original syntax (still valid)
    for i = 1:size(trnData,2)-1
        num = int2str(i);
        figure(1000*r+i);
        plotmf(valFis,'input',i);
        radiusStr = num2str(radius(r));
        message = 'TSK class dependent r = ';
        message = strcat(message,radiusStr);
        message = strcat(message,', membership functions after training for input ' );
        message = strcat(message,num);
        title(message);   
    end
    
    classes = [1 2];
    errorMatrix = zeros(2,2);
    N = size(tstData,1);
    % Error matrix
    for i = 1:2
        for j = 1:2
            errorMatrix(i,j) = size( intersect( find( Y == classes(i) ) , find(tstData(:,end) == classes(j) ) ) ,1);
        end
    end
    
    % OA
    sumCorrect = trace(errorMatrix);
    OA = 1/N*sumCorrect;
    
    % Producers accuracy and users accuracy
    sumRows = zeros(2,1);
    sumColumns = zeros(2,1);
    PA = zeros(2,1);
    UA = zeros(2,1);
    for i = 1:2
        sumRows(i) = sum( errorMatrix(i,:) );
        sumColumns(i) = sum( errorMatrix(:,i) );
    end
    
    for i = 1:2
        PA(i) = errorMatrix(i,i)/sumColumns(i);
        UA(i) = errorMatrix(i,i)/sumRows(i);
    end

    k = (N*sumCorrect - sum(sumRows.*sumColumns ) ) / (N^2 - sum(sumRows.*sumColumns) );
    
    % Save performaces metrics
    OAMatrix(2*r-1) = OA;
    PAMatrix(2*r-1,:) = PA;
    UAMatrix(2*r-1,:) = UA;
    kMatrix(2*r-1) = k;
    errorMatrices(:,:,2*r-1) = errorMatrix;
    
    % Compare with Class-Independent Scatter Partition
    fis1=genfis2(trnData(:,1:end-1),trnData(:,end),radius(r));
    [trnFis,trnError,~,valFis,valError]=anfis(trnData,fis1,[100 0 0.01 0.9 1.1],[],chkData);
    figure(2*(r-1)+2);
    plot([trnError valError],'LineWidth',2); grid on;
    legend('Training Error','Validation Error');
    xlabel('# of Epochs');
    ylabel('Error');
    string = 'Class independent subtractive clustering training error, radius =  ';
    num = num2str(radius(r));
    title2 = strcat(string,num);
    title(title2);
    Y=evalfis(valFis,tstData(:,1:end-1));
    Y=round(Y);
    diff=tstData(:,end)-Y;
    
    % Membership function plots using original syntax (still valid)
    for i = 1:size(trnData,2)-1
        num = int2str(i);
        figure(1000*r+100+i);
        plotmf(valFis,'input',i);
        radiusStr = num2str(radius(r));
        message = 'TSK class independent r = ';
        message = strcat(message,radiusStr);
        message = strcat(message,', membership functions after training for input ' );
        message = strcat(message,num);
        title(message);   
    end
    
    classes = [1 2];
    errorMatrix = zeros(2);
    % Error matrix
    for i = 1:2
        for j = 1:2
            errorMatrix(i,j) = size( intersect( find( Y == classes(i) ) , find(tstData(:,end) == classes(j) ) ) ,1);
        end
    end

    % OA
    sumCorrect = trace(errorMatrix);
    OA = 1/size(tstData,1)*sumCorrect;
    
    % Producers accuracy and users accuracy
    sumRows = zeros(2,1);
    sumColumns = zeros(2,1);
    PA = zeros(2,1);
    UA = zeros(2,1);
    for i = 1:2
        sumRows(i) = sum( errorMatrix(i,:) );
        sumColumns(i) = sum( errorMatrix(:,i) );
    end
    
    for i = 1:2
        PA(i) = errorMatrix(i,i)/sumColumns(i);
        UA(i) = errorMatrix(i,i)/sumRows(i);
    end

    k = (N*sumCorrect - sum(sumRows.*sumColumns ) ) / (N^2 - sum(sumRows.*sumColumns) );
    
    % Save performaces metrics
    OAMatrix(2*r) = OA;
    PAMatrix(2*r,:) = PA;
    UAMatrix(2*r,:) = UA;
    kMatrix(2*r) = k;
    errorMatrices(:,:,2*r) = errorMatrix;
    
end

%% ΕΜΦΑΝΙΣΗ ΑΠΟΤΕΛΕΣΜΑΤΩΝ ΣΥΜΦΩΝΑ ΜΕ ΤΗΝ ΕΚΦΩΝΗΣΗ
fprintf('\n');
fprintf('=================================================================\n');
fprintf('          ΑΠΟΤΕΛΕΣΜΑΤΑ TSK ΜΟΝΤΕΛΩΝ - HABERMAN DATASET\n');
fprintf('=================================================================\n\n');

% Ονόματα μοντέλων
modelNames = {
    'Class-Dependent Clustering με radius = 0.2'
    'Class-Independent Clustering με radius = 0.2'
    'Class-Dependent Clustering με radius = 0.8'  
    'Class-Independent Clustering με radius = 0.8'
};

fprintf('Συνολικά δείγματα test set: %d\n', size(tstData,1));
fprintf('Κλάσεις: 1 = Survived >=5 years, 2 = Died <5 years\n\n');

% Εμφάνιση αποτελεσμάτων για κάθε μοντέλο
for i = 1:4
    fprintf('-----------------------------------------------------------------\n');
    fprintf('ΜΟΝΤΕΛΟ %d: %s\n', i, modelNames{i});
    fprintf('-----------------------------------------------------------------\n\n');
    
    % Error Matrix (Πίνακας Σφαλμάτων)
    fprintf('ΠΙΝΑΚΑΣ ΣΦΑΛΜΑΤΩΝ (Error Matrix):\n');
    fprintf('+-----------------+-----------------------------+---------+\n');
    fprintf('|                 |        Predicted Class      |         |\n');
    fprintf('|   True Class    +--------------+--------------+  Total  |\n');
    fprintf('|                 | Survived (1) |  Died (2)    |         |\n');
    fprintf('+-----------------+--------------+--------------+---------+\n');
    
    currentMatrix = errorMatrices(:,:,i);
    
    % Υπολογισμός συνόλων
    total_survived = currentMatrix(1,1) + currentMatrix(1,2);
    total_died = currentMatrix(2,1) + currentMatrix(2,2);
    total_pred_survived = currentMatrix(1,1) + currentMatrix(2,1);
    total_pred_died = currentMatrix(1,2) + currentMatrix(2,2);
    
    fprintf('| Survived (1)    |     %3d      |     %3d      |   %3d   |\n', ...
        currentMatrix(1,1), currentMatrix(1,2), total_survived);
    fprintf('| Died (2)        |     %3d      |     %3d      |   %3d   |\n', ...
        currentMatrix(2,1), currentMatrix(2,2), total_died);
    fprintf('+-----------------+--------------+--------------+---------+\n');
    fprintf('| Total           |     %3d      |     %3d      |   %3d   |\n', ...
        total_pred_survived, total_pred_died, total_survived + total_died);
    fprintf('+-----------------+--------------+--------------+---------+\n\n');
    
    % Δείκτες Απόδοσης
    fprintf('ΔΕΙΚΤΕΣ ΑΠΟΔΟΣΗΣ:\n');
    fprintf('Overall Accuracy (OA):           %.4f (%.2f%%)\n', ...
        OAMatrix(i), OAMatrix(i)*100);
    fprintf('Producers Accuracy (PA):\n');
    fprintf('  Class 1 (Survived): %.4f (%.2f%%)\n', ...
        PAMatrix(i,1), PAMatrix(i,1)*100);
    fprintf('  Class 2 (Died):     %.4f (%.2f%%)\n', ...
        PAMatrix(i,2), PAMatrix(i,2)*100);
    fprintf('Users Accuracy (UA):\n');
    fprintf('  Class 1 (Survived): %.4f (%.2f%%)\n', ...
        UAMatrix(i,1), UAMatrix(i,1)*100);
    fprintf('  Class 2 (Died):     %.4f (%.2f%%)\n', ...
        UAMatrix(i,2), UAMatrix(i,2)*100);
    fprintf('Kappa Statistic (κ̂):             %.4f\n', kMatrix(i));
    
    % Ανάλυση σφαλμάτων
    TP = currentMatrix(1,1);  % True Positives
    TN = currentMatrix(2,2);  % True Negatives
    FP = currentMatrix(2,1);  % False Positives
    FN = currentMatrix(1,2);  % False Negatives
    
    fprintf('\nΑΝΑΛΥΣΗ ΣΦΑΛΜΑΤΩΝ:\n');
    fprintf('True Positives (TP):  %3d (Σωστά ως Survived)\n', TP);
    fprintf('True Negatives (TN):  %3d (Σωστά ως Died)\n', TN);
    fprintf('False Positives (FP): %3d (Died -> Predicted Survived)\n', FP);
    fprintf('False Negatives (FN): %3d (Survived -> Predicted Died)\n', FN);
    
    if FN > FP
        fprintf('ΣΗΜΕΙΩΣΗ: Περισσότερα False Negatives - Επικίνδυνο για ιατρικές εφαρμογές\n');
    elseif FP > FN  
        fprintf('ΣΗΜΕΙΩΣΗ: Περισσότερα False Positives - Συντηρητικό μοντέλο\n');
    else
        fprintf('ΣΗΜΕΙΩΣΗ: Ισορροπημένα σφάλματα\n');
    end
    
    fprintf('\n');
end

% Συγκριτική ανάλυση
fprintf('=================================================================\n');
fprintf('                        ΣΥΓΚΡΙΤΙΚΗ ΑΝΑΛΥΣΗ\n');
fprintf('=================================================================\n\n');

% Κατάταξη μοντέλων
[sortedOA, sortIdx] = sort(OAMatrix, 'descend');
fprintf('ΚΑΤΑΤΑΞΗ ΜΟΝΤΕΛΩΝ (κατά Overall Accuracy):\n');
for i = 1:4
    fprintf('%d. %s\n', i, modelNames{sortIdx(i)});
    fprintf('   OA = %.4f (%.2f%%)\n', sortedOA(i), sortedOA(i)*100);
end

% Ανάλυση επίδρασης παραμέτρων
fprintf('\nΑΝΑΛΥΣΗ ΕΠΙΔΡΑΣΗΣ ΠΑΡΑΜΕΤΡΩΝ:\n\n');

% Επίδραση radius
smallRadius_avg = mean([OAMatrix(1), OAMatrix(2)]);  % r=0.2
largeRadius_avg = mean([OAMatrix(3), OAMatrix(4)]);  % r=0.8

fprintf('ΕΠΙΔΡΑΣΗ RADIUS:\n');
fprintf('Μικρό radius (0.2):  Μέσος OA = %.4f (%.2f%%)\n', ...
    smallRadius_avg, smallRadius_avg*100);
fprintf('Μεγάλο radius (0.8): Μέσος OA = %.4f (%.2f%%)\n', ...
    largeRadius_avg, largeRadius_avg*100);
fprintf('Διαφορά: %.4f (%.2f%%)\n', ...
    abs(largeRadius_avg - smallRadius_avg), abs(largeRadius_avg - smallRadius_avg)*100);

if largeRadius_avg > smallRadius_avg
    fprintf('Συμπέρασμα: Μεγάλο radius είναι καλύτερο - Λιγότεροι κανόνες, καλύτερη γενίκευση\n');
else
    fprintf('Συμπέρασμα: Μικρό radius είναι καλύτερο - Περισσότεροι κανόνες, καλύτερη κάλυψη\n');
end

% Επίδραση clustering strategy  
classDep_avg = mean([OAMatrix(1), OAMatrix(3)]);    % Class-dependent
classIndep_avg = mean([OAMatrix(2), OAMatrix(4)]);  % Class-independent

fprintf('\nΕΠΙΔΡΑΣΗ CLUSTERING STRATEGY:\n');
fprintf('Class-Dependent:   Μέσος OA = %.4f (%.2f%%)\n', ...
    classDep_avg, classDep_avg*100);
fprintf('Class-Independent: Μέσος OA = %.4f (%.2f%%)\n', ...
    classIndep_avg, classIndep_avg*100);
fprintf('Διαφορά: %.4f (%.2f%%)\n', ...
    abs(classDep_avg - classIndep_avg), abs(classDep_avg - classIndep_avg)*100);

if classDep_avg > classIndep_avg
    fprintf('Συμπέρασμα: Class-Dependent είναι καλύτερο - Καθαρότερα clusters ανά κλάση\n');
else
    fprintf('Συμπέρασμα: Class-Independent είναι καλύτερο - Καλύτερη γενίκευση\n');
end

fprintf('\n=================================================================\n');
fprintf('Καλύτερο μοντέλο: %s με OA = %.4f (%.2f%%)\n', ...
    modelNames{sortIdx(1)}, sortedOA(1), sortedOA(1)*100);
fprintf('=================================================================\n\n');