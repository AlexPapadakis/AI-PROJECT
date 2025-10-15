%% DISPLAY RESULTS - TSK CLASSIFICATION PROJECT
% Script για εμφάνιση αποτελεσμάτων σύμφωνα με την εκφώνηση
% Χρειάζεται να έχει τρέξει πρώτα το main.m

clc;

fprintf('Έλεγχος για υπάρχοντα αποτελέσματα...\n\n');

% Έλεγχος αν υπάρχουν τα αποτελέσματα
if exist('errorMatrices', 'var') && exist('OAMatrix', 'var') && exist('PAMatrix', 'var')
    
    fprintf('Αποτελέσματα βρέθηκαν. Εμφάνιση σύμφωνα με την εκφώνηση...\n\n');
    
    % Εκτέλεση της εμφάνισης αποτελεσμάτων
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
    
    fprintf('Κλάσεις: 1 = Survived ≥5 years, 2 = Died <5 years\n\n');
    
    % Εμφάνιση αποτελεσμάτων για κάθε μοντέλο
    for i = 1:4
        fprintf('─────────────────────────────────────────────────────────────────\n');
        fprintf('ΜΟΝΤΕΛΟ %d: %s\n', i, modelNames{i});
        fprintf('─────────────────────────────────────────────────────────────────\n\n');
        
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
    
    fprintf('ΕΜΦΑΝΙΣΗ ΑΠΟΤΕΛΕΣΜΑΤΩΝ ΟΛΟΚΛΗΡΩΘΗΚΕ\n');
    fprintf('Όλοι οι πίνακες σφαλμάτων και δείκτες απόδοσης εμφανίστηκαν\n');
    fprintf('σύμφωνα με τις απαιτήσεις της εκφώνησης.\n\n');
    
else
    fprintf('Δεν βρέθηκαν αποτελέσματα.\n');
    fprintf('Πρέπει πρώτα να εκτελέσετε το main.m script.\n');
    fprintf('Ή να φορτώσετε αποθηκευμένα αποτελέσματα με: load(''results.mat'')\n\n');
    
    fprintf('Για εκτέλεση του πλήρους project τρέξτε:\n');
    fprintf('>> main\n\n');
end