
load('project3B.mat');


%% ----------- Evaluate Final Model on Check Set -----------
Xchk_sel = Xchk(:,featIdx);
Ypred_chk = evalfis(Xchk_sel,trnFis);

RMSE = sqrt(mean((Ypred_chk - Ychk).^2));
NMSE = mean((Ypred_chk - Ychk).^2)/var(Ychk);
NDEI = sqrt(NMSE);
R2 = 1 - sum((Ychk-Ypred_chk).^2)/sum((Ychk-mean(Ychk)).^2);

fprintf('\nBest features: %d, Cluster radius: %.2f\n',bestNF,bestRA);
fprintf('RMSE: %.4f, NMSE: %.4f, NDEI: %.4f, R2: %.4f\n',RMSE,NMSE,NDEI,R2);

T = table(RMSE,NMSE,NDEI,R2);
disp('Performance metrics:'); disp(T);

%% ----------- Plots -----------

% 1. Predictions vs Actual
figure;
plot(Ychk,Ypred_chk,'o'); hold on;
plot([min(Ychk) max(Ychk)],[min(Ychk) max(Ychk)],'r--','LineWidth',2);
xlabel('Actual Output'); ylabel('Predicted Output');
title('Predictions vs Actual (Check set)'); grid on;

% 2. Learning curve
figure;
plot(trnError,'-o','LineWidth',1.5);
xlabel('Epoch'); ylabel('RMSE');
title('Learning Curve'); grid on;

% 3. Example fuzzy sets (input 1)
inName = 1;
figure;
subplot(1,2,1);
[xmf, ymf] = plotmf(fis,'input',inName);
plot(xmf,ymf); title('Initial MFs (before training)');
subplot(1,2,2);
[xmf2, ymf2] = plotmf(trnFis,'input',inName);
plot(xmf2,ymf2); title('Final MFs (after training)');

% 4. Error vs Features
figure;
gscatter(results(:,1),results(:,3),results(:,2));
xlabel('Number of Features'); ylabel('Mean CV RMSE');
title('Error vs Features (colored by Cluster Radius)'); grid on;

% 5. Error vs Cluster Radius
figure;
hold on;
for nf = unique(results(:,1))'
    idx = results(:,1)==nf;
    plot(results(idx,2),results(idx,3),'-o','DisplayName',['Features=' num2str(nf)]);
end
xlabel('Cluster Radius'); ylabel('Mean CV RMSE');
title('Error vs Cluster Radius'); legend show; grid on;

% 6. Error vs Number of Rules
figure;
scatter(results(:,4),results(:,3),'filled');
xlabel('Number of Rules'); ylabel('Mean CV RMSE');
title('Error vs Number of Rules'); grid on;

