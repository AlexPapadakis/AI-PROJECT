models = {TSK_model_1, TSK_model_2, TSK_model_3, TSK_model_4};
trnErrs = {trnErr1, trnErr2, trnErr3, trnErr4};
chkErrs = {chkErr1, chkErr2, chkErr3, chkErr4};

R2 = zeros(4,1); RMSE = zeros(4,1); NMSE = zeros(4,1); NDEI = zeros(4,1);
ytrue = chkData(:,end);

for m = 1:4
    model = models{m};
    trnErr = trnErrs{m};
    chkErr = chkErrs{m};

    % === Membership functions ===
    for i = 1:5
        figure;
        [x, mf] = plotmf(model,'input',i);
        plot(x,mf,'LineWidth',2);
        title(['Membership functions of input ', num2str(i), ' - Model ', num2str(m)]);
        xlabel(['Input ', num2str(i)]);
        ylabel('Membership grade');
    end

    % === Learning curve ===
    figure;
    plot(1:length(trnErr), trnErr, 'b-', 'LineWidth',2); hold on;
    plot(1:length(chkErr), chkErr, 'r--', 'LineWidth',2);
    legend('Training Error','Validation Error');
    xlabel('Epochs'); ylabel('Error');
    title(['Learning curve - Model ', num2str(m)]);
    grid on;

    % === Predictions ===
    ypred = evalfis(chkData(:,1:end-1), model);

    % === Metrics ===
    SSres = sum((ytrue - ypred).^2);
    SStot = sum((ytrue - mean(ytrue)).^2);
    R2(m) = 1 - SSres/SStot;
    RMSE(m) = sqrt(mean((ytrue - ypred).^2));
    NMSE(m) = 1 - R2(m);
    NDEI(m) = sqrt(NMSE(m));

    % === Scatter plot ===
    figure;
    scatter(ytrue, ypred,'filled'); hold on;
    plot([min(ytrue) max(ytrue)], [min(ytrue) max(ytrue)], 'r--','LineWidth',2);
    xlabel('True'); ylabel('Predicted');
    title(['True vs Predicted - Model ', num2str(m)]);
    grid on;

    % === Residuals ===
    figure;
    plot(ytrue - ypred,'o-');
    xlabel('Sample index'); ylabel('Error');
    title(['Prediction errors (residuals) - Model ', num2str(m)]);
    grid on;
end

% === Summary table ===
ModelNames = {'Model1'; 'Model2'; 'Model3'; 'Model4'};
resultsTbl = table(ModelNames, R2, RMSE, NMSE, NDEI);
disp(resultsTbl)
