% Load dataset
data = load('airfoil_self_noise.dat'); % 1503 x 6

% Shuffle rows
rng(0); % reproducibility
idx = randperm(size(data,1));
data = data(idx,:);

% Split sizes
N = size(data,1);
Ntr   = round(0.6*N);
Nval  = round(0.2*N);
Nchk  = N - Ntr - Nval;

% Subsets
Dtr  = data(1:Ntr,:);                        % Training
Dval = data(Ntr+1:Ntr+Nval,:);               % Validation
Dchk = data(Ntr+Nval+1:end,:);               % Checking/Test

size(Dtr)   % should be ~902 x 6
size(Dval)  % should be ~301 x 6
size(Dchk)  % should be ~300 x 6

% Training, Validation, Checking sets
trainData = Dtr;
valData   = Dval;
chkData   = Dchk;


% 2 singleton (order 0)
fis1 = genfis1(trainData, 2, 'gbellmf', 'constant');

% 3 singleton (order 0)
fis2 = genfis1(trainData, 3, 'gbellmf', 'constant');

% 2 polynomial (order 1)
fis3 = genfis1(trainData, 2, 'gbellmf', 'linear');

% 3 polynomial (order 1)
fis4 = genfis1(trainData, 3, 'gbellmf', 'linear');


trainOpt = anfisOptions('InitialFIS', fis1, ...
                        'EpochNumber', 100, ...
                        'ValidationData', valData, ...
                        'DisplayANFISInformation', 0, ...
                        'DisplayErrorValues', 1, ...
                        'DisplayStepSize', 0, ...
                        'DisplayFinalResults', 1);

% Train Model 1 (2 Singleton)
trainOpt.InitialFIS = fis1;
[TSK_model_1, trnErr1, stepSize1, chkFIS1, chkErr1] = anfis(trainData, trainOpt);

% Train Model 2 (3 Singleton)
trainOpt.InitialFIS = fis2;
[TSK_model_2, trnErr2, stepSize2, chkFIS2, chkErr2] = anfis(trainData, trainOpt);

% Train Model 3 (2 Polynomial)
trainOpt.InitialFIS = fis3;
[TSK_model_3, trnErr3, stepSize3, chkFIS3, chkErr3] = anfis(trainData, trainOpt);

% Train Model 4 (3 Polynomial)
trainOpt.InitialFIS = fis4;
[TSK_model_4, trnErr4, stepSize4, chkFIS4, chkErr4] = anfis(trainData, trainOpt);


% Predictions on checking set
yhat1 = evalfis(TSK_model_1, chkData(:,1:end-1));
yhat2 = evalfis(TSK_model_2, chkData(:,1:end-1));
yhat3 = evalfis(TSK_model_3, chkData(:,1:end-1));
yhat4 = evalfis(TSK_model_4, chkData(:,1:end-1));

% Errors (RMSE for example)
rmse1 = sqrt(mean((chkData(:,end) - yhat1).^2));
rmse2 = sqrt(mean((chkData(:,end) - yhat2).^2));
rmse3 = sqrt(mean((chkData(:,end) - yhat3).^2));
rmse4 = sqrt(mean((chkData(:,end) - yhat4).^2));
