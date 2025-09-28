

% Create Mamdani-type FIS
fzpi = mamfis("Name", "FZPI");
% --- INPUT 1: e ∈ [-1, 1] ---
fzpi = addInput(fzpi, [-1 1], "Name", "e");
fzpi = addMF(fzpi, "e", "trimf", [-1.00 -1.00 -0.75], "Name", "NV");
fzpi = addMF(fzpi, "e", "trimf", [-1.00 -0.75 -0.50], "Name", "NL");
fzpi = addMF(fzpi, "e", "trimf", [-0.75 -0.50 -0.25], "Name", "NM");
fzpi = addMF(fzpi, "e", "trimf", [-0.50 -0.25  0.00], "Name", "NS");
fzpi = addMF(fzpi, "e", "trimf", [-0.25  0.00  0.25], "Name", "ZR");
fzpi = addMF(fzpi, "e", "trimf", [ 0.00  0.25  0.50], "Name", "PS");
fzpi = addMF(fzpi, "e", "trimf", [ 0.25  0.50  0.75], "Name", "PM");
fzpi = addMF(fzpi, "e", "trimf", [ 0.50  0.75  1.00], "Name", "PL");
fzpi = addMF(fzpi, "e", "trimf", [ 0.75  1.00  1.00], "Name", "PV");
% --- INPUT 2: de ∈ [-1, 1] ---
fzpi = addInput(fzpi, [-1 1], "Name", "de");
fzpi = addMF(fzpi, "de", "trimf", [-1.00 -1.00 -0.75], "Name", "NV");
fzpi = addMF(fzpi, "de", "trimf", [-1.00 -0.75 -0.50], "Name", "NL");
fzpi = addMF(fzpi, "de", "trimf", [-0.75 -0.50 -0.25], "Name", "NM");
fzpi = addMF(fzpi, "de", "trimf", [-0.50 -0.25  0.00], "Name", "NS");
fzpi = addMF(fzpi, "de", "trimf", [-0.25  0.00  0.25], "Name", "ZR");
fzpi = addMF(fzpi, "de", "trimf", [ 0.00  0.25  0.50], "Name", "PS");
fzpi = addMF(fzpi, "de", "trimf", [ 0.25  0.50  0.75], "Name", "PM");
fzpi = addMF(fzpi, "de", "trimf", [ 0.50  0.75  1.00], "Name", "PL");
fzpi = addMF(fzpi, "de", "trimf", [ 0.75  1.00  1.00], "Name", "PV");
% --- OUTPUT: du ∈ [-1, 1] ---
fzpi = addOutput(fzpi, [-1 1], "Name", "du");
fzpi = addMF(fzpi, "du", "trimf", [-1.00 -1.00 -0.66], "Name", "NL");
fzpi = addMF(fzpi, "du", "trimf", [-1.00 -0.66 -0.33], "Name", "NM");
fzpi = addMF(fzpi, "du", "trimf", [-0.66 -0.33  0.00], "Name", "NS");
fzpi = addMF(fzpi, "du", "trimf", [-0.33  0.00  0.33], "Name", "ZR");
fzpi = addMF(fzpi, "du", "trimf", [ 0.00  0.33  0.66], "Name", "PS");
fzpi = addMF(fzpi, "du", "trimf", [ 0.33  0.66  1.00], "Name", "PM");
fzpi = addMF(fzpi, "du", "trimf", [ 0.66  1.00  1.00], "Name", "PL");
% Define the output MF index map (9x9), values from 1 to 7 matching your output MFs
du_map = [
1 1 1 1 1 1 2 3 4
1 1 1 1 1 2 3 4 5
1 1 1 1 2 3 4 5 6
1 1 1 2 3 4 5 6 7
1 1 2 3 4 5 6 7 7
1 2 3 4 5 6 7 7 7
2 3 4 5 6 7 7 7 7
3 4 5 6 7 7 7 7 7
4 5 6 7 7 7 7 7 7


];


% Build rules as [e_index, de_index, du_index, weight, connection]
rules = [];
for e_idx = 1:9
 for de_idx = 1:9
   du_idx = du_map(e_idx, de_idx);
   rules = [rules; e_idx, de_idx, du_idx, 1, 1];
 end
end
% Add rules to fuzzy inference system
fzpi = addRule(fzpi, rules);
% --- PLOT CONTROL SURFACE ---
gensurf(fzpi, [1 2], 1);  % inputs: e (1), de (2); output: du (1)
title("FZPI Control Surface");
xlabel("e"); ylabel("de"); zlabel("du");
out11 = evalfis(fzpi, [1 1])







% --- Discretize the plant ---

% K was chosen from PI tuning
s = tf('s');
Gp = 25/((s+0.1)*(s+10));
K = 1.66;


Controlled_plant = Gp*K
Ts = 0.01;
Gd = c2d(Controlled_plant, Ts);
% --- Simulation setup ---
N = 1000;                      % number of samples
t = (0:N-1)*Ts;                % time vector
r = 50*ones(1, N);                % step input
y = zeros(1, N);               % output
e = zeros(1, N);               % error
de = zeros(1, N);              % delta error
du = zeros(1, N);              % delta u
u = zeros(1, N);               % control input
% --- Initialize discrete plant state ---
[Ad, Bd, Cd, Dd] = ssdata(Gd);
x = zeros(size(Ad,1), 1);      % plant state
% --- Simulation loop ---
% Scaling factors
Ke = 1/50;    % scale for error
Kde = 0.427  ;    % scale for delta error
Kdu = 9;    % scale for delta U
for k = 2:N
   e(k) = r(k) - y(k-1);
   de(k) = e(k) - e(k-1);
   % normalize error for fuzzy input
 e_norm = max(min(Ke*e(k), 1), -1);
de_norm = max(min(Kde*de(k), 1), -1);
   % evaluate fuzzy controller
   du_norm = evalfis(fzpi, [e_norm de_norm]);
   % denormalize du
   du(k) = Kdu*du_norm;
   u(k) = u(k-1) + du(k);
   % simulate plant
   x = Ad * x + Bd * u(k);
   y(k) = Cd * x + Dd * u(k);
end
% --- Plot results ---
figure;
plot(t, y, 'LineWidth', 1.5); hold on;
plot(t, r, '--', 'LineWidth', 1);
xlabel('Time (s)');
ylabel('Output');
legend('y(t)', 'Reference');
title('Closed-loop response with FZPI controller');
grid on;



%%% 
% --- Simulation setup ---
N = 2000; % number of samples
t = (0:N-1)*Ts; % time vector
r = zeros(1,N);
r(t < 5) = 50; % 0–5 s
r(t >= 5 & t < 10) = 20; % 5–10 s
r(t >= 10) = 40; % 10–20 s
y = zeros(1, N); % output
e = zeros(1, N); % error
de = zeros(1, N); % delta error
du = zeros(1, N); % delta u
u = zeros(1, N); % control input
% --- Initialize discrete plant state ---
[Ad, Bd, Cd, Dd] = ssdata(Gd);
x = zeros(size(Ad,1), 1); % plant state
% --- Simulation loop ---
% Scaling factors
Ke = 1/50; % scale for error
Kde = 0.427 ; % scale for delta error
Kdu = 9; % scale for delta U
for k = 2:N
e(k) = r(k) - y(k-1);
de(k) = e(k) - e(k-1);
% normalize error for fuzzy input
e_norm = max(min(Ke*e(k), 1), -1);
de_norm = max(min(Kde*de(k), 1), -1);
% evaluate fuzzy controller
du_norm = evalfis(fzpi, [e_norm de_norm]);
% denormalize du
du(k) = Kdu*du_norm;
u(k) = u(k-1) + du(k);
% simulate plant
x = Ad * x + Bd * u(k);
y(k) = Cd * x + Dd * u(k);
end
% --- Plot results ---
figure;
plot(t, y, 'LineWidth', 1.5); hold on;
plot(t, r, '--', 'LineWidth', 1);
xlabel('Time (s)');
ylabel('Output');
legend('y(t)', 'Reference');
title('Closed-loop response with FZPI controller');
grid on;
% --- Simulation setup ---
N = 2000; % number of samples
t = (0:N-1)*Ts; % time vector
r = zeros(1,N);
r(t < 5)              = 10 * t(t < 5);
r(t >= 5 & t < 10)    = 50;
r(t >= 10)            = -5 * t(t >= 10) + 100;


y = zeros(1, N); % output
e = zeros(1, N); % error
de = zeros(1, N); % delta error
du = zeros(1, N); % delta u
u = zeros(1, N); % control input
% --- Initialize discrete plant state ---
[Ad, Bd, Cd, Dd] = ssdata(Gd);
x = zeros(size(Ad,1), 1); % plant state
% --- Simulation loop ---
% Scaling factors
Ke = 1/50; % scale for error
Kde = 0.427 ; % scale for delta error
Kdu = 9; % scale for delta U
for k = 2:N
e(k) = r(k) - y(k-1);
de(k) = e(k) - e(k-1);
% normalize error for fuzzy input
e_norm = max(min(Ke*e(k), 1), -1);
de_norm = max(min(Kde*de(k), 1), -1);
% evaluate fuzzy controller
du_norm = evalfis(fzpi, [e_norm de_norm]);
% denormalize du
du(k) = Kdu*du_norm;
u(k) = u(k-1) + du(k);
% simulate plant
x = Ad * x + Bd * u(k);
y(k) = Cd * x + Dd * u(k);
end
% --- Plot results ---
figure;
plot(t, y, 'LineWidth', 1.5); hold on;
plot(t, r, '--', 'LineWidth', 1);
xlabel('Time (s)');
ylabel('Output');
legend('y(t)', 'Reference');
title('Closed-loop response with FZPI controller');
grid on;