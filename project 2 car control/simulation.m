%% Car Control Simulation for Multiple Initial Orientations

% Simulation parameters
xd = 10;       % target X
yd = 3.2;      % target Y
u = 0.05;      % forward speed
Xinit = 4;     
Yinit = 0.4;
dt = 1;        % time step
nSteps = 2000; % max simulation steps

% Initial orientations and colors
thetas = [0, -45, -90, -170];
colors = ['r', 'g', 'b', 'm'];

% Load Fuzzy Logic Controller
carControl = readfis('car_control.fis');

% Plot obstacles
draw_obstacles(); 
hold on;

% Simulate and plot for each initial orientation
for i = 1:length(thetas)
    carX = Xinit;
    carY = Yinit;
    carTheta = thetas(i);

    % Initialize history arrays
    Xhist = zeros(1, nSteps);
    Yhist = zeros(1, nSteps);

    % Simulation loop
    for k = 1:nSteps
        [dh, dv] = obstacle_axis_dist(carX, carY);      % sense environment
        dtheta = evalfis(carControl, [dv dh carTheta]); % FLC output
        carTheta = max(-180, min(180, carTheta + dtheta)); % update heading

        % Update position
        carX = carX + u * cosd(carTheta) * dt;
        carY = carY + u * sind(carTheta) * dt;

        % Store history
        Xhist(k) = carX;
        Yhist(k) = carY;

        % Stop if target X reached
        if carX >= xd
            Xhist = Xhist(1:k);
            Yhist = Yhist(1:k);
            break;
        end
    end

    % Plot path
    plot(Xhist, Yhist, [colors(i) '-o'], 'LineWidth', 1.5, ...
         'DisplayName', ['theta = ' num2str(thetas(i)) '°']);
end

% Finalize plot
xlabel('X'); ylabel('Y');
axis equal; grid on;
title('Car Paths for Different Initial Orientations');
legend show;
