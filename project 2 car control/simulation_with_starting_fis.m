%% Car Control Simulation for Different Initial Orientations

% Simulation parameters
xd = 10;       % target X
yd = 3.2;      % target Y
u = 0.05;      % forward speed
Xinit = 4;
Yinit = 0.4;
dt = 1;        % time step
nSteps = 2000; % maximum number of steps

% Initial orientations
thetas = [0, -45, -90, -170];
colors = ['r', 'g', 'b', 'm']; % colors for each path

% Load FLC
carControl = readfis('car_control_starting.fis');

% Draw obstacles
draw_obstacles();
hold on;

% Loop over each initial orientation
for i = 1:length(thetas)
    carX = Xinit;
    carY = Yinit;
    carTheta = thetas(i);

    Xhist = zeros(1, nSteps);
    Yhist = zeros(1, nSteps);

    % Simulation loop
    for k = 1:nSteps
        % Sense environment
        [dh, dv] = obstacle_axis_dist(carX, carY);

        % Compute steering
        dtheta = evalfis(carControl, [dv dh carTheta]);
        carTheta = max(-180, min(180, carTheta + dtheta));

        % Motion update
        carX = carX + u * cosd(carTheta) * dt;
        carY = carY + u * sind(carTheta) * dt;

        % Store position
        Xhist(k) = carX;
        Yhist(k) = carY;

        % Stop if target reached
        if carX >= xd
            Xhist = Xhist(1:k);
            Yhist = Yhist(1:k);
            break;
        end
    end

    % Plot path
    plot(Xhist, Yhist, [colors(i) '-o'], 'LineWidth', 1.5, 'DisplayName', ['theta = ' num2str(thetas(i)) '°']);
end

xlabel('X'); ylabel('Y');
axis equal;
grid on;
title('Car Paths for Different Initial Orientations');
legend show;
