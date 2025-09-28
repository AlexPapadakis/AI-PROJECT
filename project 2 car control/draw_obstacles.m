function draw_obstacles()
% Draws the staircase obstacle boundary

segments = { ...
    [5 0; 5 1];   % vertical at x=5
    [5 1; 6 1];   % horizontal at y=1
    [6 1; 6 2];   % vertical at x=6
    [6 2; 7 2];   % horizontal at y=2
    [7 2; 7 3];   % vertical at x=7
    [7 3; 10 3]   % horizontal at y=3
};

hold on; axis equal
for k = 1:numel(segments)
    p = segments{k};
    plot(p(:,1), p(:,2), 'k-', 'LineWidth', 2);
end
xlabel('X'); ylabel('Y');
title('Obstacle Boundary');
end
