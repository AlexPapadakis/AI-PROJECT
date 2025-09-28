function [dh, dv] = obstacle_axis_dist(x,y)
% Range-limited distances from (x,y) to obstacle boundary
% dh ∈ [0,1] distance to nearest obstacle to the right (max 1m)
% dv ∈ [0,1] distance to nearest obstacle below (max 1m)

dh = 1;
dv = 1;

segments = { ...
    [5 0; 5 1];  % vertical at x=5
    [5 1; 6 1];  % horizontal at y=1
    [6 1; 6 2];  % vertical at x=6
    [6 2; 7 2];  % horizontal at y=2
    [7 2; 7 3];  % vertical at x=7
    [7 3; 10 3]   % horizontal at y=3
};

for k = 1:numel(segments)
    p1 = segments{k}(1,:);
    p2 = segments{k}(2,:);
    
    % vertical segment → check right
    if p1(1)==p2(1)
        xv = p1(1);
        ylow = min(p1(2),p2(2));
        yhigh = max(p1(2),p2(2));
        if y>=ylow && y<=yhigh
            distx = xv - x;
            if distx >= 0 && distx <= 1
                dh = min(dh,distx);
            end
        end
    end
    
    % horizontal segment → check down
    if p1(2)==p2(2)
        yh = p1(2);
        xlow = min(p1(1),p2(1));
        xhigh = max(p1(1),p2(1));
        if x>=xlow && x<=xhigh
            distv = y - yh;   % positive if obstacle below
            if distv >= 0 && distv <= 1
                dv = min(dv,distv);
            end
        end
    end
end
end

