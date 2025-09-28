
s = tf('s');
Gp = 25/((s+0.1)*(s+10));
Kp = 1;
c = 0.2;
Ki = Kp*c;
Gc = Kp*(s+c)/s;
Gol = Gc*Gp;
figure
rlocus(Gol);
K = 0.225;
T = feedback( K*Gc*Gp, 1);
figure
step(T);
% tr = 3s too slow. overshoot just <8%
%%% will pick a new K with c = 0.2
figure
K = 1.66;
T = feedback(K*Gol,1);
step(T);
%%%% tr = 0.43 s // overshoot 4%
%%%Άρα:
% Κολ = Ka*Kp*25
% Με Kp=1 Ka=1.66 Aρα το ιδιο με Κp=1.66 Ka=1 -> Ki = Kp*c = 1.66*0.2 = 0.332
% Kp = 1.66 Ki = 0.332 Ka=1
