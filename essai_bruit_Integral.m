%% INVERSE REGULARISATION COURBURE
clear; close all; clc;

%% données du problème
a  = 1;
l  = 1;
Ne = 100;
dt = 1e-5;
Nt = 4000;

x = (l/Ne)*(1:Ne)';

%% conditions aux limites
Fd_ref   = zeros(Ne,Nt);
dep_impos = Ne;
uNt = zeros(Nt,1);

%% Base sinus
P = 15;
%P = 50;
Phi = zeros(Ne,P);
for m = 1:P
    Phi(:,m) = sin(m*pi*x/l);
end

%% mesures du problème direct
F_true = zeros(P,1);
F_true(1)  = 1.00;
F_true(2)  = 0.35;
F_true(3)  = -0.20;
F_true(5)  = 0.10;
F_true(6)  = 0.11;
F_true(7)  = 0.10;
F_true(8)  = 0.07;
F_true(10) = -0.05;
F_true(12) = 0.05;
F_true(13) = 0.07;
F_true(14) = 0.10; 
%F_true(16) = -0.06;F_true(17) =  0.05;F_true(18) =  0.04;F_true(20) = -0.045;F_true(22) =  0.04;F_true(24) = -0.035;F_true(26) =  0.030;F_true(28) = -0.028;F_true(30) =  0.025;
%F_true(33) = -0.020;F_true(36) =  0.018;F_true(40) = -0.015;F_true(44) =  0.012;F_true(47) = -0.010;F_true(50) =  0.008;

f_ref = Phi * F_true;

[~, U0_ref, UL_ref] = solve_direct_FE__Euler(a,l,Ne,dt,Nt,Fd_ref,dep_impos,uNt,f_ref);

Utild0 = U0_ref;
UtildL = UL_ref;

%% vecteur des mesures d (2Nt x 1)
d = zeros(2*Nt,1);
for k = 1:Nt
   d(2*k-1) = Utild0(k);
   d(2*k)   = UtildL(k);
end

%% bruit 
eps = 0.01;
%eps = 0.1;
sigma = eps * norm(d) / sqrt(length(d));
d = d + sigma * randn(size(d));

%% matrice d'observation O (2Nt x P)
O = zeros(2*Nt, P);
for m = 1:P
    f_init_m = Phi(:,m);
    [~, U0_m, UL_m] = solve_direct_FE__Euler(a,l,Ne,dt,Nt,Fd_ref,dep_impos,uNt,f_init_m);

    for k = 1:Nt
        O(2*k-1, m) = U0_m(k);
        O(2*k,   m) = UL_m(k);
    end
end

%% Least squares
F_ls = (O.'*O)\(O.'*d);
f_ls = Phi * F_ls;

%% Regularisation ordre 2

modes = (1:P)';
w = (modes*pi/l).^4;           
LTL = diag(w);                 

scale = trace(O.'*O) / trace(LTL);

alpha_list = logspace(-10, 10, 400);

err_f = zeros(size(alpha_list));   
resid = zeros(size(alpha_list));   
penal = zeros(size(alpha_list));   
F_all = zeros(P, numel(alpha_list));

for i = 1:numel(alpha_list)
    lambda = alpha_list(i) * scale;
    F_reg_i = (O.'*O + lambda*LTL) \ (O.'*d);
    F_all(:,i) = F_reg_i;
    f_reg_i = Phi * F_reg_i; 
    err_f(i) = norm(f_ref - f_reg_i, 2);   
    resid(i) = norm(d - O*F_reg_i, 2);
    penal(i) = sqrt(F_reg_i.' * LTL * F_reg_i);   
end

% Choix lambda 
[~, i_best] = min(err_f);

lambda_best = alpha_list(i_best) * scale;
F_reg = F_all(:,i_best);
f_reg = Phi * F_reg;

fprintf('lambda_best=%.3e | alpha=%.3e | err_f=%.3e | resid=%.3e | ||LF||=%.3e | i_best=%d\n', ...
        lambda_best, alpha_list(i_best), err_f(i_best), resid(i_best), penal(i_best), i_best);
    
Ef_ls  = norm(f_ref - f_ls, 2);
Ef_reg = norm(f_ref - f_reg, 2);
fprintf('Ef_ls=%.3e | Ef_reg=%.3e\n', Ef_ls, Ef_reg);


%% Figures
% f(x)
figure;
plot(x, f_ref, 'b-', 'LineWidth', 2); hold on;
plot(x, f_ls,  'r--', 'LineWidth', 1.2);
plot(x, f_reg, 'k-.', 'LineWidth', 1.6);
xlabel('x'); ylabel('f(x)=u(x,0)');
legend('f_{ref}','f_{LS}','f_{reg}','Location','Best');
title('Condition initiale : référence vs estimée (P=15, base SIN)');

% courbe erreur vs alpha 
figure; semilogx(alpha_list, err_f, 'LineWidth', 1.5); grid on; hold on;
plot(alpha_list(i_best), err_f(i_best), 'kx', 'LineWidth', 2, 'MarkerSize', 10);
xlabel('\alpha'); ylabel('||f_{ref}-f_{reg}||_2');
title('Choix de (minimum de l''erreur sur f)');

% L-curve 
figure; loglog(resid, penal, 'o-'); grid on; hold on;
loglog(resid(i_best), penal(i_best), 'kx', 'LineWidth', 2, 'MarkerSize', 10);
xlabel('||d - O F||_2'); ylabel('||L F||_2');
title('L-curve');

figure;
plot(x, f_ref,'b','LineWidth',2); hold on;
plot(x, f_reg,'k-.','LineWidth',1.6);
grid on; xlabel('x'); ylabel('f(x)');
legend('f_{ref}','f_{reg}');
ylim([min(f_ref)-0.2, max(f_ref)+0.2]);  

figure;
plot(x, f_ls,'r--','LineWidth',1.2); grid on;
xlabel('x'); ylabel('f_{LS}(x)');
title('f_{LS} (attention : peut exploser si problème mal conditionné)');



