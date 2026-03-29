%% INVERSE P=15 , regularisation L2 : comparaison de deux niveaux de bruit
clear; close all; clc;

%% données
a  = 1;
l  = 1;
Ne = 100;
dt = 1e-5;
Nt = 4000;

x = (l/Ne)*(1:Ne)';

%% conditions aux limites
Fd_ref    = zeros(Ne,Nt);
dep_impos = Ne;
uNt       = zeros(Nt,1);

%% Base sinus
P = 15;
% P = 50;

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

f_ref = Phi * F_true;

[U_ref, U0_ref, UL_ref] = solve_direct_FE__Euler(a,l,Ne,dt,Nt,Fd_ref,dep_impos,uNt,f_ref);

Utild0 = U0_ref;
UtildL = UL_ref;

%% vecteur des mesures sans bruit
d_ref = zeros(2*Nt,1);
for k = 1:Nt
    d_ref(2*k-1) = Utild0(k);
    d_ref(2*k)   = UtildL(k);
end

fprintf('Vecteur de mesures sans bruit construit.\n');

%% matrice d'observation O 
O = zeros(2*Nt, P);

for m = 1:P
    f_init_m = Phi(:,m);
    [~, U0_m, UL_m] = solve_direct_FE__Euler(a,l,Ne,dt,Nt,Fd_ref,dep_impos,uNt,f_init_m);

    for k = 1:Nt
        O(2*k-1, m) = U0_m(k);
        O(2*k,   m) = UL_m(k);
    end
end

fprintf('Matrice d''observation O construite.\n');

%% niveaux de bruit à comparer
eps_list = [0.01, 0.10];   % 1% et 10%
n_cases  = length(eps_list);

%% stockage des résultats
f_ls_all      = zeros(Ne, n_cases);
f_reg_all     = zeros(Ne, n_cases);
Ef_ls_all     = zeros(n_cases,1);
Ef_reg_all    = zeros(n_cases,1);
alpha_best_all = zeros(n_cases,1);
lambda_best_all = zeros(n_cases,1);

%% boucle sur les niveaux de bruit
for j = 1:n_cases

    eps = eps_list(j);

    rng(1);

    % ajout du bruit
    sigma = eps * norm(d_ref) / sqrt(length(d_ref));
    d = d_ref + sigma * randn(size(d_ref));

    fprintf('\n==============================\n');
    fprintf('Cas %d : bruit = %.0f%%\n', j, eps*100);
    fprintf('sigma = %.3e\n', sigma);

    %% moindres carrés
    F_ls = (O.' * O) \ (O.' * d);
    f_ls = Phi * F_ls;

    %% Régularisation Tikhonov L2
    scale = trace(O.'*O)/P;
    alpha_list = logspace(-10, 10, 400);

    resid = zeros(size(alpha_list));
    err_f = zeros(size(alpha_list));
    soln  = zeros(size(alpha_list));
    F_all = zeros(P, numel(alpha_list));

    for i = 1:numel(alpha_list)
        lambda = alpha_list(i) * scale;

        F_reg_i = (O.'*O + lambda*eye(P)) \ (O.'*d);
        F_all(:,i) = F_reg_i;

        f_reg_i = Phi * F_reg_i;
        err_f(i) = norm(f_ref - f_reg_i, 2);
        soln(i)  = norm(F_reg_i, 2);
        resid(i) = norm(d - O*F_reg_i, 2);
    end

    % choix de lambda
    [~, i_best] = min(err_f);

    lambda_best = alpha_list(i_best) * scale;
    F_reg = F_all(:,i_best);
    f_reg = Phi * F_reg;

    Ef_ls  = norm(f_ref - f_ls, 2);
    Ef_reg = norm(f_ref - f_reg, 2);

    fprintf('alpha_best   = %.3e\n', alpha_list(i_best));
    fprintf('lambda_best  = %.3e\n', lambda_best);
    fprintf('Ef_ls        = %.3e\n', Ef_ls);
    fprintf('Ef_reg       = %.3e\n', Ef_reg);

    %% stockage
    f_ls_all(:,j)       = f_ls;
    f_reg_all(:,j)      = f_reg;
    Ef_ls_all(j)        = Ef_ls;
    Ef_reg_all(j)       = Ef_reg;
    alpha_best_all(j)   = alpha_list(i_best);
    lambda_best_all(j)  = lambda_best;

    %% option : courbe du choix de alpha pour chaque bruit
    figure;
    semilogx(alpha_list, err_f, 'LineWidth', 1.5);
    grid on;
    xlabel('\alpha');
    ylabel('||f_{ref}-f_{reg}||_2');
    title(sprintf('Choix de \\alpha - bruit = %.0f%%', eps*100));

end

%% figure finale : comparaison des deux bruits sur la même courbe
figure;
plot(x, f_ref, 'b-', 'LineWidth', 2); hold on;
plot(x, f_reg_all(:,1), 'k-.', 'LineWidth', 1.8);
plot(x, f_reg_all(:,2), 'r--', 'LineWidth', 1.8);
grid on;
xlabel('x');
ylabel('f(x)=u(x,0)');
legend('f_{ref}', 'f_{reg} - bruit 1%', 'f_{reg} - bruit 10%', 'Location', 'best');
title(sprintf('Comparaison de l''effet du bruit sur la reconstruction régularisée (P=%d)', P));

%% option : comparaison LS 
figure;
plot(x, f_ref, 'b-', 'LineWidth', 2); hold on;
plot(x, f_ls_all(:,1), 'm--', 'LineWidth', 1.5);
plot(x, f_ls_all(:,2), 'c--', 'LineWidth', 1.5);
grid on;
xlabel('x');
ylabel('f(x)=u(x,0)');
legend('f_{ref}', 'f_{LS} - bruit 1%', 'f_{LS} - bruit 10%', 'Location', 'best');
title(sprintf('Comparaison de l''effet du bruit sur la solution moindres carrés (P=%d)', P));

%% option : tout sur la même figure
figure;
plot(x, f_ref, 'b-', 'LineWidth', 2); hold on;
plot(x, f_reg_all(:,1), 'k-.', 'LineWidth', 1.8);
plot(x, f_reg_all(:,2), 'r--', 'LineWidth', 1.8);
plot(x, f_ls_all(:,1),  'g:', 'LineWidth', 1.6);
plot(x, f_ls_all(:,2),  'm:', 'LineWidth', 1.6);
grid on;
xlabel('x');
ylabel('f(x)=u(x,0)');
legend('f_{ref}', ...
       'f_{reg} - bruit 1%', 'f_{reg} - bruit 10%', ...
       'f_{LS} - bruit 1%',  'f_{LS} - bruit 10%', ...
       'Location', 'best');
title(sprintf('Comparaison des reconstructions pour deux niveaux de bruit (P=%d)', P));

%% affichage 
fprintf('\n========== Résumé final ==========\n');
for j = 1:n_cases
    fprintf('Bruit = %.0f%% | Ef_ls = %.3e | Ef_reg = %.3e | alpha_best = %.3e\n', ...
        eps_list(j)*100, Ef_ls_all(j), Ef_reg_all(j), alpha_best_all(j));
    figure;
plot(x, f_ref, 'b-', 'LineWidth', 2); hold on;
plot(x, f_reg_all(:,1), 'k-', 'LineWidth', 2);
plot(x, f_reg_all(:,2), 'r--', 'LineWidth', 2);

grid on;
xlabel('x');
ylabel('f(x)=u(x,0)');
legend('Référence', 'Régularisé (bruit 1%)', 'Régularisé (bruit 10%)', 'Location', 'best');

title(sprintf('Effet du bruit sur la reconstruction régularisée (P=%d)', P));
end