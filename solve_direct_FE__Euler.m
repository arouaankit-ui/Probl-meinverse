function [U,U0,UL] = solve_direct_FE__Euler(a,L,Ne,dt,Nt,Fd_ref,~,~,f_init)

    % Maillage 
    h = L/(Ne-1);

    % Matrice de masse M 
    M = sparse(Ne,Ne);
    Me = (h/6) * [2 1; 1 2];
    for e = 1:(Ne-1)
        idx = [e e+1];
        M(idx,idx) = M(idx,idx) + Me;
    end

    % Matrice de rigidité K 
    K = sparse(Ne,Ne);
    Ke = (1/h) * [ 1 -1;
                  -1  1];
    for e = 1:(Ne-1)
        idx = [e e+1];
        K(idx,idx) = K(idx,idx) + Ke;
    end
    K = a * K;

    % Initialisation 
    U = zeros(Ne,Nt);
    U(:,1) = f_init(:);

    % Euler explicite 
    for k = 2:Nt
        rhs = Fd_ref(:,k-1) - K * U(:,k-1);
        U(:,k) = U(:,k-1) + dt * (M \ rhs);
    end

    % Mesures aux bords 
    U0 = U(1,:);
    UL = U(Ne,:);
end
