function [pi_c, mu_c, sigma_c] = EM(C, TrainsampleDCT)
% EM with dimension 64

% Initialize and normalize pi_c
pi_c = randi(1, C);          
pi_c = pi_c / sum(pi);  % Normalize the sum to 1.

% Initialize mu_c by randomly choosing C observations from data(col = 64)
[row, col] = size(TrainsampleDCT);
mu_c = TrainsampleDCT(randi([1 row],1,C),:);

% Initialize sigma_c by creating an identity matrix of random values.
sigma_c = zeros(col,col,C);
for i =1:C
    sigma_c(:,:,i) = (rand(1,col)).*eye(col);
end   

Gaussion_mixtures = zeros(row,C);
L = zeros(1,1000);
for i = 1:1000
    % E-step
    for j = 1:C
        Gaussion_mixtures(:,j) = mvnpdf(TrainsampleDCT,mu_c(j,:),sigma_c(:,:,j))*pi_c(j);    
    end
    hij = Gaussion_mixtures./sum(Gaussion_mixtures,2);
    % Compute the log-likelihood
    L(i) = sum(log(sum(Gaussion_mixtures,2)));
    
    % M-step
    % Update pi_c, mu_c and sigma_c for n+1.
    pi_c = sum(hij)/col;
    mu_c = (hij'*TrainsampleDCT)./sum(hij)';
    for j = 1:C
        sigma_c(:,:,j) = diag(diag(((TrainsampleDCT-mu_c(j,:))'.*hij(:,j)'* ... 
            (TrainsampleDCT-mu_c(j,:))./sum(hij(:,j),1))+1e-7));
    end
    % Break if likelihood hasn't changed by more than .1% between iteration stop.
    if i > 1
        if abs(L(i) - L(i-1)) < 1e-3
            break; 
        end
    end
end
end