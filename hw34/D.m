clc
clear all;
close all;
%% (a)
load('TrainingSamplesDCT_subsets_8');
load('Alpha.mat');
%load('Prior_1.mat');
load('Prior_2.mat');

cheetah = imread('cheetah.bmp');
cheetah = im2double(cheetah);
[row, col] = size(cheetah);

cheetah_mask = imread('cheetah_mask.bmp');
cheetah_mask = im2double(cheetah_mask);

% computes the mean, covariance, and priors for grass and cheetah
BG_mu = my_mean(D4_BG);
FG_mu = my_mean(D4_FG);
BG_cov = my_cov(D4_BG);
FG_cov = my_cov(D4_FG);
row_bg = size(D4_BG, 1);
row_fg = size(D4_FG, 1);
Prior_grass = row_bg / (row_bg + row_fg);
Prior_cheetah = row_fg / (row_bg + row_fg);

zigzag = load('Zig-Zag pattern.txt') + 1;
zigzag_image = zeros((row - 7)*(col - 7), 64);
% transform the initial image: DCT, zig-zag partern with an 8 x 8 sliding window
for i = 1: row - 7
    for j = 1: col - 7
        sliding_window = cheetah(i:i+7,j:j+7);
        dct_window = dct2(sliding_window);
        % zig-zag
%         for k = 1:64
%             [x,y] = find(zigzag==k);
%             temp(k) = dct_window(x,y);
%         end
        
        % Convert Matrix into Zig-Zag Pattern
        ind = reshape(1:numel(dct_window), size(dct_window));    
        ind = fliplr( spdiags(fliplr(ind)));                 
        ind(:,1:2:end) = flipud(ind(:,1:2:end));              
        ind(ind==0) = [];                                       
        
        zigzag_image((i-1)*(col-7)+j,:) = dct_window(ind);          
    end
end
% Predictive Distribution
P_error_PD = zeros(1,9);
for x = 1:9
    % computes the parameters of the predictive distribution -- mu_n and sigma_n
    cov0 = diag(alpha(x) * W0);
    k1 = row_bg * cov0 / (row_bg * cov0 + BG_cov);
    k2 = BG_cov / (row_bg * cov0 + BG_cov);
    BG_mu_n = k1 * BG_mu' + k2 * mu0_BG';
    BG_sigma_n = BG_cov * cov0 / (BG_cov + row_bg * cov0);
    BG_sigma = ((BG_cov + BG_sigma_n) + (BG_cov + BG_sigma_n)') / 2;
    
    k1 = row_fg * cov0 / (row_fg * cov0 + FG_cov);
    k2 = FG_cov / (row_fg * cov0 + FG_cov);
    FG_mu_n = k1 * FG_mu' + k2 * mu0_FG';
    FG_sigma_n = FG_cov * cov0 / (FG_cov + row_fg * cov0);  
    FG_sigma = ((FG_cov + FG_sigma_n) + (FG_cov + FG_sigma_n)') / 2;
    
    % Bayesian Decision rule to classify
    restore_image = zeros(row-7, col-7);
    for i = 1:row - 7
        for j = 1:col - 7
            P_grass = mvnpdf(zigzag_image((i-1)*(col-7)+j,:), BG_mu_n', BG_sigma)*Prior_grass;
            P_cheetah = mvnpdf(zigzag_image((i-1)*(col-7)+j,:), FG_mu_n', FG_sigma)*Prior_cheetah;
            if P_grass < P_cheetah
                restore_image(i, j) = 1;
            else
                restore_image(i, j) = 0;
            end
        end
    end

    % computes probability of error
    num_wrong = 0;
    for i=1:row-7
        for j=1:col-7
            if restore_image(i,j) ~= cheetah_mask(i,j)
                num_wrong = num_wrong+1;
            end
        end
    end
    P_error_PD(x) = num_wrong / (row * col);
end

%% (b) Maximum likelihood
P_error_ML = zeros(1,9);
for x = 1:9
    % Bayesian Decision rule to classify 
    restore_image = zeros(row-7, col-7);
    for i = 1:row - 7
        for j = 1:col - 7
            P_grass = mvnpdf(zigzag_image((i-1)*(col-7)+j,:), BG_mu, BG_cov)*Prior_grass;
            P_cheetah = mvnpdf(zigzag_image((i-1)*(col-7)+j,:), FG_mu, FG_cov)*Prior_cheetah;
            if P_grass < P_cheetah
                restore_image(i, j) = 1;
            else
                restore_image(i, j) = 0;
            end
        end
    end

    % computes probability of error
    num_wrong = 0;
    for i=1:row-7
        for j=1:col-7
            if restore_image(i,j) ~= cheetah_mask(i,j)
                num_wrong = num_wrong+1;
            end
        end
    end
    P_error_ML(x) = num_wrong / (row * col);    
end


%% (c) MAP
P_error_MAP = zeros(1,9);
for x = 1:9
    % computes the parameters of the predictive distribution -- mu_n and sigma_n
    cov0 = diag(alpha(x) * W0);
    k1 = row_bg * cov0 / (row_bg * cov0 + BG_cov);
    k2 = BG_cov / (row_bg * cov0 + BG_cov);
    BG_mu_n = k1 * BG_mu' + k2 * mu0_BG';
    BG_sigma_n = BG_cov * cov0 / (BG_cov + row_bg * cov0);
    BG_sigma = ((BG_cov + BG_sigma_n) + (BG_cov + BG_sigma_n)') / 2;
    
    k1 = row_fg * cov0 / (row_fg * cov0 + FG_cov);
    k2 = FG_cov / (row_fg * cov0 + FG_cov);
    FG_mu_n = k1 * FG_mu' + k2 * mu0_FG';
    FG_sigma_n = FG_cov * cov0 / (FG_cov + row_fg * cov0);
    FG_sigma = ((FG_cov + FG_sigma_n) + (FG_cov + FG_sigma_n)') / 2;
    
    % Bayesian Decision rule to classify
    restore_image = zeros(row-7, col-7);
    for i = 1:row - 7
        for j = 1:col - 7
            P_grass = mvnpdf(zigzag_image((i-1)*(col-7)+j,:), BG_mu_n', BG_cov)*Prior_grass;
            P_cheetah = mvnpdf(zigzag_image((i-1)*(col-7)+j,:), FG_mu_n', FG_cov)*Prior_cheetah;
            if P_grass < P_cheetah
                restore_image(i, j) = 1;
            else
                restore_image(i, j) = 0;
            end
        end
    end
 
    % computes probability of error
    num_wrong = 0;
    for i=1:row-7
        for j=1:col-7
            if restore_image(i,j) ~= cheetah_mask(i,j)
                num_wrong = num_wrong+1;
            end
        end
    end
    P_error_MAP(x) = num_wrong / (row * col);
end

figure;
hold on;
grid on;
plot(alpha, P_error_PD);
plot(alpha, P_error_ML);
plot(alpha, P_error_MAP);
set(gca,'XScale','log');
legend('PD', 'ML', 'MAP');
hold off;
%% implements user-define mean and cov function
function m = my_mean(A)
    row = size(A,1);
    m = 1/row * ones(1, row) * A;
end

% A is a matrix whose columns represent random variables and whose rows represent observations
% and covariance matrix is the corresponding column variances along the diagonal.
function cov = my_cov(A)
    row = size(A, 1);
    mu = 1/row * ones(1, row) * A;
    mu_rep = repmat(mu, row, 1);
    cov = (A - mu_rep)' * (A - mu_rep) / (row - 1);
end