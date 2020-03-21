clc
clear all
close all
%% (a)
load('TrainingSamplesDCT_8_new.mat');
BG = TrainsampleDCT_BG;
FG = TrainsampleDCT_FG;
[row_bg, ~] = size(BG);
[row_fg, ~] = size(FG);
P_cheetah = row_fg / (row_fg + row_bg);
P_grass = row_bg / (row_fg + row_bg);

%% (b)
one_bg = ones(1, row_bg);
one_fg = ones(1, row_fg);
BG_mu = (1/row_bg) * one_bg * BG;
FG_mu = (1/row_fg) * one_fg * FG; 

BG_std = sqrt(1/(row_bg)*(one_bg*BG.^2)-BG_mu.^2);
FG_std = sqrt(1/(row_fg)*(one_fg*FG.^2)-FG_mu.^2);

% below plot all 64 figure
for i=1:1:64
    if (mod(i,8)==1)
        figure;  
    end
    if (mod(i,8)==0)
        subplot(2,4,8);
    else
        subplot(2,4,mod(i,8));
    end
    grid on;
    hold on;
    P_Xk_Y_Gr=normpdf(sort(BG(:,i)),BG_mu(i),BG_std(i));
    P_Xk_Y_Ch=normpdf(sort(FG(:,i)),FG_mu(i),FG_std(i));
    plot(sort(BG(1:1053,i)), P_Xk_Y_Gr,'b');
    plot(sort(FG(1:250,i)), P_Xk_Y_Ch,'r');
    title(['Index  ' num2str(i)]);
    legend('BG','FG');
    hold off;
end
close all;

Best8 = [1 22 23 35 47 48 50 58];
Worst8 = [4 5 6 59 61 62 63 64];
flag =1;
for i=1:1:64
    if (find(Best8==i))
        subplot(2,4,flag);
        flag = flag + 1;
        grid on;
        hold on;
        P_Xk_Y_Gr=normpdf(sort(BG(:,i)),BG_mu(i),BG_std(i));
        P_Xk_Y_Ch=normpdf(sort(FG(:,i)),FG_mu(i),FG_std(i));
        plot(sort(BG(1:1053,i)), P_Xk_Y_Gr,'b');
        plot(sort(FG(1:250,i)), P_Xk_Y_Ch,'r');
        title(['Index - ' num2str(i)]);
        legend('BG','FG');
        hold off;
    end
end

Best_BG = BG(:,Best8);
Best_FG = FG(:,Best8);

Best_BG_mu = (1/1053)*(one_bg*BG);
Best_FG_mu = (1/250)*(one_fg*FG);

Best_BG_mu_rep = repmat(Best_BG_mu,1053,1);
Best_FG_mu_rep = repmat(Best_FG_mu,250,1);

Best_Cov_mat_Bg = (Best_BG-Best_BG_mu_rep)'*(Best_BG-Best_BG_mu_rep)*(1/1053);
Best_Cov_mat_Fg = (Best_FG-Best_FG_mu_rep)'*(Best_FG-Best_FG_mu_rep)*(1/250);

figure;
flag =1;
for i=1:1:64
    if (find(Worst8==i))
        subplot(2,4,flag);
        flag = flag + 1;
        grid on;
        hold on;
        P_Xk_Y_Gr=normpdf(sort(BG(:,i)),BG_mu(i),BG_std(i));
        P_Xk_Y_Ch=normpdf(sort(FG(:,i)),FG_mu(i),FG_std(i));
        plot(sort(BG(1:1053,i)), P_Xk_Y_Gr,'b');
        plot(sort(FG(1:250,i)), P_Xk_Y_Ch,'r');
        title(['Index - ' num2str(i)]);
        legend('BG','FG');
        hold off;
    end
end

%% c)
Cheetah = im2double(imread('cheetah.bmp'));
zig_zag = load('Zig-Zag pattern.txt')+1;

[row,col] = size(Cheetah);

BG_mean_rep = repmat(BG_mean,1053,1);
FG_mean_rep = repmat(FG_mean,250,1);

Cov_mat_Bg = (BG-BG_mean_rep)'*(BG-BG_mean_rep)*(1/1053);
Cov_mat_Fg = (FG-FG_mean_rep)'*(FG-FG_mean_rep)*(1/250);

% Features of 64-dimensional Gaussians 
for i = 1: row - 7
    for j = 1:col - 7
        sliding_window = abs(dct2(Cheetah(i:i+7,j:j+7)));
        for k = 1:64
         [x,y] = find(zig_zag==k);
         temp(k) = sliding_window(x,y);
        end
          P_Y_X_Gr = mvnpdf (temp',BG_mu',Cov_mat_Bg)*P_grass;
          P_Y_X_Ch = mvnpdf (temp',FG_mu',Cov_mat_Fg)*P_cheetah;
        if (P_Y_X_Gr > P_Y_X_Ch)
            g64(i,j) = 0;
        else
            g64(i,j) = 1;
        end
    end
end
figure;
imagesc(g64);
colormap(gray(255));


% Features of 8-dimensional Gaussians
for i = 1:row - 7
    for j = 1:col - 7
        sliding_window = abs(dct2(Cheetah(i:i+7,j:j+7)));
        for k = 1:8
         [x,y] = find(zig_zag==Best8(k));
         temp0(k) = sliding_window(x,y);
        end
          P_Y_X_Gr = mvnpdf (temp0,Best_BG_mu,Best_Cov_mat_Bg)*P_grass;
          P_Y_X_Ch = mvnpdf (temp0,Best_FG_mu,Best_Cov_mat_Fg)*P_cheetah;
        if (P_Y_X_Gr > P_Y_X_Ch)
            g8(i,j) = 0;
        else
            g8(i,j) = 1;
        end
    end
end
figure;
imagesc(g8);
colormap(gray(255));

mask = im2double(imread('cheetah_mask.bmp'));
error_64= sum(sum(abs(mask(1:row-7,1:col-7)- g64)),2);
error_8= sum(sum(abs(mask(1:row-7,1:col-7)- g8)),2);
P_ERROR_64 = error_64/((row-7)*(col-7))
P_ERROR_8 = error_8/((row-7)*(col-7))




