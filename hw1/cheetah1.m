clc
clear all
close all
%% (a)
load('TrainingSamplesDCT_8.mat');
BG = abs(TrainsampleDCT_BG);
FG = abs(TrainsampleDCT_FG);
[row_bg, col_bg] = size(BG);
[row_fg, col_fg] = size(FG);
P_cheetah = row_fg / (row_fg + row_bg);
P_grass = row_bg / (row_fg + row_bg);

%% (b)
sortedBG = sort(BG, 2, 'descend');
sortedFG = sort(FG, 2, 'descend');
for i= 1:1053
    bg(i) = find(BG(i,:) == sortedBG(i,2));
end
for j= 1:250
    fg(j) = find(FG(j,:) == sortedFG(j,2));
end
histogram(bg, 'BinEdges', 1:64, 'Normalization', 'pdf');
title("Foreground");
ylabel('P_X_|_Y(x|cheetah)');
figure;
histogram(fg, 'BinEdges', 1:64, 'Normalization', 'pdf');
ylabel('P_X_|_Y(x|grass)');
title("Backeground");

%% (c)
zigzag = load('Zig-Zag Pattern.txt') + 1;
cht = imread('cheetah.bmp');
cht = im2double(cht);
for i = 1:248
    for j = 1:263
        sliding_w = cht(i:i+7, j:j+7);
        sw_dct = abs(dct2(sliding_w));
        for k = 1:64
            [x, y] = find(zigzag == k);
            temp(k) = sw_dct(x, y);
        end
        sorted_temp = sort(temp, 'descend');
        T(i, j) = find(temp == sorted_temp(2));      
    end
end
P_x_cheetah = histcounts(fg, 1:64) / row_fg;
P_x_grass = histcounts(bg, 1:64) / row_bg;
for i = 1:248
    for j = 1:263
        X = T(i, j);
        P_x = P_x_cheetah(X) * P_cheetah + P_x_grass(X) * P_grass;
        P_cheetah_x = P_x_cheetah(X) * P_cheetah / P_x;
        P_grass_x = P_x_grass(X) * P_grass / P_x;
        if P_cheetah_x > P_grass_x
            g(i, j) = 1;
        else
            g(i, j) = 0;
        end
    end
end
figure;
imagesc(g);
colormap(gray(255));

%% (d)
mask = imread('cheetah_mask.bmp');
mask = im2double(mask);
error = abs(mask(1:248, 1:263) - g);
sum_error = sum(sum(error), 2);
P_error = sum_error / (248 * 263);
      