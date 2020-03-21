%% generate 5 mixtures for BG and FG
load('TrainingSamplesDCT_8_new.mat');
cheetah = imread('cheetah.bmp');
cheetah = im2double(cheetah);
[row, col] = size(cheetah);
cheetah_mask = imread('cheetah_mask.bmp');
cheetah_mask = im2double(cheetah_mask);

% convert the image from DCT transform and zig-zag pattern with 8x8 sliding window
zigzag_image = DCT_ZIGZAG(cheetah, row, col);

C = 32;
% initialize pi_c, mu_c, sigma_c of BG and FG for 5 EM classifier 
numMixture = 5;
pi_c_BG = zeros(1, C, numMixture);
mu_c_BG = zeros(C, 64, numMixture);
sigma_c_BG = zeros(64, 64, C, numMixture);

pi_c_FG = zeros(1, C, numMixture);
mu_c_FG = zeros(C, 64, numMixture);
sigma_c_FG = zeros(64, 64, C, numMixture);

% Expectation Maximization for BG and FG of 5 mixtures
for i = 1:numMixture
    [pi_c_BG(:,:,i), mu_c_BG(:,:,i), sigma_c_BG(:,:,:,i)] = EM(C, TrainsampleDCT_BG);
    [pi_c_FG(:,:,i), mu_c_FG(:,:,i), sigma_c_FG(:,:,:,i)] = EM(C, TrainsampleDCT_FG);
end
dimensions = [1,2,4,8,16,24,32,40,48,56,64];
P_error = zeros(size(dimensions,1), 5, 5);

%% 
tic;
% Classification with BDR
for x = 1:1 % BG 1:5
    for y = 1:1 % FG 1:5
        for dimIndex = 1:length(dimensions)
            dim = dimensions(dimIndex);
            restore_image = zeros(row-7, col-7);
            for i = 1:row - 7
                for j = 1:col - 7
                    P_BG = 0;
                    P_FG = 0;
                     % Compute total posterior P for BG 
                    for t = 1:C
                        P_BG = P_BG + mvnpdf(zigzag_image((i-1)*(col-7)+j,1:dim), ...
                            mu_c_BG(t,1:dim,x),sigma_c_BG(1:dim,1:dim,t,x))*pi_c_BG(:,t,x);
                    end
                    % Compute total posterior P for FG
                    for t = 1:C
                        P_FG = P_FG + mvnpdf(zigzag_image((i-1)*(col-7)+j,1:dim), ...
                            mu_c_FG(t,1:dim,y),sigma_c_FG(1:dim,1:dim,t,y))*pi_c_FG(:,t,y);
                    end
                    
                    % decision making
                    if P_BG < P_FG
                        restore_image(i, j) = 1;
                    else
                        restore_image(i, j) = 0;
                    end
                end
            end
%             figure;
%             imagesc(restore_image);
%             colormap(gray(255));

            % computes probability of error
            num_wrong = 0;
            for i=1:row-7
                for j=1:col-7
                    if restore_image(i,j) ~= cheetah_mask(i,j)
                        num_wrong = num_wrong+1;
                    end
                end
            end
            %P_error(dimIndex, x, y) = num_wrong / (row * col); % part a
            P_error_c(6,dimIndex) = num_wrong / (row * col); % part b
        end
    end
end
%% part a figure
for x = 1:1
    figure;
    hold on;
    grid on;
    for y = 1:1
        plot(dimensions, P_error(:,x,y));
    end
    legend('FG','FG2','FG3','FG4','FG5');    
end
%% part b figure
% figure;
hold on;
grid on;
plot(dimensions, P_error_c(1,:));

legend('C=1','C=2','C=4','C=8','C=16','C=32'); 
toc;