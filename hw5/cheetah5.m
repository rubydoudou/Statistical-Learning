load('TrainingSamplesDCT_8_new.mat');
tic;
cheetah = imread('cheetah.bmp');
cheetah = im2double(cheetah);
[row, col] = size(cheetah);
cheetah_mask = imread('cheetah_mask.bmp');
cheetah_mask = im2double(cheetah_mask);

% convert the image from DCT transform and zig-zag pattern with 8x8 sliding window
zigzag_image = DCT_ZIGZAG(cheetah, row, col);

C = 8;
% Expectation Maximization for BG and FG
[pi_c_BG, mu_c_BG, sigma_c_BG] = EM(C, TrainsampleDCT_BG);
[pi_c_FG, mu_c_FG, sigma_c_FG] = EM(C, TrainsampleDCT_FG);


% Classification with BDR
dimensions = [1,2,4,8,16,24,32,40,48,56,64];
P_error = zeros(1,size(dimensions,1));
for dimIndex = 1:length(dimensions)
    dim = dimensions(dimIndex);
    restore_image = zeros(row-7, col-7);
    for i = 1:row - 7
        for j = 1:col - 7
            P_BG = 0;
            P_FG = 0;
            % Compute total posterior P for BG and FG
            for t = 1:C
                P_BG = P_BG + mvnpdf(zigzag_image((i-1)*(col-7)+j,1:dim), ...
                    mu_c_BG(t,1:dim),sigma_c_BG(1:dim,1:dim,t))*pi_c_BG(t);
            end
            
            for t = 1:C
                P_FG = P_FG + mvnpdf(zigzag_image((i-1)*(col-7)+j,1:dim), ...
                    mu_c_FG(t,1:dim),sigma_c_FG(1:dim,1:dim,t))*pi_c_FG(t);
            end
            if P_BG < P_FG
                restore_image(i, j) = 1;
            else
                restore_image(i, j) = 0;
            end
        end
    end
    figure;
    imagesc(restore_image);
    colormap(gray(255));

    % computes probability of error
    num_wrong = 0;
    for i=1:row-7
        for j=1:col-7
            if restore_image(i,j) ~= cheetah_mask(i,j)
                num_wrong = num_wrong+1;
            end
        end
    end
    P_error(dimIndex) = num_wrong / (row * col);
end

figure;
hold on;
grid on;
plot(dimensions, P_error);
%legend('PoE');
hold off;
toc;