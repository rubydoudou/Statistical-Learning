load('TrainingSamplesDCT_subsets_8.mat')
load('Prior_1.mat')
load('Prior_2.mat')
load('Alpha.mat')

xDimension = 255;
modifiedXDimension = xDimension-7;
yDimension = 270;
modifiedYDimension = yDimension-7;

errorPD = zeros(1,9);
errorML = zeros(1,9);
errorMAP = zeros(1,9);

% Read in the initial image.
initialImage = imread('cheetah.bmp');
initialImage = im2double(initialImage);

% Read in the mask image.
initialMaskImage = imread('cheetah_mask.bmp');
initialMaskImage = im2double(initialMaskImage);

% Update these for each class.
BGMean = mean(D1_BG);
FGMean = mean(D1_FG);
BGCov = cov(D1_BG);
FGCov  = cov(D1_FG);
priorGrass = length(D1_BG)/(length(D1_BG)+length(D1_FG));
priorCheetah = length(D1_FG)/(length(D1_BG)+length(D1_FG));

% Convert entire initialImage into dct form to create a 65,224x64 matrix
% which contains the dct transformation of each 64x64 block.
zigZagInitialImage = zeros((modifiedXDimension)*(modifiedYDimension),64);
for i = 1:modifiedXDimension
    for j = 1:modifiedYDimension
        currentBlock = initialImage(i:i+7,j:j+7);
        dctMatrix = dct2(currentBlock);
        
        % Convert Matrix into Zig-Zag Pattern
        ind = reshape(1:numel(dctMatrix), size(dctMatrix));     %# indices of elements
        ind = fliplr( spdiags( fliplr(ind) ) );                 %# get the anti-diagonals
        ind(:,1:2:end) = flipud( ind(:,1:2:end) );              %# reverse order of odd columns
        ind(ind==0) = [];                                       %# keep non-zero indices
        zigZagInitialImage((i-1)*(modifiedYDimension)+j,:) = dctMatrix(ind);   %# get elements in zigzag order
    end
end

% Predictive Distribution.
for i=1:length(alpha)
    Cov0 = diag(alpha(i)*W0);
    % ---------- Calculating u_n for P_u|T(u|D) ----------
    Weight1BG = (length(D4_BG)* Cov0)/(length(D4_BG) * Cov0 + BGCov);
    Weight2BG = (BGCov)/(length(D4_BG) * Cov0 + BGCov);
    MuNBG = Weight1BG * BGMean' + Weight2BG * mu0_BG';
    % ---------- Calculating sigma_n^2 for P_u|T(u|D) ----------
    SigmaNBG = (BGCov * Cov0)/(BGCov + length(D4_BG) * Cov0);
    % ---------- Calculating sigma^2 + sigma_n^2 for P_X|T(x|D) ----------
    SigmaNBGCombined = ((BGCov + SigmaNBG) + (BGCov + SigmaNBG)')/2;

    Cov0 = diag(alpha(i)*W0);
    % ---------- Calculating u_n for P_u|T(u|D) ----------
    Weight1FG = (length(D4_FG)* Cov0)/(length(D4_FG) * Cov0 + FGCov);
    Weight2FG = (FGCov)/(length(D4_FG) * Cov0 + FGCov);
    MuNFG = Weight1FG * FGMean' + Weight2FG * mu0_FG';
    % ---------- Calculating sigma_n^2 for P_u|T(u|D) ----------
    SigmaNFG = (FGCov * Cov0)/(FGCov + length(D4_FG) * Cov0);
    % ---------- Calculating sigma^2 + sigma_n^2 for P_X|T(x|D) ----------
    SigmaNFGCombined = ((SigmaNFG + FGCov) + (SigmaNFG + FGCov)')/2;

    % Bayesian Decision Rule.
    maskMatrix = zeros((modifiedXDimension)*(modifiedYDimension),1);
    for x=1:length(zigZagInitialImage)
        finalBDRGrass = log(mvnpdf(zigZagInitialImage(x,:),MuNBG',SigmaNBGCombined)*priorGrass);
        finalBDRCheetah = log(mvnpdf(zigZagInitialImage(x,:),MuNFG',SigmaNFGCombined)*priorCheetah);
        if (finalBDRGrass < finalBDRCheetah)
            maskMatrix(x) = 1;
        else
            maskMatrix(x) = 0;
        end
    end

    % Reform maskMatrix into a 255x270 matrix.
    tempMask = zeros(modifiedXDimension,modifiedYDimension);
    for x=1:modifiedXDimension
        tempMask(x,:) = maskMatrix(((x-1)*(modifiedYDimension)+1):x*(modifiedYDimension))';
    end
    maskMatrix = tempMask;
    figure
    imshow(maskMatrix,[])
    
    % Using the maskMatrix and the initialMaskImage calculate PoE.
    incorrectCount = 0;
    for x=1:modifiedXDimension
        for y=1:modifiedYDimension
            if (initialMaskImage(x,y) ~= maskMatrix(x,y))
                incorrectCount = incorrectCount + 1;
            end
        end
    end
    errorPD(i) = incorrectCount/xDimension/yDimension;    
end

% Maximum Likelihood.
for i=1:length(alpha)
    % Bayesian Decision Rule.
    maskMatrix = zeros((modifiedXDimension)*(modifiedYDimension),1);
    for x=1:length(maskMatrix)
        finalBDRGrass = log(mvnpdf(zigZagInitialImage(x,:),BGMean,BGCov)*priorGrass);
        finalBDRCheetah = log(mvnpdf(zigZagInitialImage(x,:),FGMean,FGCov)*priorCheetah);
        if (finalBDRGrass < finalBDRCheetah)
            maskMatrix(x) = 1;
        else
            maskMatrix(x) = 0;
        end
    end
    
    % Reform maskMatrix into a 255x270 matrix.
    tempMask = zeros(modifiedXDimension,modifiedYDimension);
    for x=1:modifiedXDimension
        tempMask(x,:) = maskMatrix(((x-1)*(modifiedYDimension)+1):x*(modifiedYDimension))';
    end
    maskMatrix = tempMask;
    figure
    imshow(maskMatrix,[])

    % Using the maskMatrix and the initialMaskImage calculate PoE.
    incorrectCount = 0;
    for x=1:modifiedXDimension
        for y=1:modifiedYDimension
            if (initialMaskImage(x,y) ~= maskMatrix(x,y))
                incorrectCount = incorrectCount + 1;
            end
        end
    end
    errorML(i) = incorrectCount/xDimension/yDimension;   
end

% Maximum A Posteriori.
for i=1:length(alpha)
    Cov0 = diag(alpha(i)*W0);
    Weight1BG = (length(D4_BG)* Cov0)/(length(D4_BG) * Cov0 + BGCov);
    Weight2BG = (BGCov)/(length(D4_BG) * Cov0 + BGCov);
    MuNBG = Weight1BG * BGMean' + Weight2BG * mu0_BG';

    Cov0 = diag(alpha(i)*W0);
    Weight1FG = (length(D4_FG)* Cov0)/(length(D4_FG) * Cov0 + FGCov);
    Weight2FG = (FGCov)/(length(D4_FG) * Cov0 + FGCov);
    MuNFG = Weight1FG * FGMean' + Weight2FG * mu0_FG';
    
    % Bayesian Decision Rule.
    maskMatrix = zeros((modifiedXDimension)*(modifiedYDimension),1);
    for x=1:length(maskMatrix)
        finalBDRGrass = log(mvnpdf(zigZagInitialImage(x,:),MuNBG',BGCov)*priorGrass);
        finalBDRCheetah = log(mvnpdf(zigZagInitialImage(x,:),MuNFG',FGCov)*priorCheetah);
        if (finalBDRGrass < finalBDRCheetah)
            maskMatrix(x) = 1;
        else
            maskMatrix(x) = 0;
        end
    end
    
    % Reform maskMatrix into a 255x270 matrix.
    tempMask = zeros(modifiedXDimension,modifiedYDimension);
    for x=1:modifiedXDimension
        tempMask(x,:) = maskMatrix(((x-1)*(modifiedYDimension)+1):x*(modifiedYDimension))';
    end
    maskMatrix = tempMask;
    figure
    imshow(maskMatrix,[])

    % Using the maskMatrix and the initialMaskImage calculate PoE.
    incorrectCount = 0;
    for x=1:modifiedXDimension
        for y=1:modifiedYDimension
            if (initialMaskImage(x,y) ~= maskMatrix(x,y))
                incorrectCount = incorrectCount + 1;
            end
        end
    end
    errorMAP(i) = incorrectCount/xDimension/yDimension;      
end

hold on;
plot(alpha,errorPD)
plot(alpha,errorML)
plot(alpha,errorMAP)
hold off;
set(gca,'XScale','log')
legend('PD','ML','MAP')