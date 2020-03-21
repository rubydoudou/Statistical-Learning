function zigzag_image = DCT_ZIGZAG(cheetah, row, col)
% transform the initial image: DCT, zig-zag partern with an 8 x 8 sliding window
zigzag = load('Zig-Zag pattern.txt') + 1;
zigzag_image = zeros((row - 7)*(col - 7), 64);
temp = zeros(1,64);
for i = 1: row - 7
    for j = 1: col - 7
        sliding_window = cheetah(i:i+7,j:j+7);
        dct_window = dct2(sliding_window);
        % zig-zag
        for k = 1:64
            [x,y] = find(zigzag==k);
            temp(k) = dct_window(x,y);
        end    
        zigzag_image((i-1)*(col-7)+j,:) = temp;          
    end
end
end