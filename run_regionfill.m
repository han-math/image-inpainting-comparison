% Q1 Inpainting — Method 1: regionfill (baseline)
% Images converted to double [0,1] for numerical accuracy.
% Mask refined with imfill + bwareaopen.
% Outputs: filled image, mask, side-by-side comparison.

input_dir  = '/Users/xiaohan/Downloads/第五次小作业/input';
output_dir = '/Users/xiaohan/Downloads/第五次小作业/results_regionfill';
if ~exist(output_dir, 'dir'), mkdir(output_dir); end

pairs = {
    {'bricks.png',               'bricks_L_region_lost.png'};
    {'crayon_paint.png',         'crayon_missingRegion.png'};
    {'fingerprint256.png',        'finger_circle_lost5.png'};
    {'white_house.png',          'white_house_with_lost.png'};
};

fprintf('=== regionfill Inpainting (double precision) ===\n');

for i = 1:size(pairs, 1)
    orig_name = pairs{i}{1};
    hole_name = pairs{i}{2};

    % Read and convert to double [0,1]
    orig = im2double(imread(fullfile(input_dir, orig_name)));
    hole = im2double(imread(fullfile(input_dir, hole_name)));

    % Strip alpha channel if present, verify sizes match
    if size(orig,3) > 3, orig = orig(:,:,1:3); end
    if size(hole,3) > 3, hole = hole(:,:,1:3); end
    assert(isequal(size(orig), size(hole)), 'Size mismatch: %s vs %s', orig_name, hole_name);

    fprintf('\n--- %s (%d x %d, ch=%d) ---\n', orig_name, size(orig,1), size(orig,2), size(orig,3));

    % Build hole mask (threshold 5/255 adapts to [0,1] range)
    if size(orig, 3) == 1
        mask = abs(orig - hole) > 5/255;
    else
        mask = any(abs(orig - hole) > 5/255, 3);
    end

    % Refine mask: fill interior gaps, remove noise speckles
    mask = imfill(mask, 'holes');
    mask = bwareaopen(mask, 20);

    fprintf('  Hole pixels: %d (%.1f%%)\n', sum(mask(:)), 100*sum(mask(:))/numel(mask));

    % regionfill — grayscale direct, RGB channel-by-channel
    if size(hole, 3) == 1
        filled = regionfill(hole, mask);
    else
        filled = hole;
        for c = 1:3
            filled(:,:,c) = regionfill(hole(:,:,c), mask);
        end
    end

    % PSNR / SSIM (on double images)
    psnr_val = psnr(filled, orig);
    ssim_val = ssim(filled, orig);
    fprintf('  PSNR: %.2f dB   SSIM: %.4f\n', psnr_val, ssim_val);

    % Save filled image and mask
    [~, name, ~] = fileparts(orig_name);
    filled_u8 = im2uint8(filled);
    imwrite(filled_u8, fullfile(output_dir, [name '_regionfill.png']));
    imwrite(mask,       fullfile(output_dir, [name '_mask.png']));

    % Build side-by-side comparison using figure 
    orig_rgb   = gray_to_rgb_u8(orig);
    hole_rgb   = gray_to_rgb_u8(hole);
    filled_rgb = gray_to_rgb_u8(filled_u8);

    % Unify height
    H = max([size(orig_rgb,1), size(hole_rgb,1), size(filled_rgb,1)]);
    orig_rgb   = resize_to_h(orig_rgb, H);
    hole_rgb   = resize_to_h(hole_rgb, H);
    filled_rgb = resize_to_h(filled_rgb, H);

    % Side-by-side comparison with labels 
    make_labeled_comparison(orig_rgb, hole_rgb, filled_rgb, ...
        fullfile(output_dir, [name '_compare.png']));

    fprintf('  Saved: %s_compare.png\n', name);
end

fprintf('\n=== Done ===\n');


% Local helper functions
function rgb = gray_to_rgb_u8(im)
    % Convert any image (grayscale or RGB, double or uint8) to uint8 RGB
    if size(im, 3) == 1
        rgb = repmat(im2uint8(im), [1, 1, 3]);
    else
        rgb = im2uint8(im);
    end
end

function im2 = resize_to_h(im, target_h)
    % Resize image to target height, preserving aspect ratio
    if size(im, 1) == target_h
        im2 = im;
    else
        im2 = imresize(im, [target_h, NaN]);
    end
end

function make_labeled_comparison(im1, im2, im3, out_path)
    % Render three images side-by-side with titles using figure export.
    % Pure base MATLAB — no Computer Vision Toolbox needed.
    W = [size(im1,2), size(im2,2), size(im3,2)];
    H = size(im1, 1);
    W_total = sum(W);

    f = figure('Visible', 'off', 'Units', 'pixels', ...
        'Position', [100, 100, W_total, H + 40], ...
        'Color', [0.15, 0.15, 0.15]);

    t = tiledlayout(1, 3, 'TileSpacing', 'none', 'Padding', 'tight');
    titles = {'Original', 'Hole', 'regionfill'};
    ims = {im1, im2, im3};

    for k = 1:3
        ax = nexttile;
        imshow(ims{k});
        title(ax, titles{k}, 'Color', 'white', ...
            'FontSize', max(10, round(H / 22)), ...
            'FontWeight', 'bold');
        ax.TitleHorizontalAlignment = 'center';
    end

    exportgraphics(f, out_path, 'Resolution', 150, 'BackgroundColor', [0.15, 0.15, 0.15]);
    close(f);
end
