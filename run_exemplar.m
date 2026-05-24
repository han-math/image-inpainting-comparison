% Q1 Inpainting — Method 2: Exemplar-based (Criminisi 2004)
% Uses MATLAB built-in inpaintExemplar (IPT R2019b+).
% Compares with Method 1 (regionfill) to evaluate texture-aware inpainting.

input_dir  = '/Users/xiaohan/Downloads/第五次小作业/input';
output_dir = '/Users/xiaohan/Downloads/第五次小作业/results_exemplar';
if ~exist(output_dir, 'dir'), mkdir(output_dir); end

pairs = {
    {'bricks.png',               'bricks_L_region_lost.png'};
    {'crayon_paint.png',         'crayon_missingRegion.png'};
    {'fingerprint256.png',        'finger_circle_lost5.png'};
    {'white_house.png',          'white_house_with_lost.png'};
};

fprintf('=== Exemplar-based Inpainting (Criminisi 2004) ===\n');

for i = 1:size(pairs, 1)
    orig_name = pairs{i}{1};
    hole_name = pairs{i}{2};

    orig = im2double(imread(fullfile(input_dir, orig_name)));
    hole = im2double(imread(fullfile(input_dir, hole_name)));

    if size(orig,3) > 3, orig = orig(:,:,1:3); end
    if size(hole,3) > 3, hole = hole(:,:,1:3); end
    assert(isequal(size(orig), size(hole)), 'Size mismatch: %s vs %s', orig_name, hole_name);

    fprintf('\n--- %s (%d x %d, ch=%d) ---\n', orig_name, size(orig,1), size(orig,2), size(orig,3));

    % Build mask (same as regionfill)
    if size(orig, 3) == 1
        mask = abs(orig - hole) > 5/255;
    else
        mask = any(abs(orig - hole) > 5/255, 3);
    end
    mask = imfill(mask, 'holes');
    mask = bwareaopen(mask, 20);
    fprintf('  Hole pixels: %d (%.1f%%)\n', sum(mask(:)), 100*sum(mask(:))/numel(mask));

    % inpaintExemplar handles both grayscale and RGB directly
    filled = inpaintExemplar(hole, mask);

    % PSNR / SSIM
    psnr_val = psnr(filled, orig);
    ssim_val = ssim(filled, orig);
    fprintf('  PSNR: %.2f dB   SSIM: %.4f\n', psnr_val, ssim_val);

    % Save
    [~, name, ~] = fileparts(orig_name);
    filled_u8 = im2uint8(filled);
    imwrite(filled_u8, fullfile(output_dir, [name '_exemplar.png']));
    imwrite(mask,       fullfile(output_dir, [name '_mask.png']));

    % Comparison figure (Original | Hole | Exemplar)
    orig_rgb   = gray_to_rgb_u8(orig);
    hole_rgb   = gray_to_rgb_u8(hole);
    filled_rgb = gray_to_rgb_u8(filled_u8);
    H = max([size(orig_rgb,1), size(hole_rgb,1), size(filled_rgb,1)]);
    orig_rgb   = resize_to_h(orig_rgb, H);
    hole_rgb   = resize_to_h(hole_rgb, H);
    filled_rgb = resize_to_h(filled_rgb, H);
    make_labeled_comparison(orig_rgb, hole_rgb, filled_rgb, ...
        fullfile(output_dir, [name '_compare.png']));
    fprintf('  Saved: %s_exemplar.png, %s_compare.png\n', name, name);
end

fprintf('\n=== Done ===\n');

% Helper functions
function rgb = gray_to_rgb_u8(im)
    if size(im, 3) == 1
        rgb = repmat(im2uint8(im), [1, 1, 3]);
    else
        rgb = im2uint8(im);
    end
end

function im2 = resize_to_h(im, target_h)
    if size(im, 1) == target_h
        im2 = im;
    else
        im2 = imresize(im, [target_h, NaN]);
    end
end

function make_labeled_comparison(im1, im2, im3, out_path)
    W = [size(im1,2), size(im2,2), size(im3,2)];
    H = size(im1, 1);
    f = figure('Visible', 'off', 'Units', 'pixels', ...
        'Position', [100, 100, sum(W), H + 40], ...
        'Color', [0.15, 0.15, 0.15]);
    t = tiledlayout(1, 3, 'TileSpacing', 'none', 'Padding', 'tight');
    titles = {'Original', 'Hole', 'Exemplar'};
    ims = {im1, im2, im3};
    for k = 1:3
        ax = nexttile;
        imshow(ims{k});
        title(ax, titles{k}, 'Color', 'white', ...
            'FontSize', max(10, round(H / 22)), 'FontWeight', 'bold');
    end
    exportgraphics(f, out_path, 'Resolution', 150, 'BackgroundColor', [0.15, 0.15, 0.15]);
    close(f);
end
