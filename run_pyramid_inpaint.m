% Q1 Inpainting — Method 2: Multi-scale Pyramid Inpainting
% Strategy: downsample → regionfill at coarse level → upsample → refine at finer level
% 3 levels (L=2): level0=original, level1=1/2, level2=1/4
% Coarsest level uses regionfill; finer levels use inpaintExemplar.

input_dir  = '/Users/xiaohan/Downloads/第五次小作业/input';
output_dir = '/Users/xiaohan/Downloads/第五次小作业/results_pyramid';
if ~exist(output_dir, 'dir'), mkdir(output_dir); end

pairs = {
    {'bricks.png',               'bricks_L_region_lost.png'};
    {'crayon_paint.png',         'crayon_missingRegion.png'};
    {'fingerprint256.png',        'finger_circle_lost5.png'};
    {'white_house.png',          'white_house_with_lost.png'};
};

level_list = [2, 3, 4];

fprintf('=== Multi-scale Pyramid Inpainting (L=2,3,4) ===\n');

for i = 1:size(pairs, 1)
    orig_name = pairs{i}{1};
    hole_name = pairs{i}{2};

    orig = im2double(imread(fullfile(input_dir, orig_name)));
    hole = im2double(imread(fullfile(input_dir, hole_name)));

    if size(orig,3) > 3, orig = orig(:,:,1:3); end
    if size(hole,3) > 3, hole = hole(:,:,1:3); end
    assert(isequal(size(orig), size(hole)), 'Size mismatch: %s vs %s', orig_name, hole_name);

    fprintf('\n--- %s (%d x %d, ch=%d) ---\n', orig_name, size(orig,1), size(orig,2), size(orig,3));

    % Build mask
    if size(orig, 3) == 1
        mask = abs(orig - hole) > 5/255;
    else
        mask = any(abs(orig - hole) > 5/255, 3);
    end
    mask = imfill(mask, 'holes');
    mask = bwareaopen(mask, 20);
    fprintf('  Hole pixels: %d (%.1f%%)\n', sum(mask(:)), 100*sum(mask(:))/numel(mask));

    % Test each pyramid depth 
    [~, name, ~] = fileparts(orig_name);
    best_psnr = -inf;
    best_ssim = -inf;
    best_L_psnr = 0;
    best_L_ssim = 0;

    for nl = level_list
        filled = pyramid_inpaint(hole, mask, nl);

        psnr_val = psnr(filled, orig);
        ssim_val = ssim(filled, orig);
        fprintf('  L=%d: PSNR %.2f dB   SSIM %.4f\n', nl, psnr_val, ssim_val);

        % Save per-level result
        imwrite(im2uint8(filled), ...
            fullfile(output_dir, sprintf('%s_pyramid_L%d.png', name, nl)));

        % Track best
        if psnr_val > best_psnr
            best_psnr = psnr_val;
            best_L_psnr = nl;
        end
        if ssim_val > best_ssim
            best_ssim = ssim_val;
            best_L_ssim = nl;
        end
    end

    fprintf('  Best PSNR: L=%d (%.2f dB)   Best SSIM: L=%d (%.4f)\n', ...
        best_L_psnr, best_psnr, best_L_ssim, best_ssim);
end

fprintf('\n=== Done ===\n');


% Core: multi-scale pyramid inpainting (grayscale or RGB)
% Coarsest level: regionfill (PDE — coarse structure, small hole)
% Finer levels:   inpaintExemplar (Criminisi 2004 — texture-aware)

function filled = pyramid_inpaint(I, mask, n_levels)
    ch = size(I, 3);

    % Build pyramids (image + mask)
    I_pyr    = cell(n_levels, 1);
    mask_pyr = cell(n_levels, 1);
    I_pyr{1}    = I;
    mask_pyr{1} = mask;
    for k = 2:n_levels
        scale = 1 / 2^(k-1);
        I_pyr{k}    = imresize(I, scale);
        mask_pyr{k} = logical(imresize(mask, scale, 'method', 'nearest'));
    end

    % Level L (coarsest): regionfill (per-channel if RGB)
    L = n_levels;
    if ch == 1
        filled = regionfill(I_pyr{L}, mask_pyr{L});
    else
        filled = I_pyr{L};
        for c = 1:ch
            filled(:,:,c) = regionfill(I_pyr{L}(:,:,c), mask_pyr{L});
        end
    end

    % Coarse → fine: upsample, composite, then exemplar at each level
    for k = L-1:-1:1
        up = imresize(filled, [size(I_pyr{k},1), size(I_pyr{k},2)]);

        % Composite: hole ← upsampled coarse fill, known ← original
        composite = I_pyr{k};
        mask_3d = repmat(mask_pyr{k}, [1, 1, ch]);
        composite(mask_3d) = up(mask_3d);

        % Refine with exemplar-based inpainting
        filled = inpaintExemplar(composite, mask_pyr{k});
    end
end


% Helper functions (shared with run_regionfill.m)
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
    titles = {'Original', 'Hole', 'Pyramid'};
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
