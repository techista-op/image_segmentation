def compute_fg_ratio(mask):
    """
    mask: (H, W) binary tensor or array
    Returns: float between 0 and 1
    Also returns bucket: "0-5%", "5-15%", "15-100%"
    """
    pass

# ============================================================
# DIAGNOSTIC PIPELINE — OUTLINE
# ============================================================

# ---- DATA SAMPLING ----

def sample_100_per_subset(dataset_root, split_file, n=100):
    """
    From the test split, sample n images per subset
    (HCOCO, HAdobe5k, HFlickr, Hday2night).
    Returns a dict: {subset_name: [list of (composite_path, 
                     gt_path, mask_path)]}
    """
    pass


# ---- INFERENCE ----

def run_inference_harmonizer(image_paths, model, device):
    """
    Run base Harmonizer on a list of images.
    Returns dict: {img_name: pred_tensor (C,H,W, 0-1)}
    """
    pass

def run_inference_ours(image_paths, model, device):
    """
    Run your model on a list of images.
    Returns dict: {img_name: (pred_tensor, arg_maps)}
    arg_maps shape: (6, H, W), values in [-1, 1]
    This is the only model that also returns argument maps.
    """
    pass

def run_inference_hdnet(image_paths, model, device):
    """
    Run HDNet on a list of images.
    Returns dict: {img_name: pred_tensor (C,H,W, 0-1)}
    """
    pass


# ---- METRICS ----

def compute_metrics(pred, gt, mask):
    """
    Given pred, gt, mask tensors (C,H,W, 0-1):
    Returns dict with keys: mse, fmse, psnr
    psnr uses MAX=255 convention.
    fmse normalizes error by foreground pixel count only.
    """
    pass

def compute_all_metrics(predictions, gt_dict, mask_dict):
    """
    Runs compute_metrics for every image in predictions dict.
    Returns nested dict:
    {img_name: {mse, fmse, psnr}}
    """
    pass

def aggregate_metrics_by_subset(per_image_metrics, subset_map):
    """
    Groups per-image metrics by subset and averages.
    Returns:
    {subset_name: {model_name: {mse, fmse, psnr}}}
    Also returns overall average across all subsets.
    """
    pass

def build_comparison_table(aggregated_metrics):
    """
    Prints/saves a formatted comparison table:
    Subset | Harmonizer PSNR/MSE/fMSE | Ours | HDNet
    Also computes: your gain vs Harmonizer, your gap vs HDNet.
    Saves as comparison_table.csv
    """
    pass


# ---- ERROR MAPS ----

def compute_error_map(pred, gt, mask):
    """
    Computes per-pixel squared error averaged across RGB.
    Zeroes out background (mask=0).
    Returns: (H, W) tensor — foreground error only.
    """
    pass

def visualize_single_image(composite, pred_harmonizer, 
                            pred_ours, pred_hdnet,
                            gt, mask, error_map_harmonizer,
                            error_map_ours, error_map_hdnet,
                            arg_maps, img_name, save_dir):
    """
    Saves a single multi-panel figure for one image:
    Row 1: composite | gt | mask
    Row 2: harmonizer pred | ours pred | hdnet pred
    Row 3: harmonizer error | ours error | hdnet error
    Row 4: 6 argument map channels (your model only)
    All error maps use same colorscale (vmax=shared max)
    for fair visual comparison.
    Saves to save_dir/img_name.png
    """
    pass

def compute_dataset_error_heatmap(error_maps_dict, mask_dict):
    """
    Aggregates per-image foreground error maps into one
    dataset-level spatial heatmap.
    Resizes every foreground bbox to canonical 256x256,
    accumulates weighted by mask, normalizes.
    Returns: (256, 256) average error map.
    Call separately for each model.
    Saves heatmap as dataset_heatmap_{model_name}.png
    """
    pass


# ---- ARGUMENT MAP ANALYSIS ----

def analyze_argument_maps(arg_maps_dict):
    """
    For each image's 6 argument maps, computes:
    - per-channel mean, std, min, max
    - is_collapsed flag: std < threshold (e.g. 0.01)
    - is_dead flag: abs(mean) < threshold (e.g. 0.005)
    - texture_correlation: correlation of arg map gradient
      with image gradient (high = following texture, bad)
    Returns: {img_name: {channel_i: {mean, std, 
              is_collapsed, is_dead, texture_correlation}}}
    """
    pass

def visualize_argument_maps(arg_maps, composite, img_name, 
                             save_dir):
    """
    Saves a figure with 6 argument map channels as heatmaps
    (RdBu colormap, vmin=-1, vmax=1).
    Title of each panel shows filter name + std value.
    Side by side with composite image for reference.
    Saves to save_dir/args_{img_name}.png
    """
    pass

def summarize_argument_map_stats(arg_maps_analysis):
    """
    Across all images, reports:
    - What % of images have collapsed maps per channel
    - What % have dead channels per channel
    - Average texture correlation per channel
    - Which channels are most/least spatially active
    Prints summary + saves as arg_stats.csv
    """
    pass


# ---- FAILURE TAXONOMY ----

def get_worst_and_best_cases(per_image_metrics, 
                              model_name, n=20):
    """
    Sorts images by fMSE for a given model.
    Returns: (worst_n, best_n) — lists of img_names.
    """
    pass

def classify_failure_mode(error_map, arg_maps, 
                           composite, mask):
    """
    Heuristic auto-classifier for failure type.
    Checks:
    - boundary_artifact: error concentrated at mask edge
      (erosion test: does eroding mask 5px drop error >30%?)
    - global_color_shift: error uniform across foreground
      (spatial std of error map is low)
    - one_sided_illumination: error asymmetric left/right
      or top/bottom (compare quadrant means)
    - dead_channel: any arg map channel is dead
    - texture_following: high texture correlation in arg maps
    Returns: {failure_type: bool, confidence: float}
    """
    pass

def build_failure_taxonomy(worst_cases, per_image_metrics,
                            error_maps, arg_maps_dict,
                            composite_dict, mask_dict):
    """
    Runs classify_failure_mode on worst N images.
    Also checks: does HDNet also fail on these images?
    (compare hdnet fmse on same images)
    Saves:
    - failure_taxonomy.csv: img_name, failure_types,
      our_fmse, hdnet_fmse, hdnet_also_fails flag
    - A grid figure of worst 20 images with their
      classified failure type as title
    """
    pass


# ---- SUBSET SPECIFIC ANALYSIS ----

def hadobe5k_large_fg_analysis(per_image_metrics, 
                                error_maps, mask_dict):
    """
    For HAdobe5k only — splits images by fg ratio
    (small <15% vs large >15%) and compares error patterns.
    Key question: does error map on large FG show
    spatially varying pattern (local problem) or 
    uniform fill (global problem)?
    Saves: hadobe_fg_size_analysis.png
    """
    pass


# ---- MASTER RUNNER ----

def run_full_diagnostic(config):
    """
    Orchestrates everything in the right order:
    
    1. sample_100_per_subset
    2. run_inference for all 3 models
    3. compute_all_metrics for all 3 models
    4. aggregate_metrics_by_subset
    5. build_comparison_table
    6. compute_error_map for all images x all models
    7. visualize_single_image for all 400 images
    8. compute_dataset_error_heatmap for all 3 models
    9. analyze_argument_maps (your model only)
    10. summarize_argument_map_stats
    11. get_worst_and_best_cases (your model)
    12. build_failure_taxonomy (worst 20)
    13. hadobe5k_large_fg_analysis
    
    All outputs go to config.output_dir with subfolders:
    /per_image_panels/
    /arg_maps/
    /heatmaps/
    /taxonomy/
    /tables/
    """
    pass


# ---- CONFIG ----

if __name__ == "__main__":
    config = {
        "dataset_root": "/path/to/iHarmony4",
        "split_file":   "/path/to/test_split.txt",
        "output_dir":   "./diagnostic_output",
        "device":       "cuda",
        
        # model checkpoints
        "harmonizer_ckpt": "/path/to/harmonizer.pth",
        "ours_ckpt":       "/path/to/our_model.pth",
        "hdnet_ckpt":      "/path/to/hdnet.pth",
        
        # sampling
        "n_per_subset": 100,
        "seed": 42,  # for reproducible sampling
        
        # thresholds for failure classification
        "collapse_std_threshold":  0.01,
        "dead_mean_threshold":     0.005,
        "boundary_erosion_px":     5,
        "boundary_drop_threshold": 0.30,
    }
    
    run_full_diagnostic(config)
