import argparse
from pathlib import Path
from PIL import Image
from tqdm import tqdm

def process_evaluation_directory(eval_dir: Path):
    """
    Looks for preds/*/viz folder in the given evaluation directory.
    Splits images into ground_truth and prediction folders.
    """
    preds_dir = eval_dir / "preds"
    if not preds_dir.exists():
        return

    # Iterate over models (e.g. best, omniparser, etc.)
    for model_dir in preds_dir.iterdir():
        if not model_dir.is_dir():
            continue
            
        viz_dir = model_dir / "viz"
        if not viz_dir.exists():
            continue
            
        print(f"Processing visualizations for model '{model_dir.name}' in {eval_dir.name}")
        
        # Create output dirs inside a new 'viz_separated' folder
        out_root = model_dir / "viz_separated"
        gt_dir = out_root / "ground_truth"
        pred_dir = out_root / "prediction"
        
        gt_dir.mkdir(parents=True, exist_ok=True)
        pred_dir.mkdir(parents=True, exist_ok=True)
        
        viz_files = list(viz_dir.glob("*.viz.jpg"))
        count = 0
        
        for viz_file in tqdm(viz_files, desc=f"  Splitting {model_dir.name}", unit="img"):
            try:
                with Image.open(viz_file) as img:
                    width, height = img.size
                    half_width = width // 2
                    
                    if half_width == 0:
                        continue
                        
                    # Split horizontally: Left is GT, Right is Prediction
                    gt_img = img.crop((0, 0, half_width, height))
                    pred_img = img.crop((half_width, 0, width, height))
                    
                    # Recover original stem (e.g. "site-id.viz.jpg" -> "site-id")
                    original_stem = viz_file.name.replace(".viz.jpg", "")
                    
                    gt_path = gt_dir / f"{original_stem}.jpg"
                    pred_path = pred_dir / f"{original_stem}.jpg"
                    
                    gt_img.save(gt_path, quality=100)
                    pred_img.save(pred_path, quality=100)
                    count += 1
            except Exception as e:
                print(f"Error processing {viz_file}: {e}")
        
        print(f"  Saved {count} separated images to {out_root}")

def main():
    parser = argparse.ArgumentParser(
        description="Split .viz.jpg into Ground Truth and Prediction images for inspection/papers."
    )
    parser.add_argument(
        "--eval_root",
        type=str,
        required=True,
        help="Root directory containing one or more evaluation folders."
    )
    
    args = parser.parse_args()
    root = Path(args.eval_root)
    
    if not root.exists():
        print(f"Root path does not exist: {root}")
        return

    # Check if root is itself an evaluation dir (has 'preds') or a container of them
    if (root / "preds").exists():
        process_evaluation_directory(root)
    else:
        # Assume it contains multiple evaluation directories
        found = False
        for entry in root.iterdir():
            if entry.is_dir() and (entry / "preds").exists():
                process_evaluation_directory(entry)
                found = True
        
        if not found:
            print(f"No evaluation directories (subdirs creating 'preds/...') found in {root}")

if __name__ == "__main__":
    main()
