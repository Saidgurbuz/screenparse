# Contributing to Webshot

Thank you for your interest in contributing to Webshot! This guide focuses on extending the evaluation framework with new models, metrics, and datasets.

## Quick Start

```bash
# Fork and clone the repository
git clone https://github.com/your-username/webshot-dataset.git
cd webshot-dataset

# Install in development mode
pip install -e .
playwright install chromium
```

## Extending the Evaluation Framework

The evaluation framework is designed to be extensible. You can add new models, metrics, and datasets by implementing the provided interfaces.

### Adding a New Model

To add support for a new model, create a new file in `src/evaluation/models/` and implement the `ModelRunner` interface:

```python
# src/evaluation/models/your_model.py
from typing import Sequence
from ..models.base import ModelRunner
from ..types import EvaluationSample, UIElement, BoundingBox

class YourModelRunner(ModelRunner):
    """Runner for YourModel."""
    
    def __init__(self, model_path: str, **kwargs):
        super().__init__(name="YourModel")
        # Initialize your model here
        self.model = load_your_model(model_path)
        
    def predict(self, sample: EvaluationSample) -> Sequence[UIElement]:
        """
        Run inference on a single sample.
        
        Args:
            sample: EvaluationSample with image_path and ground_truth
            
        Returns:
            List of predicted UIElement objects
        """
        # Load image
        image = load_image(sample.image_path)
        
        # Run your model
        predictions = self.model.predict(image)
        
        # Convert to UIElement format
        elements = []
        for pred in predictions:
            element = UIElement(
                bbox=BoundingBox(
                    x=pred['x'],
                    y=pred['y'], 
                    w=pred['width'],
                    h=pred['height']
                ),
                label=pred['class_name'],
                text=pred.get('text'),
                score=pred.get('confidence')
            )
            elements.append(element)
            
        return elements
    
    def predict_batch(self, samples: Iterable[EvaluationSample]) -> List[Sequence[UIElement]]:
        """Optional: Implement batch prediction for efficiency."""
        # If your model supports batching, implement it here
        # Otherwise, the default implementation will call predict() for each sample
        return [self.predict(s) for s in samples]
    
    def close(self):
        """Optional: Cleanup resources."""
        if hasattr(self, 'model'):
            del self.model
```

**Register your model in the CLI** (`src/evaluation/cli.py`):

```python
# Add argument for your model
parser.add_argument('--your-model', help='Path to YourModel weights')

# In the model loading section:
if args.your_model:
    from .models.your_model import YourModelRunner
    models.append(YourModelRunner(args.your_model))
```

### Adding a New Metric

To add a new evaluation metric, create a file in `src/evaluation/metrics/` and implement the `Metric` interface:

```python
# src/evaluation/metrics/your_metric.py
from typing import Sequence
from ..metrics.base import Metric
from ..types import EvaluationSample, MetricResult, UIElement

class YourMetric(Metric):
    """Your custom evaluation metric."""
    
    def __init__(self, threshold: float = 0.5):
        super().__init__(name="your_metric")
        self.threshold = threshold
        
    def compute(self, sample: EvaluationSample, predictions: Sequence[UIElement]) -> MetricResult:
        """
        Compute metric for a single sample.
        
        Args:
            sample: EvaluationSample with ground_truth elements
            predictions: Predicted UIElement objects
            
        Returns:
            MetricResult with value and optional details
        """
        ground_truth = sample.ground_truth
        
        # Implement your metric logic here
        # Example: compute some score between predictions and ground truth
        score = self._compute_score(predictions, ground_truth)
        
        return MetricResult(
            name=self.name,
            value=score,
            details={
                'num_predictions': len(predictions),
                'num_ground_truth': len(ground_truth),
                'threshold': self.threshold
            }
        )
    
    def _compute_score(self, predictions, ground_truth):
        """Your metric computation logic."""
        # Implement your scoring logic
        return 0.0
    
    def aggregate(self, results: List[MetricResult]) -> MetricResult:
        """
        Optional: Custom aggregation logic.
        Default implementation computes mean of all values.
        """
        # If you need custom aggregation (e.g., weighted average, median)
        # implement it here. Otherwise, use the default from base class.
        return super().aggregate(results)
```

**Register your metric in the CLI** (`src/evaluation/cli.py`):

```python
# In the metrics initialization section:
AVAILABLE_METRICS = {
    'page_iou': PageIoUMetric,
    'label_page_iou': LabelPageIoUMetric,
    'map': MeanAveragePrecisionMetric,
    'your_metric': YourMetric,  # Add your metric here
}

# Metrics are automatically loaded based on --metrics argument
```

### Adding a New Dataset Format

To add support for a new dataset format, add a loader function in `src/evaluation/datasets.py`:

```python
def build_your_dataset(
    root_dir: str,
    split: str = 'test',
    max_samples: Optional[int] = None,
    label_mapper: Optional[LabelMapper] = None,
) -> List[EvaluationSample]:
    """
    Load your custom dataset format.
    
    Args:
        root_dir: Root directory of the dataset
        split: Dataset split (train/val/test)
        max_samples: Maximum number of samples to load
        label_mapper: Optional label mapper for class conversion
        
    Returns:
        List of EvaluationSample objects
    """
    base = Path(root_dir)
    samples: List[EvaluationSample] = []
    
    # Load your dataset structure
    # Example: read annotations file
    annotations_file = base / f'{split}_annotations.json'
    annotations = _read_json(annotations_file)
    
    for ann in annotations[:max_samples] if max_samples else annotations:
        # Load image
        image_path = base / 'images' / ann['image_filename']
        
        # Convert annotations to UIElement format
        elements = []
        for obj in ann['objects']:
            element = UIElement(
                bbox=BoundingBox(
                    x=obj['bbox'][0],
                    y=obj['bbox'][1],
                    w=obj['bbox'][2],
                    h=obj['bbox'][3]
                ),
                label=obj['category'],
                text=obj.get('text'),
                score=1.0  # Ground truth has confidence 1.0
            )
            elements.append(element)
        
        # Apply label mapping if provided
        if label_mapper:
            elements = [label_mapper.map_element(e) for e in elements]
        
        # Create sample
        sample = EvaluationSample(
            image_path=str(image_path),
            ground_truth=elements,
            sample_id=ann['image_id'],
            metadata={'split': split, 'source': 'your_dataset'}
        )
        samples.append(sample)
    
    return samples
```

**Register your dataset in the CLI** (`src/evaluation/cli.py`):

```python
# In the dataset loading section:
if dataset_config['format'] == 'your_dataset':
    from .datasets import build_your_dataset
    samples = build_your_dataset(
        root_dir=dataset_config['root'],
        split=dataset_config.get('split', 'test'),
        max_samples=dataset_config.get('max_samples'),
        label_mapper=label_mapper
    )
```

## Code Style

- Follow PEP 8 style guidelines
- Use type hints for function signatures
- Add docstrings to public functions and classes
- Keep functions focused and modular

## Testing Your Changes

```bash
# Test your model
wsd-eval \
  --format yolo \
  --image-dir data/yolo/images/val \
  --labels-dir data/yolo/labels/val \
  --classes data/yolo/classes.txt \
  --your-model path/to/model \
  --metrics your_metric,page_iou

# Test with a small sample first
wsd-eval --format yolo --image-dir data/yolo/images/val \
  --labels-dir data/yolo/labels/val --classes data/yolo/classes.txt \
  --your-model path/to/model --metrics your_metric \
  --max-samples 10
```

## Submitting Your Contribution

1. **Test thoroughly** with different datasets and configurations
2. **Add documentation** for your new component
3. **Create a pull request** with:
   - Clear description of the new feature
   - Example usage
   - Any dependencies required

## Questions?

Open an issue on GitHub with the `question` label or contact the maintainers.

## License

By contributing, you agree that your contributions will be licensed under the MIT License.