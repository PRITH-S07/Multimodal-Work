# Installation and Usage Guide

This guide will help set up and run the Human Pose Estimation Model Comparison Framework that I created here.

## Prerequisites

- Python 3.8 or higher
- pip (Python package installer)
- Git (optional, for cloning the repository)

## Installation Steps

1. Navigate to the model_comparison folder:

```bash
cd model_comparison
```

2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Running the Framework

After installing the dependencies, run the HPE model comparison framework with:

```bash
python hpe_model_comparison.py
```

By default, this will:
1. Load various HPE models (ViTPose, OpenPose, HRNet, DEKR)
2. Analyze a sample image with a provided text prompt
3. Generate comparative visualizations
4. Save the visualizations as PNG files in the current directory

## Customization

To analyze images and text prompts of one's choice, one can modify the following lines at the bottom of `hpe_model_comparison.py`:

```python
if __name__ == "__main__":
    image_path = "/path/to/your/image.jpg"  # Change this to your image path
    text_prompt = "Your custom text description"  # Change this to your text prompt
    
    analyze_hpe_models(image_path, text_prompt)
```

## Output Files

After running the script, the following visualization files shall be created in the directory:
- feature_importance.png
- token_similarity.png
- token_importance.png
