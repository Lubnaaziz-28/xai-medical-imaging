"""Configuration management for MedCard."""

import yaml
from pathlib import Path
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field
import os


@dataclass
class Config:
    """Configuration for MedCard model card generation."""
    
    # Model info
    model_path: str = ""
    model_name: str = "Medical Imaging Model"
    model_version: str = "1.0.0"
    author: str = "Lubna Research Team"
    author_email: str = "research@lubna.ai"
    
    # Data
    data_path: str = ""
    data_format: str = "directory"  # directory, csv, numpy
    class_names: List[str] = field(default_factory=list)
    threshold: float = 0.5
    
    # Output
    output_path: str = "model_card.pdf"
    
    # Model description
    intended_use: str = "Clinical decision support for medical imaging"
    target_population: str = "Patients undergoing medical imaging"
    clinical_context: str = ""
    
    # XAI settings
    generate_xai: bool = True
    xai_methods: List[str] = field(default_factory=lambda: ["gradcam", "shap"])
    gradcam_layer: Optional[str] = None
    gradcam_class_idx: Optional[int] = None
    shap_background_samples: int = 100
    shap_max_evals: int = 500
    num_xai_samples: int = 10
    
    # Fairness settings
    generate_fairness: bool = True
    sensitive_attributes: List[str] = field(default_factory=list)
    privileged_groups: Dict[str, str] = field(default_factory=dict)
    fairness_threshold: float = 0.5
    
    # Metrics
    compute_ci: bool = True
    ci_confidence: float = 0.95
    ci_bootstrap_samples: int = 1000
    
    # Template
    template_path: Optional[str] = None
    include_cover_page: bool = True
    include_appendix: bool = True
    
    # Advanced
    random_seed: int = 42
    verbose: bool = False


def load_config(config_path: str) -> Config:
    """Load configuration from YAML file."""
    config = Config()
    with open(config_path, "r") as f:
        data = yaml.safe_load(f)
    
    if data:
        for key, value in data.items():
            if hasattr(config, key):
                setattr(config, key, value)
    
    # Expand environment variables in paths
    for attr in ["model_path", "data_path", "output_path", "template_path"]:
        value = getattr(config, attr)
        if value:
            setattr(config, attr, os.path.expandvars(os.path.expanduser(value)))
    
    return config


def save_config(config: Config, output_path: str) -> None:
    """Save configuration to YAML file."""
    data = {}
    for key, value in config.__dict__.items():
        if not key.startswith("_"):
            data[key] = value
    
    with open(output_path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)


def create_default_config() -> Config:
    """Create a default configuration with example values."""
    return Config(
        model_path="models/best_model.h5",
        data_path="data/test",
        class_names=["Normal", "Pathology"],
        model_name="Breast Cancer Detection Model",
        model_version="1.0.0",
        author="Dr. Lubna Aziz",
        author_email="engr.lubnaaziz@gmail.com",
        intended_use="Computer-aided detection of breast cancer in mammography",
        target_population="Women aged 40+ undergoing screening mammography",
        clinical_context="",
        sensitive_attributes=["age_group", "breast_density"],
        privileged_groups={"age_group": "50-60", "breast_density": "fatty"},
        xai_methods=["gradcam", "shap"],
        num_xai_samples=15,
        generate_xai=True,
        generate_fairness=True,
    )


DEFAULT_CONFIG_YAML = """
# MedCard Configuration
# Copy this file and modify as needed

# Model Information
model_path: "models/best_model.h5"
model_name: "Breast Cancer Detection Model"
model_version: "1.0.0"
author: "Dr. Lubna Aziz"
author_email: "engr.lubnaaziz@gmail.com"

# Data
data_path: "data/test"
data_format: "directory"  # directory, csv, numpy
class_names:
  - "Normal"
  - "Cancer"
threshold: 0.5

# Output
output_path: "model_card.pdf"

# Model Description
intended_use: "Computer-aided detection of breast cancer in mammography"
target_population: "Women aged 40+ undergoing screening mammography"
clinical_context: |
  This model is intended to assist radiologists in detecting breast cancer 
  in screening mammograms. It is not intended to replace clinical judgment.

# XAI Settings
generate_xai: true
xai_methods:
  - "gradcam"
  - "shap"
gradcam_layer: null  # auto-detect
gradcam_class_idx: null
shap_background_samples: 100
shap_max_evals: 500
num_xai_samples: 10

# Fairness Settings
generate_fairness: true
sensitive_attributes:
  - "age_group"
  - "breast_density"
  - "ethnicity"
privileged_groups:
  age_group: "50-60"
  breast_density: "fatty"
  ethnicity: "white"
fairness_threshold: 0.5

# Metrics
compute_ci: true
ci_confidence: 0.95
ci_bootstrap_samples: 1000

# Template
template_path: null
include_cover_page: true
include_appendix: true

# Advanced
random_seed: 42
verbose: false
"""