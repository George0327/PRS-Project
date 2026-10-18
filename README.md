# Romanian License Plate Recognition

A Windows-focused prototype that detects Romanian license plates in images, segments characters, and recognizes them with a lightweight neural classifier. The solution pairs a C++ OpenCV pipeline for detection/segmentation with a Python/PyTorch training script that produces an ONNX model consumed at runtime.

## What the project does
- Scans vehicle images, locates plate candidates using morphological preprocessing and contour heuristics tuned for Romanian aspect ratios.
- Segments individual glyphs from detected plates, normalizing them to 32×32 monochrome crops.
- Extracts 30 handcrafted features per glyph (Hu moments, spatial densities, projections, bounding-box metrics, contour count).
- Classifies characters via a shallow MLP exported to ONNX and run through OpenCV DNN, targeting the charset 0–9 and A–Z without I/O.
- Organizes intermediate artifacts: binarized plates, segmented characters, and label mappings used by both training and inference.

## Key components
- C++ inference app: [OpenCVApplication.cpp](OpenCVApplication.cpp) drives preprocessing, detection, segmentation, feature extraction, ONNX loading, and console reporting.
- Shared helpers: [common.h](common.h) and [common.cpp](common.cpp) wrap Windows dialogs and simple image utilities.
- Model training: [train_classifier.py](train_classifier.py) mirrors the C++ feature pipeline, trains an MLP in PyTorch, exports `mlp_classifier.onnx`, and writes `label_mapping.txt`.
- Dataset prep: [preprocess_training_data.py](preprocess_training_data.py) binarizes per-class glyph crops into a normalized set under `TrainingData_BW`.

## Project diagram

```
							  Inference path (C++)
	  DataSet/ images ──▶ preprocess (blackhat + Sobel) ──▶ plate candidates
												   │
												   ▼
						   segmentation (contours, morphology, L→R ordering)
												   │
												   ▼
						 32×32 chars → feature extractor (30D handcrafted)
												   │
												   ▼
						  ONNX MLP (OpenCV DNN) → predicted glyphs → console

							  Training path (Python)
TrainingData/ chars ──▶ preprocess_training_data.py (binarize to 32×32) ──▶ TrainingData_BW/
												   │
												   ▼
				   train_classifier.py (same 30D features) ──▶ PyTorch MLP
												   │
									  exports ONNX + label_mapping.txt
```

## Data and artifacts
- Input vehicle photos live in `DataSet/`.
- Character-class folders follow `TrainingData/CNN letter Dataset/<class>` and the binarized counterpart in `TrainingData_BW/...` for consistent training/inference preprocessing.
- Generated outputs: `PreprocessedImages/` (plate masks), `Characters/` (segmented glyphs), `mlp_classifier.onnx`, `label_mapping.txt`, and optional `best_model.pth` from PyTorch training.

## Design notes
- Detection relies on blackhat filtering, Sobel gradients, Otsu thresholding, and contour geometry (area and aspect ratio) rather than a heavy detector, keeping runtime light for classroom/demo use.
- Segmentation uses simple morphological cleanup and left-to-right contour ordering; it favors frontal plates with moderate contrast.
- Feature-based MLP keeps the model tiny and interpretable; ONNX export aligns Python preprocessing with the C++ feature extractor to avoid train/serve skew.
- Charset omits I and O to reduce ambiguity with 1 and 0 in Romanian plates.

## Current limitations
- Performance degrades on extreme angles, motion blur, or plates outside expected aspect ratios.
- The heuristic detector may miss plates in cluttered scenes; no multi-plate handling is implemented.
- Character segmentation assumes reasonably separated glyphs; touching or heavily distorted characters are not explicitly handled.

## Build and run

Requirements: Windows, Visual Studio 2022, OpenCV 4.9.0 (the course `OpenCVApplication-VS2022_OCV490_basic` template, whose headers are included under `OpenCV/`), and Python 3 with PyTorch for training.

**Inference**

1. Open `OpenCVApplication.sln` and build in Visual Studio.
2. Run from the project directory so that `DataSet/`, `mlp_classifier.onnx` and `label_mapping.txt` are found. The program processes every image in `DataSet/` and writes the intermediate results to `PreprocessedImages/` and `Characters/`.

**Training**

The training images are not included in the repository. Put one folder of character crops per class under `TrainingData/CNN letter Dataset/<class>/`, then run:

```bash
python preprocess_training_data.py   # binarise and normalise the crops
python train_classifier.py           # train, then export mlp_classifier.onnx and label_mapping.txt
```
