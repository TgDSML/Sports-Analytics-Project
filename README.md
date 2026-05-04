# sports-analytics-project

A Python starter project for sports analytics using video and deep learning. The project is organized for computer vision workflows such as object detection, tracking, and downstream analytics.

## Setup

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/sports-analytics-project.git
cd sports-analytics-project
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate the environment.

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Linux/macOS:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

You can also run the setup script for your platform:

Windows:

```bat
setup.bat
```

Linux/macOS:

```bash
chmod +x setup.sh
./setup.sh
```

## Run

Place a sample video at `data/sample.mp4`, then run:

```bash
python main.py --video data/sample.mp4
```

## Project Structure

```text
sports-analytics-project/
├── data/              # Local input videos and datasets
├── notebooks/         # Experiments and exploratory analysis
├── outputs/           # Generated results, clips, plots, and reports
├── src/
│   ├── detection/     # Object detection code
│   ├── tracking/      # Object tracking code
│   ├── analytics/     # Sports metrics and analysis code
│   └── utils/         # Shared helper functions
├── main.py            # Minimal video entry point
├── requirements.txt   # Python dependencies
├── setup.sh           # Linux/macOS setup script
├── setup.bat          # Windows setup script
└── README.md
```

## Notes

- Files in `data/` and `outputs/` are ignored by Git except for `.gitkeep` placeholders.
- Ultralytics YOLO model weights are downloaded automatically the first time a model is used.
- Keep large videos, datasets, model weights, and generated outputs out of Git.
