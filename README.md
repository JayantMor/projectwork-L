# Project Prototype (College Project Demo)

This is a college project demo for action recognition.

## Scope
- Use pre-trained action recognition model (fight vs non-fight)
- Output: event type, confidence, timestamp
- Save short 3–5 sec clip on detection
- Expose results via Flask/FastAPI API
- Simple web dashboard to display alerts + play clips
- Export alerts (CSV/JSON) + auto-delete old clips

## Workflow
1. Work must be done inside this repo (feature branches).
2. Daily commits + updates in CHECKLIST.md
3. All deliverables (code + docs + demo) will be submitted here.

## Project Structure
projectwork-L/
│
├── backend/                # Flask/FastAPI backend
│   ├── app.py              # Main API entry point
│   ├── routes/             # API routes
│   ├── services/           # Helper services (ML inference etc.)
│   ├── requirements.txt    # Python dependencies
│   └── tests/              # Backend test scripts
│
├── frontend/               # Dashboard (basic web UI)
│   ├── index.html          # Dashboard homepage
│   ├── static/             # CSS, JS, images
│   └── templates/          # Jinja2/HTML templates (if Flask used)
│
├── models/                 # ML models (ignored in .gitignore)
│   ├── action_model.pt     # Placeholder, not pushed
│   └── README.md           # Note: download weights separately
│
├── clips/                  # Input/output video clips (ignored)
│   └── sample.mp4          # Example (not pushed to repo)
│
├── notebooks/              # Colab/Jupyter notebooks for ML testing
│   └── prototype.ipynb
│
├── docs/                   # Documentation, usage guides, demo notes
│   └── demo_instructions.md
│
├── .gitignore              # Ignore rules (Python + custom)
├── README.md               # Project overview
├── CHECKLIST.md            # Daily update log (developer fills)
└── CONTRIBUTING.md         # Contribution rules
