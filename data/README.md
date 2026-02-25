# data/ — Training Data Directory

This directory is **gitignored** and is not committed to the repository.

Font files are binary assets and must not be stored in git history.

---

## How to Populate

Run the download script from the repository root:

```bash
python src/model/data/download_fonts.py
```

Optional: set a Google Fonts API key for faster, targeted downloads:

```bash
export GOOGLE_FONTS_API_KEY=<your_key>
python src/model/data/download_fonts.py
```

Get an API key at: https://developers.google.com/fonts/docs/developer_api

---

## Expected Structure After Download

```
data/
└── fonts/
    ├── Roboto/
    │   ├── regular.ttf
    │   ├── bold.ttf
    │   └── italic.ttf
    ├── Open_Sans/
    │   ├── regular.ttf
    │   └── bold.ttf
    ├── Lato/
    │   └── regular.ttf
    └── …  (~400 font families)
```

Only fonts with **both Latin and Cyrillic (Russian)** coverage and an
**OFL (Open Font License)** are downloaded.

Expect: ~300–450 font families, ~2–4 GB total disk space.

---

## License

All downloaded fonts are OFL-licensed.
Training on OFL fonts and distributing generated fonts under OFL is permitted.
See: https://scripts.sil.org/OFL
