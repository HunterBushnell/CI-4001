from pathlib import Path; import re, pandas as pd
root=Path("/home/fabric/.cache/kagglehub/datasets/wajahat1064/emotion-recognition-using-eeg-and-computer-games/versions/2/Dataset - Emotion Recognition data Based on EEG Signals and Computer Games/Database for Emotion Recognition System Based on EEG Signals and Various Computer Games - GAMEEMO/GAMEEMO"); channel="T7"
subs=sorted([d for d in root.iterdir() if d.is_dir() and re.search(r"S\d{2}", d.name)],
            key=lambda p: int(re.search(r"S(\d{2})", p.name).group(1)))
missing=[]
for d in subs:
    sid=int(re.search(r"S(\d{2})", d.name).group(1))
    ok=False
    for g in (1,2,3,4):
        for base in ["Preprocessed EEG Data",".csv format"]:
            pass
        f=d/"Preprocessed EEG Data"/".csv format"/f"S{sid:02d}G{g}AllChannels.csv"
        if not f.exists(): f=d/"Preprocessed"/".csv format"/f"S{sid:02d}G{g}AllChannels.csv"
        if f.exists():
            df=pd.read_csv(f,nrows=1)
            has=channel in df.columns and not df.columns.str.contains("^Unnamed").all()
            ok = ok or has
    if not ok: missing.append(sid)
print("Subjects missing channel:", missing)
