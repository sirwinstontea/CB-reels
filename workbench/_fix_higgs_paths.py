from pathlib import Path

p = Path(__file__).resolve().parent / "run_higgs_s01.py"
t = p.read_text(encoding="utf-8")
t = t.replace('cour.run(job.name, cid, model, f"{SID}.plate.mp4", refs,',
              "cour.run(job.name, cid, model, str(plate), refs,")
t = t.replace('cour.run(job.name, hcid, hmodel, f"{SID}.rough.mp4", shot_refs,',
              "cour.run(job.name, hcid, hmodel, str(rough), shot_refs,")
p.write_text(t, encoding="utf-8")
print("ok")
