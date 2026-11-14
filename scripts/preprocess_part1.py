"""Create canonical Part 1 pinhole RGB inputs; leave raw/staging data unchanged."""
import csv, hashlib, json, shutil, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools/python'))
import cv2
import numpy as np
from PIL import Image
SCENES=['Waymo-405841','DL3DV-2','Re10k-1']
def digest(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def parse_calib(p):
 import re
 s=p.read_text(); names=['fx','fy','cx','cy','k1','k2','p1','p2','k3']
 return {name:float(re.search(r'\b'+name+r':\s*([^\s]+)',s).group(1)) for name in names}
rows=[]; summaries={}
for scene in SCENES:
 source=ROOT/'data/prepared/part1'/scene
 if not source.exists():raise RuntimeError(f'Run prepare_data.py first: {source}')
 outroot=ROOT/'data/processed/part1'/scene
 records=[];cache={};scene_meta={'input_stage':'raw RGB staging','output_stage':'canonical pinhole RGB','scene':scene}
 for split in ['train','test']:
  folder=outroot/split/'images';folder.mkdir(parents=True,exist_ok=True)
  for src in sorted((source/split/'images').iterdir()):
   if src.suffix.lower() not in ['.png','.jpg','.jpeg']:continue
   out=folder/src.name
   with Image.open(src) as image: old_size=image.size; old_mode=image.mode
   if old_mode!='RGB':raise RuntimeError(f'Unexpected mode: {src}')
   if scene=='Waymo-405841':
    if old_size!=(1920,1280):raise RuntimeError('Waymo resolution does not match official calibration')
    calib_file=ROOT/'data/raw/405841/FRONT/calib'/f'{src.stem}.txt'
    c=parse_calib(calib_file);key=tuple(c.items())
    if key not in cache:
     K=np.array([[c['fx'],0,c['cx']],[0,c['fy'],c['cy']],[0,0,1.]],dtype=np.float64)
     D=np.array([c[k] for k in ['k1','k2','p1','p2','k3']])
     Knew,roi=cv2.getOptimalNewCameraMatrix(K,D,old_size,0,old_size,centerPrincipalPoint=True)
     inflation=1.0
     for iteration in range(101):
      mx,my=cv2.initUndistortRectifyMap(K,D,np.eye(3),Knew,old_size,cv2.CV_32FC1)
      valid=(mx>=0)&(mx<old_size[0]-1)&(my>=0)&(my<old_size[1]-1)
      if valid.all():break
      Knew[0,0]*=1.001;Knew[1,1]*=1.001;inflation*=1.001
     else:raise RuntimeError('Unable to form entirely valid pinhole output')
     finalKcv=Knew.copy()
     finalKcv[0,0]*=.5;finalKcv[1,1]*=.5
     finalKcv[0,2]=.5*(Knew[0,2]+.5)-.5
     finalKcv[1,2]=.5*(Knew[1,2]+.5)-.5
     finalK=finalKcv.copy();finalK[0,2]+=.5;finalK[1,2]+=.5
     cache[key]=(mx,my,finalK,{'original_K':K.tolist(),'distortion_k1_k2_p1_p2_k3':D.tolist(),'rectified_native_K':Knew.tolist(),'final_K':finalK.tolist(),'final_K_colmap':finalK.tolist(),'final_K_opencv':finalKcv.tolist(),'pixel_origin_note':'OpenCV pixel centers start at (0,0); resize uses c_dst=scale*(c_src+0.5)-0.5. COLMAP centers start at (0.5,0.5), so add 0.5 to final principal point.','native_size':list(old_size),'final_size':[960,640],'alpha':0,'centerPrincipalPoint':True,'focal_safety_factor':inflation,'valid_remap_fraction':float(valid.mean())})
    mx,my,finalK,trans=cache[key]
    rgb=np.asarray(Image.open(src).convert('RGB'))
    undistorted=cv2.remap(rgb,mx,my,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)
    output=cv2.resize(undistorted,(960,640),interpolation=cv2.INTER_AREA)
    Image.fromarray(output).save(out)
    method='opencv_pinhole_centered_alpha0_then_half_area'
   else:
    if not out.exists():shutil.copy2(src,out)
    if digest(src)!=digest(out):raise RuntimeError(f'Unchanged RGB output differs: {out}')
    trans={'native_size':list(old_size),'final_size':list(old_size),'operation':'identity byte copy'}
    method='identity'
   with Image.open(out) as final:final.load();new_size=final.size
   record={'scene':scene,'split':split,'filename':src.name,'source_path':src.relative_to(ROOT).as_posix(),'output_path':out.relative_to(ROOT).as_posix(),'source_sha256':digest(src),'output_sha256':digest(out),'source_width':old_size[0],'source_height':old_size[1],'width':new_size[0],'height':new_size[1],'operation':method}
   rows.append(record);records.append(record)
 scene_meta.update({'counts':{s:sum(r['split']==s for r in records) for s in ['train','test']},'preprocessing':list(x[3] for x in cache.values()) if cache else trans,'images':records,'opencv_version':cv2.__version__,'note':'Reference calibration is used for shared Waymo rectification only. Reference poses/depth are never initializer inputs.'})
 (outroot/'preprocessing.json').write_text(json.dumps(scene_meta,indent=2),encoding='utf-8')
 summaries[scene]={k:v for k,v in scene_meta.items() if k!='images'}
 print(scene,len(records),'canonical RGB outputs checked',flush=True)
with (ROOT/'data/manifests/processed_frames.csv').open('w',newline='',encoding='utf-8') as f:
 writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
manifest={'version':'part1-data-v1','opencv_version':cv2.__version__,'scene_summaries':summaries,'image_count':len(rows),'matrix_note':'Waymo final K maps rays into the canonical processed image grid; DL3DV/Re10k normalized intrinsics still apply because image bytes are unchanged. Pose interpretation is independent of rectification.'}
(ROOT/'data/manifests/preprocessing.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
print(json.dumps({s:v['counts'] for s,v in summaries.items()},indent=2),flush=True)
