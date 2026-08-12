"""
01_extraction / clips_creator.py
---------------------------------------------------------
Pipeline for extracting penalty-kick clips.

Steps:
  1. ANNOTATE — Manual annotation of the T0 frame (foot-ball
     contact instant), kicking foot, goal zone, result, and
     goalkeeper action. Output: dataset_manual.csv

  2. EXTRACT — Automatic cut of 20-frame clips ending
     at T0 (inclusive): [T0-19, ..., T0].
     Output: {video}__event_{id}__t0_{frame}.mp4

  3. CROP — YOLO detects every person in the first frame,
     the user selects the kicker, ByteTrack follows them for
     the remaining 19 frames, letterbox crop 512x1024.
     Output: crop_{video}__event_{id}__t0_{frame}.mp4

Usage:
  python clips_creator.py annotate --videos .                                                — Manually annotate T0 and metadata for each penalty.
  python clips_creator.py extract  --csv dataset_manual.csv --videos . --clips clips_raw/    — Cut 20-frame clips from the annotated CSV.
  python clips_creator.py crop     --clips clips_raw/ --output clips_cropped/                — Isolate the kicker via YOLO and ByteTrack in each clip.
  python clips_creator.py all      --videos . --output processed/                            — Run all three steps in automatic sequence.
"""
# ==============================================================================================================

import cv2, csv, os, json, argparse, numpy as np # type: ignore
from pathlib import Path
from uuid    import uuid4

VIDEO_EXT = {'.mp4','.avi','.mov','.mkv','.mts','.m4v',
             '.MP4','.AVI','.MOV','.MKV'}                     # set instead of list so it doesn't scan every extension
N_FRAMES  = 20
FONT      = cv2.FONT_HERSHEY_SIMPLEX
CSV_COLS  = ["event_id","video_name","t0_frame",
             "kick_foot","goal_zone","gk_guessed","result"]

C_WHITE=(255,255,255); C_BLACK=(0,0,0);    C_YELLOW=(0,220,255)
C_GREEN=(50,220,80);   C_RED=(60,60,240);  C_BLUE=(220,120,30)
C_GRAY=(180,180,180);  C_DARK=(30,30,30);  C_ORANGE=(0,165,255)
C_PINK=(180,80,200)

GOAL_W,GOAL_H=720,500; MARGIN_X=80; MARGIN_Y=55                # window opencv opens for the goal zones
GRID_W=GOAL_W-2*MARGIN_X; GRID_H=280                           # 9-zone grid
CELL_W=GRID_W//3; CELL_H=GRID_H//3                             # cell of each zone
BTN_W,BTN_H=200,44
BTN_X=GOAL_W//2-BTN_W//2; BTN_Y=MARGIN_Y+GRID_H+18             # "BALL OUT" button
SEEK_PAD_X=12; SEEK_H=8; SEEK_Y_OFF=30                         # seekbar (footer)
SPEEDS=[(40,1),(20,2),(10,4),(7,6),(4,10)]                     # (ms, frames)
SPEED_LABELS=["1x","2x","4x","6x","10x"]
MAIN_WIN="Penalty Annotator"; GOAL_WIN="Goal Map"              # windows
ZONE_LABELS={1:"Corner\nTop Left",2:"Top\nCenter",3:"Corner\nTop Right",
             4:"Middle\nLeft",5:"Center",6:"Middle\nRight",
             7:"Corner\nBottom Left",8:"Bottom\nCenter",9:"Corner\nBottom Right"}

def fit_image(img,max_w=1280,max_h=720):                                      # resizes to fit within 1280x720
    h,w=img.shape[:2]; s=min(1.0,max_w/w,max_h/h)
    return (cv2.resize(img,(int(w*s),int(h*s))),s) if s<1.0 else (img,1.0)

def ensure_csv(p):                                                            # creates the csv
    if not os.path.exists(p):
        with open(p,'w',newline='',encoding='utf-8') as f:
            csv.DictWriter(f,fieldnames=CSV_COLS).writeheader()

def append_row(p,row):                                                        # appends one event
    with open(p,'a',newline='',encoding='utf-8') as f:
        csv.DictWriter(f,fieldnames=CSV_COLS).writerow(row)

def list_videos(d):                                                          # ensures videos are always processed in the same order
    return sorted([p for p in Path(d).iterdir() if p.suffix in VIDEO_EXT])

def clip_name(vname,eid,t0):                                                 # builds the clip name
    return f"{Path(vname).stem}__event_{eid}__t0_{t0}.mp4"

# ==============================================================================================================

def build_goal_image(hover=None):                                                                     # opens the goal-zone picker window
    img=np.full((GOAL_H,GOAL_W,3),240,dtype=np.uint8)

    for y in range(GOAL_H):
        v=int(200+40*(y/GOAL_H)); img[y,:]=(min(v,255),min(v+10,255),min(v+20,255))
    shadow=np.array([[MARGIN_X+4,MARGIN_Y+4],[MARGIN_X+GRID_W+4,MARGIN_Y+4],
                     [MARGIN_X+GRID_W+4,MARGIN_Y+GRID_H+4],[MARGIN_X+4,MARGIN_Y+GRID_H+4]],np.int32)

    cv2.fillPoly(img,[shadow],(160,160,160))
    for z in range(1,10):
        r,c=divmod(z-1,3); x0=MARGIN_X+c*CELL_W; y0=MARGIN_Y+r*CELL_H
        col=(180,230,180) if z==hover else C_WHITE
        cv2.rectangle(img,(x0,y0),(x0+CELL_W,y0+CELL_H),col,-1)

    for i in range(4):
        cv2.line(img,(MARGIN_X+i*CELL_W,MARGIN_Y),(MARGIN_X+i*CELL_W,MARGIN_Y+GRID_H),C_BLACK,3 if i in(0,3) else 1)
    for j in range(4):
        cv2.line(img,(MARGIN_X,MARGIN_Y+j*CELL_H),(MARGIN_X+GRID_W,MARGIN_Y+j*CELL_H),C_BLACK,3 if j in(0,3) else 1)
    for z in range(1,10):
        r,c=divmod(z-1,3); cx=MARGIN_X+c*CELL_W+CELL_W//2; cy=MARGIN_Y+r*CELL_H+CELL_H//2
        cv2.putText(img,str(z),(cx-12,cy+8),FONT,1.2,C_RED,3,cv2.LINE_AA)
        for li,ln in enumerate(ZONE_LABELS[z].split("\n")):
            cv2.putText(img,ln,(cx-38,cy+28+li*14),FONT,0.38,C_DARK,1,cv2.LINE_AA)

    cv2.putText(img,"Click where the ball entered",(GOAL_W//2-145,32),FONT,0.7,C_DARK,2,cv2.LINE_AA)
    bc=(60,60,200) if hover==0 else (80,80,180)
    cv2.rectangle(img,(BTN_X,BTN_Y),(BTN_X+BTN_W,BTN_Y+BTN_H),bc,-1)
    cv2.rectangle(img,(BTN_X,BTN_Y),(BTN_X+BTN_W,BTN_Y+BTN_H),C_WHITE,2)
    cv2.putText(img,"BALL OUT",(BTN_X+28,BTN_Y+29),FONT,0.75,C_WHITE,2,cv2.LINE_AA)
    return img

def coord_to_zone(x,y):                                                                               # converts mouse coordinates into a zone number
    if not(MARGIN_X<=x<=MARGIN_X+GRID_W and MARGIN_Y<=y<=MARGIN_Y+GRID_H): return None
    return int(min((y-MARGIN_Y)//CELL_H,2)*3+min((x-MARGIN_X)//CELL_W,2)+1)

def draw_overlay(frame,fname,cur,total,t0,kf,status,spd=0):                                           # draws the on-screen overlay over the video
    h,w=frame.shape[:2]; ov=frame.copy()
    cv2.rectangle(ov,(0,0),(w,48),C_DARK,-1); cv2.addWeighted(ov,0.72,frame,0.28,0,frame)
    cv2.putText(frame,fname,(10,20),FONT,0.52,C_YELLOW,1,cv2.LINE_AA)
    cv2.putText(frame,f"Frame: {cur:05d}/{total-1:05d}",(10,40),FONT,0.52,C_WHITE,1,cv2.LINE_AA)

    if t0 is not None:
        cv2.putText(frame,f"T0={t0}  Foot={'Left' if kf=='L' else 'Right'}",(w-210,20),FONT,0.52,C_GREEN,1,cv2.LINE_AA)

    cv2.putText(frame,f"SPD:{SPEED_LABELS[spd]}",(w-105,40),FONT,0.50,C_ORANGE if spd>0 else C_GRAY,1,cv2.LINE_AA)
    cv2.putText(frame,status,(w//2-160,40),FONT,0.46,C_GRAY,1,cv2.LINE_AA)

    bar_y=h-40; cv2.rectangle(frame,(0,bar_y),(w,h),C_DARK,-1)
    x1=SEEK_PAD_X; x2=w-SEEK_PAD_X; y1=h-SEEK_Y_OFF; y2=y1+SEEK_H

    cv2.rectangle(frame,(x1,y1),(x2,y2),(85,85,85),-1)
    fx=x1+int((x2-x1)*(cur/(total-1))) if total>1 else x1

    cv2.rectangle(frame,(x1,y1),(fx,y2),C_PINK,-1)
    cv2.circle(frame,(fx,(y1+y2)//2),6,C_WHITE,-1)
    cv2.putText(frame," A/D:-/+1  Q/W:-/+10  R/F:-/+50  X:speed(1x>2x>4x>6x>10x)  S:T0  M:repeat  ESC:quit",
                (8,h-8),FONT,0.38,C_GRAY,1,cv2.LINE_AA)
    return frame

def ask_overlay(cap, cur_init, title, options, total=None):   # shows a question over the video and waits for the answer
    cur = cur_init                                            # local copy, cur_init is not modified

    while True:
        cap.set(cv2.CAP_PROP_POS_FRAMES, cur)                # seek before reading, cap.read() advances the pointer automatically
        ret, frame = cap.read()
        if not ret: return None                              # invalid frame

        h, w = frame.shape[:2]
        ov = frame.copy()                                    # copy for the semi-transparency effect

        bh = 60 + len(options) * 36; by = h//2 - bh//2      # panel height depends on the number of options
        cv2.rectangle(ov, (w//2-200, by), (w//2+200, by+bh), C_DARK, -1)
        cv2.addWeighted(ov, 0.8, frame, 0.2, 0, frame)      # 80% dark panel + 20% original video

        cv2.putText(frame, title, (w//2-160, by+30), FONT, 0.65, C_YELLOW, 2, cv2.LINE_AA)
        for i, (ch, lbl, col) in enumerate(options):         # each option takes up 36px vertically
            cv2.putText(frame, f"[{ch.upper()}]  {lbl}", (w//2-160, by+60+i*36), FONT, 0.60, col, 1, cv2.LINE_AA)

        nav = f"Frame {cur}" + (f"/{total-1}" if total else "")
        cv2.putText(frame, f"A/D:-/+1  Q/W:-/+10  |  {nav}  |  ESC:cancel",
                    (8, h-8), FONT, 0.38, C_GRAY, 1, cv2.LINE_AA)
        cv2.imshow(MAIN_WIN, fit_image(frame)[0])

        k = cv2.waitKey(30) & 0xFF
        if k == 27: return None                              # ESC, cancels the whole event

        delta = 0                                            # if it doesn't answer the question, it just changes frame
        if   k == ord('a'): delta = -1
        elif k == ord('d'): delta =  1
        elif k == ord('q'): delta = -10
        elif k == ord('w'): delta =  10
        if delta and total:
            cur = max(0, min(total-1, cur + delta)); continue

        for ch, lbl, _ in options:                           # accepts upper and lower case
            if k == ord(ch) or k == ord(ch.upper()): return ch


# ==============================================================================================================
# STEP 1 — ANNOTATE
# ==============================================================================================================

def run_annotate(videos_dir, output_csv, session_path=None):                         # TODO: expose --session in argparse or remove it (never passed by any command)
    ensure_csv(output_csv)
    videos=list_videos(videos_dir)
    if not videos: print(f"[ERROR] No videos: {videos_dir}"); return
    print(f"\n[ANNOTATE] {len(videos)} video(s)  →  {output_csv}\n")

    start_vid=0; start_frame=0
    if session_path and Path(session_path).exists():                                   # checks whether a previous session file exists
        try:
            s=json.loads(Path(session_path).read_text())
            start_vid=s.get('video_index',0); start_frame=s.get('frame',0)             # reads the json and recovers the video and frame
            print(f"[SESSION] Resuming: video #{start_vid+1}, frame {start_frame}")
        except Exception: pass

    prev={k:None for k in ['event_id','kick_foot','goal_zone','gk_guessed','result']}  # stores the last event's data for M
    cv2.namedWindow(MAIN_WIN,cv2.WINDOW_NORMAL); cv2.resizeWindow(MAIN_WIN,1280,720)

    for vi,vpath in enumerate(videos):
        if vi<start_vid: continue
        fname=vpath.name; print(f"\n[{vi+1}/{len(videos)}]  {fname}")
        cap=cv2.VideoCapture(str(vpath)); total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cur=start_frame if vi==start_vid else 0
        cap.set(cv2.CAP_PROP_POS_FRAMES,cur)
        t0=kf=zone=gk=res=None; spd=0
        wait_ms,frame_skip=SPEEDS[spd]; paused=True
        status="Navigate  |  S=mark T0  |  ESC=quit"
        seek_st={'target':None}                                                      # stores the target frame of the seekbar click
        def on_seek(event,mx,my,flags,_):                                            # click on the seekbar jumps to the corresponding frame
            if event==cv2.EVENT_LBUTTONDOWN:
                x1=SEEK_PAD_X; x2=1280-SEEK_PAD_X; y1=720-SEEK_Y_OFF; y2=y1+SEEK_H
                if x1<=mx<=x2 and y1-6<=my<=y2+6: seek_st['target']=int((mx-x1)/(x2-x1)*(total-1))   # converts x position into frame number
        cv2.setMouseCallback(MAIN_WIN,on_seek)

        while True:
            if seek_st['target'] is not None:
                cur=seek_st['target']; cap.set(cv2.CAP_PROP_POS_FRAMES,cur)
                seek_st['target']=None; paused=True
            cap.set(cv2.CAP_PROP_POS_FRAMES,cur); ret,frame=cap.read()
            if not ret: break
            disp=draw_overlay(frame.copy(),fname,cur,total,t0,kf,status,spd)
            cv2.imshow(MAIN_WIN,fit_image(disp)[0])
            key=cv2.waitKey(wait_ms if not paused else 30)&0xFF

            # ESC — checked first, before any other key
            if key==27:
                print("[ESC] Exiting...")
                cap.release()
                cv2.destroyAllWindows()
                return

            delta=0
            if   key==ord('a'): delta=-1;  paused=True                               # each key advances delta frames
            elif key==ord('d'): delta=1;   paused=True
            elif key==ord('q'): delta=-10; paused=True
            elif key==ord('w'): delta=10;  paused=True
            elif key==ord('r'): delta=-50; paused=True
            elif key==ord('f'): delta=50;  paused=True
            elif key==ord(' '): paused=not paused
            elif key==ord('x') or key==ord('X'):
                spd=(spd+1)%len(SPEEDS)
                wait_ms,frame_skip=SPEEDS[spd]
            if delta: cur=max(0,min(total-1,cur+delta)); continue
            if not paused: cur=min(total-1,cur+frame_skip); continue

            if key==ord('s'):                                                       # mark T0
                t0=cur; print(f"\n[T0] {t0}")
                fc=ask_overlay(cap,cur,"Kicking foot?",[('1',"Right",C_GREEN),('0',"Left",C_RED)],total=total)
                if not fc: t0=None; status="Cancelled."; continue
                kf='R' if fc=='1' else 'L'                                          # to mark the foot used, 1 or 0 is chosen but converted to R or L

                for fi in range(1, 16):                                             # play 15 frames after 's' to see the ball trajectory
                    fn = min(t0 + fi, total - 1)
                    cap.set(cv2.CAP_PROP_POS_FRAMES, fn)
                    ret2, f2 = cap.read()
                    if not ret2: break                                              # if the video ends before the 15 frames
                    disp2 = draw_overlay(f2.copy(), fname, fn, total, t0, kf,
                                         f"Advancing... {fi}/15 frames", spd)
                    cv2.imshow(MAIN_WIN, fit_image(disp2)[0])
                    cv2.waitKey(40)                                                 # ~25fps
                cur = min(t0 + 15, total - 1)
                cap.set(cv2.CAP_PROP_POS_FRAMES, cur)

                zst={'hover':None,'zone':None}                                                                      # goal-zone window
                def on_goal(event,mx,my,flags,_):
                    if event==cv2.EVENT_MOUSEMOVE:
                        z=coord_to_zone(mx,my)
                        zst['hover']=z if z else(0 if BTN_X<=mx<=BTN_X+BTN_W and BTN_Y<=my<=BTN_Y+BTN_H else None)  # 'Out' button
                    elif event==cv2.EVENT_LBUTTONDOWN:
                        z=coord_to_zone(mx,my)
                        if z: zst['zone']=z
                        elif BTN_X<=mx<=BTN_X+BTN_W and BTN_Y<=my<=BTN_Y+BTN_H: zst['zone']=0
                cv2.namedWindow(GOAL_WIN,cv2.WINDOW_NORMAL); cv2.resizeWindow(GOAL_WIN,GOAL_W,GOAL_H)
                cv2.setMouseCallback(GOAL_WIN,on_goal)
                while zst['zone'] is None:                                                                          # waits until the user clicks a zone
                    cap.set(cv2.CAP_PROP_POS_FRAMES, cur)
                    ret_g, frame_g = cap.read()
                    if ret_g:
                        disp_g = draw_overlay(frame_g.copy(), fname, cur, total, t0, kf,
                                              "Click the goal zone  |  A/D:-/+1  Q/W:-/+10", spd)
                        cv2.imshow(MAIN_WIN, fit_image(disp_g)[0])
                    cv2.imshow(GOAL_WIN, build_goal_image(zst['hover']))
                    k2 = cv2.waitKey(30) & 0xFF
                    if k2 == 27: break                                                                              # ESC cancels
                    elif k2 == ord('a'): cur = max(0, cur - 1)
                    elif k2 == ord('d'): cur = min(total - 1, cur + 1)
                    elif k2 == ord('q'): cur = max(0, cur - 10)
                    elif k2 == ord('w'): cur = min(total - 1, cur + 10)
                cv2.destroyWindow(GOAL_WIN)                                                                         # closes the window
                if zst['zone'] is None: t0=kf=None; status="Cancelled."; continue
                zone=zst['zone']                                                                                    # stores the chosen zone

                gc=ask_overlay(cap,cur,"GK guessed?",[('1',"Yes",C_GREEN),('0',"No",C_RED)],total=total)
                if not gc: t0=kf=None; status="Cancelled."; continue
                gk=int(gc)                                                                                          # converts from string to int
                rc=ask_overlay(cap,cur,"Result?",[('0',"Goal",C_GREEN),('1',"Saved",C_RED),('2',"Post",C_ORANGE),('4',"Out",C_BLUE)],total=total)  # 3 reserved for "Crossbar" (removed)
                if not rc: t0=kf=None; status="Cancelled."; continue
                res=int(rc)                                                                                         # converts from string to int

                eid=uuid4().hex                                                                                     # generates a unique id for this event
                row={"event_id":eid,"video_name":fname,"t0_frame":t0,
                     "kick_foot":"Right" if kf=='R' else "Left",
                     "goal_zone":"Out" if zone==0 else zone,
                     "gk_guessed":gk,"result":res}
                append_row(output_csv,row)                                                                          # appends the event to the csv
                prev.update({'event_id':eid,'kick_foot':kf,'goal_zone':zone,'gk_guessed':gk,'result':res})          # stores for M
                status=f"Saved! Zone={'Out' if zone==0 else zone}  GK={gk}  Result={res}"
                print(f"[OK] {eid}"); t0=kf=None
                cv2.setMouseCallback(MAIN_WIN,on_seek)

            elif key==ord('m'):                                                                                     # # M — Repeat (different camera / same penalty)
                if not prev['event_id']: status="No previous event."; continue
                t0=cur
                row={"event_id":prev['event_id'],"video_name":fname,"t0_frame":t0,
                     "kick_foot":"Right" if prev['kick_foot']=='R' else "Left",
                     "goal_zone":"Out" if prev['goal_zone']==0 else prev['goal_zone'],
                     "gk_guessed":prev['gk_guessed'],"result":prev['result']}                                       # all metadata stays the same, only video and frame change
                append_row(output_csv,row)
                status="Saved (M)  event_id reused"
                print(f"[M] {prev['event_id']}"); t0=None

        cap.release(); start_frame=0
    cv2.destroyAllWindows()                                                                                         # closes the video and resets the start frame for the next one
    print(f"\n[ANNOTATE] Done → {output_csv}")                                                                 # advances to the next video in the session

# ==============================================================================================================
# STEP 2 — EXTRACT
# ==============================================================================================================

def run_extract(csv_path, videos_dir, clips_dir, n_frames=N_FRAMES):
    import pandas as pd  # type: ignore
    if not Path(csv_path).exists(): print(f"[ERROR] CSV not found: {csv_path}"); return
    df=pd.read_csv(csv_path)
    print(f"\n[EXTRACT] {len(df)} record(s)  |  {n_frames} frames per clip")
    print(f"  Videos  : {videos_dir}")
    print(f"  Output  : {clips_dir}\n")
    Path(clips_dir).mkdir(parents=True,exist_ok=True)
    ok=skip=error=0

    for _,row in df.iterrows():                               # iterates each csv row
        vname=str(row['video_name']); eid=str(row['event_id']); t0=int(row['t0_frame'])
        vpath=Path(videos_dir)/vname
        if not vpath.exists():                 # tries to find the video
            matches=[m for m in Path(videos_dir).glob(f"{Path(vname).stem}.*") if m.suffix in VIDEO_EXT]
            if not matches: print(f"  ⚠  Not found: {vname}"); error+=1; continue
            vpath=matches[0]

        out_path=Path(clips_dir)/clip_name(vname,eid,t0)        # creates the output folder if missing
        if out_path.exists(): print(f"  ↷  {out_path.name}"); skip+=1; continue      # counters for the final summary

        cap=cv2.VideoCapture(str(vpath))
        total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps=cap.get(cv2.CAP_PROP_FPS) or 25.0               # uses 25fps if the video has no fps defined
        fw=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); fh=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        start=max(0,t0-(n_frames-1))                # video's first frame, never negative
        cap.set(cv2.CAP_PROP_POS_FRAMES,start)
        frames=[]
        for _ in range(t0-start+1):           # reads between start and T0
            ret,f=cap.read()
            if not ret: break            # stops if the video ends before T0
            frames.append(f)
        cap.release()

        if not frames: print(f"  ✗  No frames: {out_path.name}"); error+=1; continue

        writer=cv2.VideoWriter(str(out_path),cv2.VideoWriter_fourcc(*'mp4v'),fps,(fw,fh))     # creates the output file
        for f in frames: writer.write(f)                                 # writes each frame to the clip
        writer.release()                   # closes and saves the file
        ok+=1; print(f"  ✓  {out_path.name}  ({len(frames)} frames)")

    print(f"\n[EXTRACT] ✓{ok}  ↷{skip}  ✗{error}  total={len(df)}")

# ==============================================================================================================
# STEP 3 — CROP
# ==============================================================================================================

def draw_bar(img,text,color=C_WHITE):               # info bar at the top of the window
    ov=img.copy(); cv2.rectangle(ov,(0,0),(img.shape[1],36),C_DARK,-1)
    cv2.addWeighted(ov,0.65,img,0.35,0,img)
    cv2.putText(img,text,(8,25),FONT,0.58,color,1,cv2.LINE_AA)

def draw_persons(img,persons,hover_id=None,sel_id=None):        # draws the bounding boxes
    for p in persons:
        x1,y1,x2,y2=[int(v) for v in p['bbox']]; tid=p['id']
        if   tid==sel_id:   col,th,lbl=C_YELLOW,3,f"#{tid} KICKER"
        elif tid==hover_id: col,th,lbl=C_GREEN,2,f"#{tid}"
        else:               col,th,lbl=C_WHITE,1,f"#{tid}"
        cv2.rectangle(img,(x1,y1),(x2,y2),col,th)
        cv2.putText(img,lbl,(x1,max(y1-6,14)),FONT,0.58,col,1,cv2.LINE_AA)

class YOLOCropper:
    def __init__(self,model_path='yolov8n.pt',conf=0.15,imgsz=640,              # model chosen: Yolov8n (6mb, enough for this task) with 0.15 confidence since selection is manual
                 pad_x=60,pad_y=80,out_w=512,out_h=1024,max_gap=3,device=None):
        from ultralytics import YOLO # type: ignore
        import torch # type: ignore
        if device is None: device='0' if torch.cuda.is_available() else 'cpu'               # auto-detects GPU, else uses CPU
        self.device=device; self.use_fp16=(device!='cpu') and torch.cuda.is_available()         # FP16 only on GPU, the precision loss is negligible but the speed gain is significant
        self.conf=conf; self.imgsz=imgsz; self.pad_x=pad_x; self.pad_y=pad_y
        self.out_w=out_w; self.out_h=out_h; self.max_gap=max_gap
        print(f"  GPU FP16: {'Yes' if self.use_fp16 else 'No'}")
        print(f"  Loading {model_path} ...", end=' ', flush=True)
        self.model=YOLO(model_path)
        self._fp_kw={"quantize":"fp16"} if self.use_fp16 else {}
        dummy=np.zeros((64,64,3),dtype=np.uint8)                             # warm-up with a small image
        self.model(dummy,device=self.device,verbose=False,**self._fp_kw)
        print("Ready.\n")

    def _reset(self):                       # resets ByteTrack between clips
        try:
            if hasattr(self.model,'predictor') and self.model.predictor:
                for t in self.model.predictor.trackers: t.reset()
        except Exception: pass

    def _detect(self,frame):                      # detects people
        res=self.model.track(frame,persist=True,tracker='bytetrack.yaml',classes=[0],              # persist=True keeps the IDs across frames of the same clip, class[0] = person
                             conf=self.conf,imgsz=self.imgsz,device=self.device,
                             verbose=False,**self._fp_kw)
        ps=[]
        if res[0].boxes is not None:
            for box in res[0].boxes:
                if box.id is None: continue
                ps.append({'id':int(box.id[0]),
                           'bbox':[float(v) for v in box.xyxy[0]],              # x1,y1,x2,y2 coordinates
                           'conf':float(box.conf[0])})
        return ps                                    # returns list of {id, bbox, conf}

    def _select(self,frame,persons,win,msg):
        if not persons: return None
        st={'hover':None,'clicked':None}; scale=[1.0]                   # mouse state and window scale
        def on_mouse(event,mx,my,flags,_):
            ox,oy=int(mx/scale[0]),int(my/scale[0]); st['hover']=None             # converts window coordinates into real frame coordinates
            for p in persons:
                x1,y1,x2,y2=[int(v) for v in p['bbox']]
                if x1<=ox<=x2 and y1<=oy<=y2:                         # checks whether the mouse is inside the bounding box
                    st['hover']=p['id']
                    if event==cv2.EVENT_LBUTTONDOWN: st['clicked']=p['id']           # records the id
                    break                                     # only one person can be under the cursor
        cv2.setMouseCallback(win,on_mouse)
        while True:
            if st['clicked'] is not None: return st['clicked']
            disp=frame.copy(); draw_persons(disp,persons,hover_id=st['hover'])
            draw_bar(disp,msg,C_GREEN); d,s=fit_image(disp); scale[0]=s; cv2.imshow(win,d)      # updates the scale
            k=cv2.waitKey(15)&0xFF
            if k in(ord('q'),27): return 'quit'
            if k==ord('s'):       return 'skip'

    @staticmethod
    def _lb(img,tw,th):                                # letterbox, resizes keeping proportions and padding with black
        h,w=img.shape[:2]; s=min(tw/w,th/h); nw,nh=int(w*s),int(h*s)
        res=cv2.resize(img,(nw,nh),interpolation=cv2.INTER_LANCZOS4)
        out=np.zeros((th,tw,3),dtype=np.uint8)
        out[(th-nh)//2:(th-nh)//2+nh,(tw-nw)//2:(tw-nw)//2+nw]=res           # centers the image on the canvas
        return out

    def _smooth(self,crops,fw,fh):                    # smooths the bounding boxes to avoid jumps
        f=list(crops); last=None
        for i,c in enumerate(f):
            if c: last=c
            elif last: f[i]=last
        last=None
        for i in range(len(f)-1,-1,-1):
            if f[i]: last=f[i]
            elif last: f[i]=last
        ema,r=None,[]                         # EMA = Exponential Moving Average to smooth the box position
        for c in f:
            if not c: r.append(None); continue
            x1,y1,x2,y2=[int(v) for v in c]
            arr=np.array([max(0,x1-self.pad_x),max(0,y1-self.pad_y),
                          min(fw,x2+self.pad_x),min(fh,y2+self.pad_y)],float)
            ema=arr if ema is None else 0.3*arr+0.7*ema; r.append(ema.astype(int).tolist())        # 30% current position and 70% previous position
        return r

    def process(self,inp,out,win):                  # processes the clip
        cap=cv2.VideoCapture(inp); fps=cap.get(cv2.CAP_PROP_FPS) or 25.0
        frames=[]
        while True:                         # reads all frames into memory
            ret,f=cap.read()
            if not ret: break
            frames.append(f)
        cap.release()
        if not frames: return 'skip'
        fh,fw=frames[0].shape[:2]; self._reset()        # resets ByteTrack
        p1=self._detect(frames[0])               # detects on the 1st frame
        if not p1:
            d=frames[0].copy(); draw_bar(d,"No person found  |  S:skip  |  Q:quit",(60,60,220))
            cv2.imshow(win,fit_image(d)[0]); k=cv2.waitKey(0)&0xFF
            return 'quit' if k in(ord('q'),27) else 'skip'                          # respects Q/ESC as the message indicates
        tid=self._select(frames[0],p1,win,"CLICK the kicker  |  S:skip  |  Q:quit")                  # choose the kicker
        if tid in('skip','quit'): return tid
        crops=[next((p['bbox'] for p in p1 if p['id']==tid),None)]; gap=0
        for idx in range(1,len(frames)):
            ps=self._detect(frames[idx]); found=next((p['bbox'] for p in ps if p['id']==tid),None)
            if found: crops.append(found); gap=0                   # ByteTrack found the kicker
            else:
                gap+=1; crops.append(None)
                if gap>self.max_gap and ps:           # if it lost track, asks for a new selection
                    nid=self._select(frames[idx],ps,win,f"Lost track (frame {idx+1}) — click again")
                    if nid in('skip','quit'): return nid
                    if nid: tid=nid; gap=0; crops[idx]=next((p['bbox'] for p in ps if p['id']==tid),None)
        sm=self._smooth(crops,fw,fh); valid=[c for c in sm if c]
        if not valid: return 'skip'
        os.makedirs(os.path.dirname(os.path.abspath(out)),exist_ok=True)
        wr=cv2.VideoWriter(out,cv2.VideoWriter_fourcc(*'mp4v'),fps,(self.out_w,self.out_h))
        fb=valid[0]                    # fallback for frames without detection
        for frame,crop in zip(frames,sm):
            x1,y1,x2,y2=crop if crop else fb; piece=frame[y1:y2,x1:x2]
            piece=self._lb(piece,self.out_w,self.out_h) if piece.size>0 else np.zeros((self.out_h,self.out_w,3),dtype=np.uint8)    # letterbox
            wr.write(piece)
        wr.release(); return 'saved'


def run_crop(clips_dir,output_dir,model='yolov8n.pt',conf=0.15,imgsz=1280,          # kicker crop
             pad_x=60,pad_y=80,out_w=512,out_h=1024,max_gap=3,device=None):
    clips=sorted([p for p in Path(clips_dir).iterdir() if p.suffix in VIDEO_EXT])
    if not clips: print(f"[ERROR] No clips: {clips_dir}"); return
    print(f"\n[CROP] {len(clips)} clip(s)  →  {output_dir}")
    Path(output_dir).mkdir(parents=True,exist_ok=True)
    cropper=YOLOCropper(model_path=model,conf=conf,imgsz=imgsz,                        # initializes YOLO
                         pad_x=pad_x,pad_y=pad_y,out_w=out_w,out_h=out_h,
                         max_gap=max_gap,device=device)
    win="Penalty Taker Extractor"
    cv2.namedWindow(win,cv2.WINDOW_NORMAL); cv2.resizeWindow(win,1280,720)
    ok=skip=0
    for i,clip in enumerate(clips,1):
        out_path=Path(output_dir)/f"crop_{clip.name}"                        # final name = crop_ + step 2's clip name
        if out_path.exists(): print(f"[{i}/{len(clips)}] ↷ {out_path.name}"); skip+=1; continue
        cv2.setWindowTitle(win,f"[{i}/{len(clips)}]  {clip.name}")      # updates the window title with progress
        print(f"[{i}/{len(clips)}]  {clip.name}")
        r=cropper.process(str(clip),str(out_path),win)
        if   r=='quit': print("[Q] Interrupted."); break
        elif r=='saved': ok+=1; print("  ✓")
        else:            skip+=1; print("  ↷")
    cv2.destroyAllWindows()
    print(f"\n[CROP] ✓{ok}  ↷{skip}  total={len(clips)}")

# ==============================================================================================================
# MAIN
# ==============================================================================================================

def main():
    ap=argparse.ArgumentParser(description="01_extraction pipeline — 3 steps",
                               formatter_class=argparse.RawDescriptionHelpFormatter,epilog=__doc__)         # shows the docstring in --help
    sub=ap.add_subparsers(dest='cmd',required=True)          # subcommands

    # -- Step 1 ------------------------------------------------------
    p1=sub.add_parser('annotate',help='Step 1: manual annotation')
    p1.add_argument('--videos','-v',required=True)
    p1.add_argument('--csv','-c',default='dataset_manual.csv')

    # -- Step 2 ------------------------------------------------------
    p2=sub.add_parser('extract',help='Step 2: clip extraction')
    p2.add_argument('--csv','-c',required=True)
    p2.add_argument('--videos','-v',required=True)
    p2.add_argument('--clips','-o',required=True)
    p2.add_argument('--frames','-n',type=int,default=N_FRAMES)

    # -- Step 3 ------------------------------------------------------
    p3=sub.add_parser('crop',help='Step 3: YOLO crop')
    p3.add_argument('--clips','-i',required=True)
    p3.add_argument('--output','-o',required=True)
    p3.add_argument('--model',default='yolov8n.pt')
    p3.add_argument('--device',default=None)                                # None = auto-detects GPU
    p3.add_argument('--conf',type=float,default=0.15)
    p3.add_argument('--imgsz',type=int,default=640)
    p3.add_argument('--pad-x',type=int,default=60)
    p3.add_argument('--pad-y',type=int,default=80)
    p3.add_argument('--out-w',type=int,default=512)
    p3.add_argument('--out-h',type=int,default=1024)
    p3.add_argument('--max-gap',type=int,default=3)

    # -- Pipeline ----------------------------------------------------
    pa=sub.add_parser('all',help='Full pipeline (1+2+3)')
    pa.add_argument('--videos','-v',required=True)
    pa.add_argument('--output','-o',required=True)
    pa.add_argument('--csv',default='dataset_manual.csv')
    pa.add_argument('--frames',type=int,default=N_FRAMES)
    pa.add_argument('--model',default='yolov8n.pt')
    pa.add_argument('--device',default=None)

    args=ap.parse_args()

    if args.cmd=='annotate':
        run_annotate(args.videos,args.csv)
    elif args.cmd=='extract':
        run_extract(args.csv,args.videos,args.clips,args.frames)
    elif args.cmd=='crop':
        run_crop(args.clips,args.output,model=args.model,device=args.device,
                 conf=args.conf,imgsz=args.imgsz,pad_x=args.pad_x,pad_y=args.pad_y,
                 out_w=args.out_w,out_h=args.out_h,max_gap=args.max_gap)
    elif args.cmd=='all':                                                                # runs the 3 steps in sequence
        out=Path(args.output)
        csv_p=str(out/args.csv) if not Path(args.csv).is_absolute() else args.csv     # csv inside the output folder if not an absolute path
        clips_raw=str(out/'clips_raw'); clips_crop=str(out/'clips_cropped')           # folder with clips before cropping
        print("\n"+"="*55+"\nFULL PIPELINE — 3 STEPS\n"+"="*55)                  # folder with clips with the kicker isolated
        print("\n-- STEP 1: ANNOTATION --"); run_annotate(args.videos,csv_p)
        print("\n-- STEP 2: EXTRACTION --"); run_extract(csv_p,args.videos,clips_raw,args.frames)
        print("\n-- STEP 3: YOLO CROP --"); run_crop(clips_raw,clips_crop,model=args.model,device=args.device,imgsz=640)   # imgsz same as the 'crop' command default
        print("\n"+"="*55+f"\nDONE\n  CSV    : {csv_p}\n  Clips  : {clips_raw}\n  Cropped: {clips_crop}\n"+"="*55)

if __name__=='__main__':
    main()
