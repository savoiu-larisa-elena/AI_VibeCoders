"""AI VibeCoders live telemetry UI. Run: python dashboard.py --detectors drowsiness,head_pose,phone,gestures"""
from __future__ import annotations
import argparse, json, sys, threading, time, uuid, csv, io
from collections import deque
from pathlib import Path
from datetime import datetime
import cv2
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse, FileResponse, Response
import uvicorn

DASHBOARD_DIR = Path(__file__).resolve().parent
ROOT = DASHBOARD_DIR.parent  # main AI_VibeCoders project directory
sys.path.insert(0, str(ROOT / 'src'))
from dms.config import Config,LOGS_DIR
from dms.pipeline import Pipeline

app=FastAPI(title='VibeCoders Driver Telemetry')
lock=threading.Lock()
state={'camera':'starting','calibrating':True,'fps':0,'signals':{},'statuses':{},'events':[],
       'latest_warning':'Waiting for events','frame':None,'session':'running',
       'started_at':time.time(),'ended_at':None,'history':[], 'session_events':[], 'session_id':uuid.uuid4().hex[:12]}
# Archived dashboard sessions are independent of the continuous JSONL detector logs.
ARCHIVE_DIR=ROOT/'logs'/'dashboard_sessions'
ARCHIVE_DIR.mkdir(parents=True,exist_ok=True)

def session_snapshot():
    return {k:state[k] for k in ('session_id','started_at','ended_at','history','session_events')}

def archive_session(snapshot):
    # Atomic write: archived sessions survive application restarts.
    target=ARCHIVE_DIR/f"{snapshot['session_id']}.json"
    temp=target.with_suffix('.tmp')
    temp.write_text(json.dumps(snapshot,ensure_ascii=False),encoding='utf-8')
    temp.replace(target)

def archived_sessions():
    result=[]
    for file in ARCHIVE_DIR.glob('*.json'):
        try:
            data=json.loads(file.read_text(encoding='utf-8'))
            result.append({'session_id':data['session_id'],'started_at':data['started_at'],
                           'ended_at':data.get('ended_at'),'samples':len(data.get('history',[])),
                           'events':len(data.get('session_events',[]))})
        except (ValueError, KeyError, OSError):
            continue
    return sorted(result,key=lambda d:d['started_at'],reverse=True)

stop=threading.Event()
recalibrate=threading.Event()
# The CV pipeline and JSONL logging run continuously, independent of dashboard sessions.
# Dashboard sessions only control which measurements are collected for the UI.
SAMPLE_INTERVAL=0.5
MAX_SAMPLES=14400  # up to two hours at 2 Hz, per UI session
FALLBACK={'eyes_closed':'Keep your eyes on the road. Stop somewhere safe if you feel sleepy.',
          'distraction':'Please return your attention to the road.',
          'phone':'Put your phone away and focus on driving.',
          'yawn':'You may be getting tired. Consider taking a break.'}

HTML_FILE=DASHBOARD_DIR/'dashboard_ui.html'

@app.get('/')
def home():
    return FileResponse(HTML_FILE,media_type="text/html")

@app.get('/state')
def get_state():
    with lock:
        return {k:v for k,v in state.items() if k!='frame'}

@app.get('/sessions')
def sessions():
    return {'sessions':archived_sessions()}

@app.get('/sessions/{session_id}')
def session_detail(session_id: str):
    from fastapi import HTTPException
    if not re_fullmatch_session(session_id):
        raise HTTPException(status_code=400,detail='Invalid session ID')
    path=ARCHIVE_DIR/f'{session_id}.json'
    if not path.is_file():
        raise HTTPException(status_code=404,detail='Session not found')
    return json.loads(path.read_text(encoding='utf-8'))

def re_fullmatch_session(value):
    return len(value)==12 and all(ch in '0123456789abcdef' for ch in value)

@app.get('/sessions/{session_id}/csv')
def export_session_csv(session_id: str):
    from fastapi import HTTPException
    if not re_fullmatch_session(session_id):
        raise HTTPException(status_code=400,detail='Invalid session ID')
    path=ARCHIVE_DIR/f'{session_id}.json'
    if not path.is_file():
        raise HTTPException(status_code=404,detail='Session not found')
    data=json.loads(path.read_text(encoding='utf-8'))
    out=io.StringIO()
    writer=csv.DictWriter(out,fieldnames=['t','perclos','ear','yaw','away'])
    writer.writeheader()
    writer.writerows(data.get('history',[]))
    return Response(out.getvalue(),media_type='text/csv',headers={
        'Content-Disposition':f'attachment; filename="telemetry_{session_id}.csv"'})

@app.post('/end-session')
def end_session():
    # Only freeze and archive UI telemetry; the camera and original JSONL logs continue.
    with lock:
        if state['session']=='running':
            state['session']='ended'
            state['ended_at']=time.time()
            snapshot=session_snapshot()
        else:
            snapshot=None
        result={'ok':True,'session':state['session'],'session_id':state['session_id']}
    if snapshot is not None:
        archive_session(snapshot)
    return result

@app.post('/start-session')
def start_session():
    with lock:
        if state['session']=='ended':
            state.update({'session':'running','started_at':time.time(),'ended_at':None,
                          'history':[],'session_events':[], 'session_id':uuid.uuid4().hex[:12]})
        return {'ok':True,'session':state['session'],'session_id':state['session_id']}

@app.post('/recalibrate')
def trigger_recalibrate():
    recalibrate.set()
    return {'ok':True}

@app.get('/video')
def video():
    def frames():
        while not stop.is_set():
            with lock:
                frame=state['frame']
            if frame:
                yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n'+frame+b'\r\n'
            time.sleep(.06)
    return StreamingResponse(frames(),media_type='multipart/x-mixed-replace; boundary=frame')

def number(v):
    if isinstance(v,(int,float)) and not isinstance(v,bool):
        result=float(v)
        return result if result==result and abs(result)!=float('inf') else None
    return None

def camera_worker(args):
    config=Config()
    if args.camera is not None:
        config.camera.index=args.camera
    try:
        pipeline=Pipeline([x.strip() for x in args.detectors.split(',') if x.strip()],config)
    except Exception as exc:
        with lock:
            state['camera']='pipeline error'
            state['latest_warning']=str(exc)
        return
    backend=cv2.CAP_DSHOW if sys.platform=='win32' else cv2.CAP_ANY
    cap=cv2.VideoCapture(config.camera.index,backend)
    if not cap.isOpened():
        with lock:
            state['camera']='unavailable'
            state['latest_warning']='Cannot open camera. Try --camera 1.'
        pipeline.close()
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,config.camera.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,config.camera.height)
    LOGS_DIR.mkdir(parents=True,exist_ok=True)
    path=LOGS_DIR/f'session_{datetime.now():%Y%m%d_%H%M%S_%f}.jsonl'
    last=time.monotonic()
    last_sample=0.0
    try:
        # One continuous source log per server run, irrespective of UI session state.
        with path.open('a',encoding='utf8') as log:
            while not stop.is_set():
                ok,frame=cap.read()
                if not ok:
                    with lock:
                        state['camera']='unavailable'
                    time.sleep(0.2)
                    continue
                if config.camera.mirror:
                    frame=cv2.flip(frame,1)
                if recalibrate.is_set():
                    pipeline.recalibrate()
                    recalibrate.clear()
                signals,events=pipeline.process(frame)
                pipeline.draw(frame)
                success,encoded=cv2.imencode('.jpg',frame,[cv2.IMWRITE_JPEG_QUALITY,75])
                now=time.monotonic()
                fps=1/max(now-last,1e-5)
                last=now
                for event in events:
                    log.write(event.to_json()+'\n')
                if events:
                    log.flush()
                with lock:
                    state['camera']='active'
                    state['calibrating']=bool(signals.get('calibrating'))
                    state['fps']=round(fps,1)
                    state['signals']=signals
                    state['statuses']={k:v.status for k,v in pipeline.outputs.items()}
                    if success:
                        state['frame']=encoded.tobytes()
                    for event in events:
                        item=json.loads(event.to_json())
                        state['events'].append(item)
                        state['latest_warning']=FALLBACK.get(event.type,
                            f'Driver safety event detected: {event.type.replace("_"," ")}.')
                        if state['session']=='running':
                            state['session_events'].append(item)
                    state['events']=state['events'][-100:]
                    state['session_events']=state['session_events'][-100:]
                    if state['session']=='running' and now-last_sample>=SAMPLE_INTERVAL:
                        d=signals.get('drowsiness') or {}
                        hp=signals.get('head_pose') or {}
                        perclos=number(d.get('perclos'))
                        state['history'].append({
                            't':round(time.time()-state['started_at'],1),
                            'perclos':perclos*100 if perclos is not None else None,
                            'ear':number(d.get('ear')),
                            'yaw':number(hp.get('yaw')),
                            'away':number(hp.get('away_seconds'))})
                        if len(state['history'])>MAX_SAMPLES:
                            state['history']=state['history'][-MAX_SAMPLES:]
                        last_sample=now
    except Exception as exc:
        with lock:
            state['camera']='pipeline error'
            state['latest_warning']=f'Camera pipeline error: {exc}'
        raise
    finally:
        cap.release()
        pipeline.close()

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--detectors',default='drowsiness,head_pose')
    parser.add_argument('--camera',type=int,default=None)
    parser.add_argument('--port',type=int,default=8000)
    args=parser.parse_args()
    threading.Thread(target=camera_worker,args=(args,),daemon=True).start()
    try:
        uvicorn.run(app,host='127.0.0.1',port=args.port,log_level='info')
    finally:
        stop.set()
