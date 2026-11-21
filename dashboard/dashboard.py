"""AI VibeCoders live telemetry UI.

Run from the project root:
    python dashboard/dashboard.py --detectors drowsiness,head_pose,phone,gestures          # no LLM (free)
    python dashboard/dashboard.py --detectors drowsiness,head_pose,phone,gestures --llm    # with LLM assistant
then open http://127.0.0.1:8000
(This app uses the camera itself - don't run run.py at the same time.)
"""
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
from dms.agent import Agent, DriverContext, LLMClient
from dms.agent.agent import FALLBACKS
from dms.perf import FpsCounter, boost_process

app=FastAPI(title='VibeCoders Driver Telemetry')
lock=threading.Lock()
log_lock=threading.Lock()   # camera thread and LLM thread both write the JSONL log
state={'camera':'starting','calibrating':True,'fps':0,'signals':{},'statuses':{},'events':[],
       'latest_warning':'Waiting for events','frame':None,'session':'running',
       'started_at':time.time(),'ended_at':None,'history':[], 'session_events':[], 'session_id':uuid.uuid4().hex[:12],
       # LLM assistant
       'attention':100,'llm':{'enabled':False,'provider':None,'model':None},
       'assistant':None,'assistant_log':[],'muted_seconds':0,
       # occupants + performance diagnostics
       'driver_present':False,'passengers':0,'timing':{},'frame_id':0,
       # AI voice (spoken on the laptop by dms.voice.Speaker); toggled from the dashboard
       'voice':{'available':False,'enabled':True}}
STREAM_WIDTH=960        # browser video is downscaled: cheaper to encode, send and decode
STREAM_QUALITY=70
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
def get_state(since: float = -1.0, session_id: str = ''):
    """Returns live state. History is sent incrementally: the page passes the last sample time
    it already has (`since`) and only newer samples come back, so the payload stays small
    even in long sessions. A different session_id means the page must reset its history."""
    with lock:
        out={k:v for k,v in state.items() if k not in ('frame','history')}
        if session_id!=state['session_id']:
            since=-1.0                      # new session (or first load): send everything
        out['history']=[d for d in state['history'] if d['t']>since]
        out['history_reset']=since<0
        out['history_total']=len(state['history'])
        return out

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
    writer=csv.DictWriter(out,fieldnames=['t','perclos','ear','yaw','away','attention'],
                          extrasaction='ignore')
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

AGENT={'agent':None}   # set by camera_worker when the LLM assistant runs

@app.post('/unmute')
def unmute():
    """End the open-palm warning mute early (same as a thumbs up while muted)."""
    agent=AGENT['agent']
    done=bool(agent and agent.unmute())
    if done:
        with lock:
            state['muted_seconds']=0
    return {'ok':True,'unmuted':done}

@app.post('/voice/{mode}')
def set_voice(mode: str):
    """Dashboard VOICE button: turn the spoken AI messages on/off."""
    with lock:
        state['voice']['enabled']=(mode=='on')
        return {'ok':True,'voice':state['voice']}

@app.get('/video')
def video():
    def frames():
        sent=-1
        while not stop.is_set():
            with lock:
                frame,fid=state['frame'],state['frame_id']
            if frame and fid!=sent:          # only send NEW frames (no duplicate JPEGs)
                sent=fid
                yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n'+frame+b'\r\n'
            time.sleep(.03)
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
    last_sample=0.0
    fps_counter=FpsCounter()
    log=None
    timing={}   # smoothed ms per part of the frame, shown on the dashboard

    def smooth(key,ms):
        timing[key]=ms if key not in timing else 0.85*timing[key]+0.15*ms
    phone=next((d for d in pipeline.detectors if d.name=='phone'),None)

    def write_log(line):
        with log_lock:
            if log is not None and not log.closed:   # a late LLM reply may arrive after shutdown
                log.write(line+'\n'); log.flush()

    # ---- driver context (attention score) always; LLM assistant only with --llm ----
    context=DriverContext()
    agent=None
    if args.llm:
        PRIO={'info':1,'warning':2,'critical':3}
        speaker=None
        try:
            from dms.voice import Speaker, fixed_phrases
            speaker=Speaker()   # voice from TTS_VOICE in .env (default: troy)
            # pre-generate audio for every sentence the app can say WITHOUT the LLM
            # (fallbacks + instant gesture replies) -> the voice still works offline
            speaker.prewarm(fixed_phrases(FALLBACKS))
            with lock:
                state['voice']['available']=True
        except Exception as exc:   # voice must never break the dashboard
            print(f'[VOICE] disabled: {exc}',flush=True)
        def on_reply(r):   # called from the agent's background thread
            write_log(r.to_json())
            item={'speak':r.speak,'priority':r.priority,'action':r.action,'note':r.dashboard_note,
                  'trigger':r.trigger,'level':r.level,'source':r.source,
                  'latency':round(r.latency,2),'error':r.error,'ts':r.ts}
            with lock:
                state['assistant']=item
                state['assistant_log']=(state['assistant_log']+[item])[-20:]
                state['latest_warning']=r.speak
                voice_on=state['voice']['enabled']
            if speaker and voice_on:
                speaker.say(r.speak,PRIO.get(r.priority,1))
        try:
            llm=LLMClient()
            agent=Agent(context,on_reply,llm)
            AGENT['agent']=agent
            with lock:
                state['llm']={'enabled':True,'provider':llm.provider,'model':llm.model}
        except Exception as exc:   # missing key etc. -> keep running without LLM
            with lock:
                state['llm']={'enabled':False,'provider':None,'model':None,'error':str(exc)}
    try:
        # One continuous source log per server run, irrespective of UI session state.
        with path.open('a',encoding='utf8') as log:
            while not stop.is_set():
                t0=time.perf_counter()
                ok,frame=cap.read()
                t1=time.perf_counter()
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
                t2=time.perf_counter()
                pipeline.draw(frame)
                h,w=frame.shape[:2]
                small=cv2.resize(frame,(STREAM_WIDTH,int(h*STREAM_WIDTH/w))) if w>STREAM_WIDTH else frame
                success,encoded=cv2.imencode('.jpg',small,[cv2.IMWRITE_JPEG_QUALITY,STREAM_QUALITY])
                t3=time.perf_counter()
                # where does the frame time go? camera wait / AI models / overlay+video
                smooth('camera',(t1-t0)*1000)
                smooth('ai',(t2-t1)*1000)
                smooth('video',(t3-t2)*1000)
                for k,v in pipeline.timing.items():
                    smooth(k,v)
                if phone:
                    timing['objects_bg']=phone.last_run_ms
                now=time.monotonic()
                fps=fps_counter.tick()   # real frames in the last second
                for event in events:
                    write_log(event.to_json())
                if not signals.get('calibrating'):
                    context.update(signals,events)
                    if agent:
                        for event in events:
                            agent.handle(event)   # returns immediately; LLM runs in background
                with lock:
                    state['camera']='active'
                    state['calibrating']=bool(signals.get('calibrating'))
                    state['fps']=round(fps,1)
                    state['attention']=round(context.attention)
                    state['muted_seconds']=max(0,round(agent.snoozed_until-time.time())) if agent else 0
                    state['signals']=signals
                    state['statuses']={k:v.status for k,v in pipeline.outputs.items()}
                    occ=signals.get('occupants') or {}
                    state['driver_present']=bool(occ.get('driver_present'))
                    state['passengers']=int(occ.get('passengers',0))
                    state['timing']={k:round(v,1) for k,v in timing.items()}
                    if success:
                        state['frame']=encoded.tobytes()
                        state['frame_id']+=1
                    for event in events:
                        item=json.loads(event.to_json())
                        state['events'].append(item)
                        if not agent and event.type in FALLBACK:   # with --llm the assistant writes the guidance
                            state['latest_warning']=FALLBACK[event.type]
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
                            'away':number(hp.get('away_seconds')),
                            'attention':round(context.attention,1)})
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
    parser.add_argument('--llm',action='store_true',help='enable the LLM assistant (key in .env; uses API quota)')
    args=parser.parse_args()
    print('Performance:',boost_process())   # stop Windows from moving us to slow cores
    worker=threading.Thread(target=camera_worker,args=(args,),daemon=True)
    worker.start()
    try:
        uvicorn.run(app,host='127.0.0.1',port=args.port,log_level='info')
    finally:
        stop.set()
        worker.join(timeout=3)   # let the camera loop finish cleanly (no error spam on Ctrl+C)