#!/usr/bin/env python3
"""Run directly: ./frontend/app.py (or python frontend/app.py)."""
import sys
from pathlib import Path

# Direct execution starts the web app; Streamlit reruns call main() below.
if __name__ == "__main__":
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        in_streamlit = get_script_run_ctx(suppress_warning=True) is not None
    except ImportError:
        in_streamlit = False
    if not in_streamlit:
        import subprocess
        folder = Path(__file__).resolve().parent
        python = folder / ".venv/bin/python"
        raise SystemExit(subprocess.call(
            [str(python) if python.exists() else sys.executable, "-m", "streamlit", "run",
             str(Path(__file__).resolve()), "--server.address=127.0.0.1", *sys.argv[1:]],
            cwd=folder,
        ))

import base64
import io
import json
import tempfile

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from rasterio.warp import transform

from engine import (CHECKPOINT, load_fire, save_uploads, read_points,
                    point_masks, run_forecast, rank_points, recommendation)

MAP_HTML = r'''
<!doctype html>
<html lang="en"><head><meta charset="utf-8"><style>
*{box-sizing:border-box}html,body{width:100%;height:100%;overflow-x:hidden}body{display:flex;flex-direction:column;margin:0;color:#eee;background:#171614;font:14px system-ui,sans-serif}
.row{display:flex;align-items:center;gap:12px;margin:8px 0;flex-wrap:wrap}
input[type=range]{width:100%;accent-color:#ec744a;margin:10px 0;cursor:pointer}
select,button,input:not([type=range]){font:inherit;color:#eee;background:#292723;border:1px solid #595651;border-radius:5px;padding:6px 10px}
button[aria-pressed=true]{border-color:#69f1ff;color:#69f1ff}
#pointForm{position:absolute;left:12px;top:12px;background:#242220;border:1px solid #888;border-radius:6px;padding:14px;width:min(290px,calc(100% - 24px));max-height:calc(100% - 24px);overflow-y:auto;z-index:2}
#pointForm label{display:block;margin:8px 0}#pointForm input{display:block;width:100%;margin-top:3px}#pointMessage{color:#ffd293}
button{cursor:pointer}button:focus-visible,select:focus-visible,input:focus-visible,canvas:focus-visible{outline:2px solid #72e4ef}
#date{margin-left:auto;font-variant-numeric:tabular-nums}#stage{position:relative;flex:1;min-height:160px}
body>label,.row,#day,#legend,.hint{flex-shrink:0}select{max-width:100%}
#map{display:block;width:100%;height:100%;touch-action:pan-y;cursor:grab;background:#171614}
#tip{position:absolute;pointer-events:none;background:#161616eF;border:1px solid #666;padding:7px;max-width:260px;display:none;white-space:pre-line}
#legend{width:100%;height:43px;display:block}.hint{font-size:12px;color:#c4bfb9;margin:5px 0}
</style></head><body>
<label for="day">Forecast day <output id="dayNumber"></output></label>
<input id="day" type="range" min="0" value="1" step="1" aria-label="Forecast day">
<div class="row"><label for="layer">Map layer</label><select id="layer">
<option value="burned">Cumulative burned</option><option value="active">Active fire</option><option value="new">Newly burned</option>
</select><button id="zoomIn" aria-label="Zoom in">+</button><button id="zoomOut" aria-label="Zoom out">−</button>
<button id="reset">Reset view</button><button id="addPoint" aria-pressed="false">Add point</button><span id="date" aria-live="polite">Loading map…</span></div>
<div id="stage"><canvas id="map" tabindex="0" aria-label="Wildfire forecast map. Drag to pan, Ctrl+scroll to zoom. Arrow keys pan; plus and minus zoom."></canvas><div id="tip"></div>
<form id="pointForm" hidden><strong>New point of interest</strong>
<label for="pointName">Location name<input id="pointName" required maxlength="100"></label>
<label for="pointValue">Protection value (1–10)<input id="pointValue" type="number" min="1" max="10" step="1" value="5" required></label>
<label for="pointRadius">Buffer radius (km)<input id="pointRadius" type="number" min="0.1" max="20" step="0.1" value="2" required></label>
<p id="pointMessage" role="status"></p><button type="submit">Save point</button> <button id="cancelPoint" type="button">Cancel</button></form></div>
<canvas id="legend" aria-label="Nonlinear square-root score legend"></canvas>
<p class="hint">Nonlinear scale  |  Model score 0–1  |  Ctrl+scroll to zoom</p>
<script>
const data=__PAYLOAD__;
const day=document.querySelector('#day'), layer=document.querySelector('#layer'), label=document.querySelector('#date');
const canvas=document.querySelector('#map'),ctx=canvas.getContext('2d'),tip=document.querySelector('#tip');
day.max=data.dates.length-1;day.value=Math.min(data.view?.day??1,Number(day.max));day.disabled=Number(day.max)===0;
if(data.view?.layer)layer.value=data.view.layer;
let zoom=1,ox=0,oy=0,cw=0,ch=0,background=null,current=null,requested=0,paintQueued=false,drag=null;
let adding=false,candidate=null;
const pointForm=document.querySelector('#pointForm'),addButton=document.querySelector('#addPoint');
const overlays=new Map();
// Interpolate in sqrt(score), so 0.25 gets half the color range, not a quarter.
const stops=[[0,[66,33,112]],[.25,[119,64,190]],[.5,[231,96,70]],[.75,[255,155,42]],[1,[255,35,40]]];
function color(p){const t=Math.sqrt(p);let j=1;while(j<stops.length-1&&t>stops[j][0])j++;
 const a=stops[j-1],b=stops[j],f=(t-a[0])/(b[0]-a[0]);return a[1].map((v,k)=>Math.round(v+(b[1][k]-v)*f));}
const lut=Array.from({length:256},(_,v)=>[...color(v/255),Math.round(240*Math.min(1,(v/255)/.06))]);
function loadImage(src){return new Promise((resolve,reject)=>{const image=new Image();image.onload=()=>resolve(image);image.onerror=reject;image.src=src;});}
async function overlay(key,index){const id=key+':'+index;if(overlays.has(id))return overlays.get(id);
 const job=(async()=>{const image=await loadImage(data.frames[key][index]);const buffer=document.createElement('canvas');
 buffer.width=data.width;buffer.height=data.height;const bc=buffer.getContext('2d',{willReadFrequently:true});bc.drawImage(image,0,0);
 const pixels=bc.getImageData(0,0,data.width,data.height),raw=new Uint8Array(data.width*data.height);
 for(let i=0;i<raw.length;i++){const v=pixels.data[i*4];raw[i]=v;pixels.data.set(lut[v],i*4);}bc.putImageData(pixels,0,0);
 return {buffer,raw,key,index};})();overlays.set(id,job);
 if(overlays.size>12)overlays.delete(overlays.keys().next().value);return job;}
function draw(){if(!background||!current)return;
 canvas.dataset.view=JSON.stringify({zoom,ox,oy});canvas.dataset.day=current.index;canvas.dataset.layer=current.key;
 ctx.save();ctx.clearRect(0,0,cw,ch);ctx.translate(ox,oy);ctx.scale(zoom,zoom);ctx.imageSmoothingEnabled=false;
 ctx.filter='grayscale(1) brightness(.55)';ctx.drawImage(background,0,0);ctx.filter='none';ctx.drawImage(current.buffer,0,0);ctx.restore();
 for(const p of [...data.points,...(candidate?[candidate]:[])]){ctx.beginPath();ctx.arc(ox+p.x*zoom,oy+p.y*zoom,5,0,2*Math.PI);ctx.fillStyle=p===candidate?'#fff':'#69f1ff';ctx.fill();ctx.strokeStyle='#101010';ctx.lineWidth=1.5;ctx.stroke();}}
function queueDraw(){if(paintQueued)return;paintQueued=true;requestAnimationFrame(()=>{paintQueued=false;draw();});}
async function selectFrame(){const version=++requested,index=Number(day.value),key=layer.value;
 document.querySelector('#dayNumber').textContent=index;
 try{const next=await overlay(key,index);if(version!==requested)return;current=next;
 label.textContent=data.dates[index]+'  |  '+(index?'Forecast +'+index+' days':'Observed');
 // The old image stays painted until the new overlay is fully decoded.
 queueDraw();for(const nearby of [index-1,index+1])if(nearby>=0&&nearby<data.dates.length)overlay(key,nearby).catch(()=>{});
 }catch(error){label.textContent='Unable to load this frame';}}
function fit(){const xs=data.points.map(p=>p.x),ys=data.points.map(p=>p.y);
 const left=xs.length?Math.max(0,Math.min(...xs)-40):0,right=xs.length?Math.min(data.width,Math.max(...xs)+40):data.width;
 const top=ys.length?Math.max(0,Math.min(...ys)-40):0,bottom=ys.length?Math.min(data.height,Math.max(...ys)+40):data.height;
 zoom=Math.min(cw/(right-left),ch/(bottom-top));ox=(cw-(right+left)*zoom)/2;oy=(ch-(bottom+top)*zoom)/2;queueDraw();}
function magnify(f,x=cw/2,y=ch/2){const next=Math.min(50,Math.max(.1,zoom*f));ox=x-(x-ox)*next/zoom;oy=y-(y-oy)*next/zoom;zoom=next;queueDraw();}
function legend(){const lc=document.querySelector('#legend');lc.width=Math.round(cw*devicePixelRatio);lc.height=43*devicePixelRatio;
 const c=lc.getContext('2d');c.scale(devicePixelRatio,devicePixelRatio);const pad=22,w=cw-2*pad;
 for(let x=0;x<w;x++){c.fillStyle='rgb('+color((x/w)**2).join(',')+')';c.fillRect(pad+x,0,1,12);}
 c.fillStyle='#ddd';c.font='12px system-ui';c.textAlign='center';for(const p of [0,.01,.05,.1,.25,.5,.75,1]){const x=pad+Math.sqrt(p)*w;c.fillRect(x,12,1,5);c.fillText(String(p),x,32);}}
new ResizeObserver(()=>{const rect=canvas.getBoundingClientRect(),oldW=cw,oldH=ch;cw=rect.width;ch=rect.height;
 canvas.width=Math.round(cw*devicePixelRatio);canvas.height=Math.round(ch*devicePixelRatio);ctx.setTransform(devicePixelRatio,0,0,devicePixelRatio,0,0);
 if(!oldW){if(data.view){zoom=data.view.zoom;ox=cw/2-data.view.centerX*zoom;oy=ch/2-data.view.centerY*zoom;queueDraw();}else fit();}
 else{ox+=(cw-oldW)/2;oy+=(ch-oldH)/2;queueDraw();}legend();}).observe(canvas);
day.addEventListener('input',selectFrame);layer.addEventListener('change',selectFrame);
document.querySelector('#reset').onclick=fit;document.querySelector('#zoomIn').onclick=()=>magnify(1.3);document.querySelector('#zoomOut').onclick=()=>magnify(1/1.3);
canvas.addEventListener('wheel',e=>{if(!e.ctrlKey)return;e.preventDefault();magnify(Math.exp(-e.deltaY*.001),e.offsetX,e.offsetY);},{passive:false});
function setAdding(value){adding=value;addButton.setAttribute('aria-pressed',value);addButton.textContent=value?'Click a location':'Add point';canvas.style.cursor=value?'crosshair':'grab';}
function cancelPoint(){candidate=null;pointForm.hidden=true;setAdding(false);queueDraw();}
addButton.onclick=()=>{if(adding)cancelPoint();else setAdding(true);};
document.querySelector('#cancelPoint').onclick=cancelPoint;
pointForm.onsubmit=e=>{e.preventDefault();if(!candidate)return;
 const name=document.querySelector('#pointName').value.trim();
 if(!name||data.points.some(p=>p.name===name)){document.querySelector('#pointMessage').textContent='Choose a unique location name.';return;}
 const event={...candidate,name,value:Number(document.querySelector('#pointValue').value),radius_km:Number(document.querySelector('#pointRadius').value),
  id:crypto.randomUUID(),fire_token:data.fire_token,view:{zoom,centerX:(cw/2-ox)/zoom,centerY:(ch/2-oy)/zoom,day:Number(day.value),layer:layer.value}};
 window.parent.postMessage({type:'pips:point-added',point:event},'*');pointForm.hidden=true;setAdding(false);
};
canvas.addEventListener('pointerdown',e=>{drag={x:e.clientX,y:e.clientY,startX:e.clientX,startY:e.clientY,moved:false};canvas.setPointerCapture(e.pointerId);canvas.style.cursor=adding?'crosshair':'grabbing';tip.style.display='none';});
canvas.addEventListener('pointerup',e=>{if(adding&&drag&&!drag.moved){const x=(e.offsetX-ox)/zoom,y=(e.offsetY-oy)/zoom;
 if(x>=0&&y>=0&&x<data.width&&y<data.height){candidate={x,y};pointForm.hidden=false;document.querySelector('#pointName').value='Point '+(data.points.length+1);document.querySelector('#pointMessage').textContent='';document.querySelector('#pointName').focus();queueDraw();}}
 drag=null;canvas.style.cursor=adding?'crosshair':'grab';});canvas.addEventListener('pointercancel',()=>{drag=null;canvas.style.cursor=adding?'crosshair':'grab';});
canvas.addEventListener('pointermove',e=>{if(drag){drag.moved ||= Math.hypot(e.clientX-drag.startX,e.clientY-drag.startY)>5;
 if(!adding){ox+=e.clientX-drag.x;oy+=e.clientY-drag.y;}drag.x=e.clientX;drag.y=e.clientY;queueDraw();return;}
 const x=(e.offsetX-ox)/zoom,y=(e.offsetY-oy)/zoom,p=data.points.find(p=>Math.hypot(ox+p.x*zoom-e.offsetX,oy+p.y*zoom-e.offsetY)<12);
 if(!current||x<0||y<0||x>=data.width||y>=data.height){tip.style.display='none';return;}
 const score=current.raw[Math.floor(y)*data.width+Math.floor(x)]/255;
 tip.textContent=(p?p.name+'\n':'')+'Model score ≈ '+score.toFixed(3);tip.style.left=Math.min(e.offsetX+12,cw-200)+'px';tip.style.top=Math.max(0,e.offsetY-50)+'px';tip.style.display='block';});
canvas.addEventListener('pointerleave',()=>{tip.style.display='none';});
canvas.addEventListener('keydown',e=>{if(e.key==='+'||e.key==='=')magnify(1.3);else if(e.key==='-')magnify(1/1.3);
 else if(e.key==='ArrowLeft')ox+=25;else if(e.key==='ArrowRight')ox-=25;else if(e.key==='ArrowUp')oy+=25;else if(e.key==='ArrowDown')oy-=25;else return;e.preventDefault();queueDraw();});
loadImage(data.background).then(image=>{background=image;return selectFrame();}).catch(()=>{label.textContent='Unable to load background';});
</script></body></html>
'''


map_component = st.components.v2.component(
    "pips_clickable_map",
    html='<iframe title="Interactive wildfire prediction monitor" style="display:block;width:100%;height:520px;border:0"></iframe>',
    js="""export default function({data,parentElement,setTriggerValue}) {
        const frame=parentElement.querySelector('iframe');
        const resize=()=>{frame.style.height=Math.min(660,Math.max(320,window.innerHeight-180))+'px';};
        resize();
        window.addEventListener('resize',resize);
        if(frame.srcdoc!==data.html) frame.srcdoc=data.html;
        const receive=event=>{
            if(event.source===frame.contentWindow && event.data?.type==='pips:point-added'
               && event.data.point.fire_token===data.fire_token)
                setTriggerValue('point_added',event.data.point);
        };
        window.addEventListener('message',receive);
        return ()=>{window.removeEventListener('message',receive);window.removeEventListener('resize',resize);};
    }""",
)


def point_from_map(fire, event, existing):
    """Convert continuous canvas pixels through the source CRS to WGS84."""
    x, y = float(event["x"]), float(event["y"])
    name = str(event["name"]).strip()
    value, radius = float(event["value"]), float(event["radius_km"])
    if not np.isfinite([x, y, value, radius]).all():
        raise ValueError("Point values must be finite.")
    if not (0 <= x < fire["shape"][1] and 0 <= y < fire["shape"][0]):
        raise ValueError("Click inside the fire image.")
    if not name or len(name) > 100 or name in [p["name"] for p in existing]:
        raise ValueError("Choose a unique location name of 1–100 characters.")
    if not (1 <= value <= 10 and .1 <= radius <= 20):
        raise ValueError("Use a value of 1–10 and a buffer of 0.1–20 km.")
    east, north = fire["transform"] * (x, y)
    lon, lat = transform(fire["crs"], "EPSG:4326", [east], [north])
    return dict(name=name, longitude=lon[0], latitude=lat[0], value=value, radius_km=radius)


def png_uri(array):
    stream = io.BytesIO()
    Image.fromarray(array).save(stream, format="PNG")
    return "data:image/png;base64," + base64.b64encode(stream.getvalue()).decode()


def viewer_html(fire, result, points, view=None):
    observed = {key: fire[key] for key in ("active", "burned")}
    observed["new"] = np.maximum(fire["burned"] - (
        fire["images"][-2][28] if len(fire["images"]) > 1 else fire["burned"]), 0)
    observed["date"] = fire["dates"][-1]
    frames = [observed, *(result["forecasts"] if result else [])]
    locations = []
    for point in points:
        x, y = transform("EPSG:4326", fire["crs"], [point["longitude"]], [point["latitude"]])
        col, row = ~fire["transform"] * (x[0], y[0])
        locations.append(dict(name=point["name"], x=col, y=row))
    # Display-only quantization saves transfer/memory. Inference and decisions
    # retain the original float32 probabilities; hover values say approximate.
    data = dict(background=png_uri(fire["rgb"]), width=fire["shape"][1], height=fire["shape"][0],
                fire_token=str(id(fire)), view=view,
                points=locations, dates=[f["date"] for f in frames],
                frames={key: [png_uri(np.rint(np.clip(f[key], 0, 1)*255).astype(np.uint8)) for f in frames]
                        for key in ("active", "burned", "new")})
    payload = json.dumps(data).replace("<", "\\u003c")
    return MAP_HTML.replace("__PAYLOAD__", payload)


def main():
    st.set_page_config(page_title="PIPS | Fire forecast", layout="wide")
    st.title("Fire forecast")


    def load(folder):
        fire = load_fire(folder)
        points = read_points(Path(folder) / "points.geojson")
        st.session_state.fire = fire
        st.session_state.points = points
        st.session_state.pop("result", None)
        st.session_state.pop("forecast_day", None)
        st.session_state.pop("map_view", None)
        st.session_state.editor_version = st.session_state.get("editor_version", 0) + 1


    with st.sidebar:
        st.header("Fire history")
        mode = st.radio("Upload format", ["ZIP", "Folder"], horizontal=True)
        uploads = st.file_uploader("Upload one fire", type=["zip"] if mode == "ZIP" else ["tif", "geojson"],
                                  accept_multiple_files=False if mode == "ZIP" else "directory",
                                  max_upload_size=512)
        if st.button("Load upload", disabled=not uploads, width="stretch"):
            try:
                with st.spinner("Reading and aligning the uploaded history…"):
                    with tempfile.TemporaryDirectory(prefix="pips-upload-") as temp:
                        save_uploads([uploads] if mode == "ZIP" else uploads, temp)
                        load(temp)
                        st.session_state.fire["name"] = "Uploaded fire"
            except Exception as error:
                st.error(str(error))
        with st.expander("Required folder format"):
            st.code("fire/\n  VIIRS_Day/YYYY-MM-DD_VIIRS_Day.tif\n  VIIRS_Night/YYYY-MM-DD_VIIRS_Night.tif\n  FirePred/YYYY-MM-DD_FirePred.tif\n  points.geojson  (optional)")
            st.write("Use consecutive dates from one fire. Day TIFFs contain the original eight TS-SatFire bands. Missing night or environmental measurements use the trained validity-mask handling.")
        st.header("Forecast")
        days = st.number_input("Maximum future days", 1, 64, 32)
        threshold = st.slider("Exposure trigger", .05, .95, .50, .05,
                              help="A pixel in a location’s buffer must cross this model score. This is not confirmed fire arrival.")
        stop = st.checkbox("Stop at the first point reached", value=True)
        run = st.button("Run forecast", type="primary", width="stretch",
                        disabled="fire" not in st.session_state)

    if "fire" not in st.session_state:
        st.info("Upload a fire to begin.")
        st.stop()

    fire = st.session_state.fire
    st.subheader(f"{fire['name'].replace('_', ' ').title()}")
    st.caption(f"{fire['dates'][0]} – {fire['dates'][-1]}  |  {len(fire['dates'])} observed days")

    with st.expander("Edit points of interest", expanded=False):
        st.caption("Click Add point on the map to place a location.")
        columns = ["name", "latitude", "longitude", "value", "radius_km"]
        edited = st.data_editor(pd.DataFrame(st.session_state.points, columns=columns),
                               num_rows="dynamic", hide_index=True, width="stretch",
                               key=f"poi_editor_{st.session_state.get('editor_version', 0)}",
                               column_config={
                                   "name": st.column_config.TextColumn("Location", required=True),
                                   "latitude": st.column_config.NumberColumn("Latitude", required=True, min_value=-90., max_value=90., format="%.5f"),
                                   "longitude": st.column_config.NumberColumn("Longitude", required=True, min_value=-180., max_value=180., format="%.5f"),
                                   "value": st.column_config.NumberColumn("Protection value", required=True, min_value=1., max_value=10., default=5.),
                                   "radius_km": st.column_config.NumberColumn("Buffer (km)", required=True, min_value=.1, max_value=20., default=2.),
                               })
        points = edited.dropna(how="all").to_dict("records")
        try:
            if any(any(pd.isna(v) for v in p.values()) for p in points):
                raise ValueError("Complete every field in the points table.")
            point_masks(fire, points)
            valid_points = True
        except (ValueError, TypeError) as error:
            st.error(str(error))
            valid_points = False

    if run and valid_points:
        status = st.status("Running epoch 50…", expanded=True)
        bar = status.progress(0.)
        try:
            result = run_forecast(fire, points, int(days), threshold, stop,
                                  lambda done, total, message: bar.progress(done / total, text=message))
            st.session_state.result = result
            st.session_state.forecast_day = 1
            status.update(label=f"Finished {len(result['forecasts'])} forecast days", state="complete", expanded=False)
        except Exception as error:
            status.update(label="Forecast did not complete", state="error")
            st.error(str(error))

    result = st.session_state.get("result")
    if result:
        changed = (result["points"] != points or result["threshold"] != threshold
                   or result["requested_days"] != days or result.get("stop_on_hit", True) != stop)
        if changed:
            st.warning("Settings changed. Results below belong to the previous run; run again to update them.")
        st.subheader("Recommended action")
        st.info(result["recommendation"])
        st.caption("Planning guidance for operator review.")
        if result["ranking"]:
            st.dataframe(pd.DataFrame(result["ranking"]).rename(columns={
                "name": "Location", "value": "Assigned value", "priority": "Priority score",
                "first_exposure_day": "First trigger (+days)", "peak_exposure": "Peak exposure score",
                "observed_active": "Active at issuance", "already_burned_fraction": "Already burned fraction",
            }), hide_index=True, width="stretch")
        report = {key: value for key, value in result.items() if key != "forecasts"}
        report.update(history_dates=fire["dates"], forecast_dates=[x["date"] for x in result["forecasts"]])
        st.download_button("Download decision report", json.dumps(report, indent=2), "decision.json", "application/json")

    st.subheader("Forecast map")
    display_points = points if valid_points else []
    signature = (id(fire), id(result), json.dumps(display_points, sort_keys=True),
                 Path(__file__).stat().st_mtime_ns)
    if st.session_state.get("monitor_signature") != signature:
        st.session_state.monitor_html = viewer_html(fire, result, display_points, st.session_state.get("map_view"))
        st.session_state.monitor_signature = signature
    map_event = map_component(
        data={"html": st.session_state.monitor_html, "fire_token": str(id(fire))},
        key="prediction_monitor", on_point_added_change=lambda: None,
    )
    event = map_event.point_added
    if event and event.get("id") != st.session_state.get("last_map_event"):
        st.session_state.last_map_event = event.get("id")
        try:
            if event.get("fire_token") != str(id(fire)) or not valid_points:
                raise ValueError("Reload the current map or finish the existing point edits first.")
            point = point_from_map(fire, event, points)
            updated_points = [*points, point]
            masks = point_masks(fire, updated_points)
            st.session_state.points = updated_points
            st.session_state.editor_version = st.session_state.get("editor_version", 0) + 1
            st.session_state.map_view = event["view"]
            if result:
                result["points"] = updated_points
                result["ranking"] = rank_points(fire, updated_points, masks, result["forecasts"], result["threshold"])
                result["recommendation"] = recommendation(result["ranking"], len(result["forecasts"]), result["stopped"])
            st.rerun()
        except (ValueError, KeyError, TypeError) as error:
            st.error(str(error))
    with st.expander("Model and decision details"):
        st.write("PIPS epoch 50 uses probability feedback. Future imagery and weather stay fixed at the last observed day. The model does not simulate suppression actions, resource effectiveness, or safe evacuation routes.")
        st.write("For each buffer and forecast day, exposure is the larger of the maximum active-fire score and maximum cumulative-burned score over pixels unburned at issuance. Priority = maximum across simulated days of assigned value × exposure / (1 + forecast lead). The recommendation chooses the highest-priority threshold-crossing point; otherwise it recommends monitoring. These scores are not calibrated asset-impact probabilities.")
        st.write("Stopping at the first reached point limits the ranking to that simulated horizon. Disable the stop option to compare all points through the requested final day. Burned does not mean safe.")
        st.caption(f"Checkpoint: {CHECKPOINT.name}. No future observations or labels are used.")


if __name__ == "__main__":
    main()
