const $ = id => document.getElementById(id);
const video=$('video'), canvas=$('handCanvas'), ctx=canvas.getContext('2d');
const cameraStage=$('cameraStage'), cameraPlaceholder=$('cameraPlaceholder'), cameraStatus=$('cameraStatus'), liveBadge=$('liveBadge'), trackingBadge=$('trackingBadge');
const startBtn=$('startCameraBtn'), stopBtn=$('stopCameraBtn'), switchBtn=$('switchCameraBtn'), fullscreenBtn=$('fullscreenBtn'), installBtn=$('installBtn');
const detectedSign=$('detectedSign'), confidenceValue=$('confidenceValue'), confidenceBar=$('confidenceBar'), translationText=$('translationText'), typePill=$('typePill'), recognitionState=$('recognitionState');
const toast=$('toast');
const arOverlay=$('arOverlay'), arSign=$('arSign'), arMeaning=$('arMeaning');
const signChart=$('signChart'), signSearch=$('signSearch'), signCategory=$('signCategory');
const navButtons=[...document.querySelectorAll('.nav-btn')], pageSections=[...document.querySelectorAll('.page-section')];
const sentenceOutput=$('sentenceOutput'), rawOutput=$('rawOutput'), speakBtn=$('speakBtn'), deleteBtn=$('deleteBtn'), clearSentenceBtn=$('clearSentenceBtn'), autoSpeakToggle=$('autoSpeakToggle'), voiceLanguage=$('voiceLanguage');

let stream=null, facingMode='user', hands=null, trackingLoopId=null, processingFrame=false, lastVideoTime=-1;
let sentenceTokens=[], deferredInstallPrompt=null;
// Recognition Engine V3: a lightweight adaptive landmark model learns from high-confidence confirmed signs.
const MODEL_KEY='signar-adaptive-landmark-model-v1';
let adaptiveModel=loadAdaptiveModel();
let latestFeatureVectors=[];
let handTrails=[[],[]];
let candidateLabel='', candidateType='', candidateFrames=0, candidateScoreSum=0;
let gestureLocked=false, lockedLabel='', releaseFrames=0, ambiguousFrames=0;
const REQUIRED_STABLE_FRAMES=9;
const REQUIRED_RELEASE_FRAMES=7;
const MIN_COMMIT_CONFIDENCE=78;
const MP_VERSION='0.4.1646424915';
const MP_BASE=`https://cdn.jsdelivr.net/npm/@mediapipe/hands@${MP_VERSION}`;
const CONNECTIONS=[[0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],[5,9],[9,10],[10,11],[11,12],[9,13],[13,14],[14,15],[15,16],[13,17],[17,18],[18,19],[19,20],[0,17]];


function loadAdaptiveModel(){try{return JSON.parse(localStorage.getItem(MODEL_KEY)||'{}')}catch{return {}}}
function saveAdaptiveModel(){try{localStorage.setItem(MODEL_KEY,JSON.stringify(adaptiveModel))}catch{}}
function normalizedVector(lm){
  const wrist=lm[0], scale=dist(lm[5],lm[17])||.08, out=[];
  for(const p of lm){out.push((p.x-wrist.x)/scale,(p.y-wrist.y)/scale,(p.z||0)/scale)}
  return out;
}
function vectorDistance(a,b){if(!a||!b||a.length!==b.length)return Infinity;let s=0;for(let i=0;i<a.length;i++){const d=a[i]-b[i];s+=d*d}return Math.sqrt(s/a.length)}
function learnExample(label,vectors){if(!label||!vectors?.length)return;const key=label+'#'+vectors.length;const v=vectors.flat();const item=adaptiveModel[key]||{label,count:0,centroid:v.slice()};item.count=Math.min(60,(item.count||0)+1);const rate=Math.max(.08,1/item.count);if(item.centroid.length!==v.length)item.centroid=v.slice();else for(let i=0;i<v.length;i++)item.centroid[i]=item.centroid[i]*(1-rate)+v[i]*rate;adaptiveModel[key]=item;saveAdaptiveModel()}
const LEGACY_LABELS={HELLO:'KUMUSTA','THANK YOU':'SALAMAT',SORRY:'PAUMANHIN','I LOVE YOU':'MAHAL KITA',GOOD:'MABUTI',BAD:'MASAMA',HELP:'TULONG',STOP:'TIGIL',YOU:'IKAW',WE:'TAYO',MORE:'DAGDAG'};
function modelPrediction(vectors){if(!vectors?.length)return null;const v=vectors.flat();let best=null;for(const item of Object.values(adaptiveModel)){if(!item?.centroid||item.centroid.length!==v.length||(item.count||0)<2)continue;const d=vectorDistance(v,item.centroid);if(!best||d<best.distance)best={label:LEGACY_LABELS[item.label]||item.label,distance:d,count:item.count}}if(!best||best.distance>.34)return null;return {...best,confidence:Math.round(clamp(100-best.distance*145,55,97))}}
function hybridize(ruleResult,vectors){const ml=modelPrediction(vectors);if(!ml)return ruleResult;if(!ruleResult)return ml.confidence>=88?{label:ml.label,confidence:ml.confidence,type:'Adaptive model',dynamic:false,model:true}:null;if(ml.label===ruleResult.label)return {...ruleResult,confidence:Math.min(98,Math.round(ruleResult.confidence*.72+ml.confidence*.28)),model:true};if(ml.confidence>=94&&ruleResult.confidence<88)return {label:ml.label,confidence:ml.confidence,type:'Adaptive model',dynamic:false,model:true};return {...ruleResult,confidence:Math.max(55,ruleResult.confidence-6)} }

function showToast(m){toast.textContent=m;toast.classList.add('show');clearTimeout(showToast.t);showToast.t=setTimeout(()=>toast.classList.remove('show'),2600)}
function setState(text,cls=''){recognitionState.textContent=text;recognitionState.className=`recognition-state ${cls}`.trim()}
function resetResult(msg='No sign detected yet.'){detectedSign.textContent='Waiting...';confidenceValue.textContent='0%';confidenceBar.style.width='0%';translationText.textContent=msg;typePill.textContent='Automatic';setState('SEARCHING');setARFeedback()}
function setMirror(){const mirror=facingMode==='user';video.classList.toggle('mirror',mirror);canvas.classList.toggle('mirror',mirror)}
function resizeCanvas(){const w=video.videoWidth||video.clientWidth||1280,h=video.videoHeight||video.clientHeight||720;if(canvas.width!==w||canvas.height!==h){canvas.width=w;canvas.height=h;handTrails=[[],[]]}}
const dist=(a,b)=>Math.hypot(a.x-b.x,a.y-b.y);
const clamp=(v,a,b)=>Math.max(a,Math.min(b,v));
const avg=a=>a.length?a.reduce((s,v)=>s+v,0)/a.length:0;

function angle(a,b,c){const ab={x:a.x-b.x,y:a.y-b.y}, cb={x:c.x-b.x,y:c.y-b.y};const dot=ab.x*cb.x+ab.y*cb.y;const mag=Math.hypot(ab.x,ab.y)*Math.hypot(cb.x,cb.y)||1;return Math.acos(clamp(dot/mag,-1,1))*180/Math.PI}
function drawTrail(trail){if(trail.length<2)return;ctx.save();ctx.lineWidth=Math.max(2,canvas.width/500);ctx.strokeStyle='rgba(110,231,183,.38)';ctx.beginPath();trail.forEach((p,i)=>{const x=p.x*canvas.width,y=p.y*canvas.height;i?ctx.lineTo(x,y):ctx.moveTo(x,y)});ctx.stroke();ctx.restore()}
function drawHand(lm,index,label,confidence){const w=canvas.width,h=canvas.height;ctx.save();ctx.lineWidth=Math.max(2,w/420);ctx.strokeStyle=index===0?'#38bdf8':'#a78bfa';ctx.lineCap='round';CONNECTIONS.forEach(([a,b])=>{ctx.beginPath();ctx.moveTo(lm[a].x*w,lm[a].y*h);ctx.lineTo(lm[b].x*w,lm[b].y*h);ctx.stroke()});lm.forEach((p,i)=>{ctx.beginPath();ctx.arc(p.x*w,p.y*h,i===0?Math.max(5,w/180):Math.max(3.2,w/260),0,Math.PI*2);ctx.fillStyle=i===0?'#f8fafc':'#6ee7b7';ctx.fill()});const xs=lm.map(p=>p.x*w),ys=lm.map(p=>p.y*h);const minX=Math.min(...xs),maxX=Math.max(...xs),minY=Math.min(...ys),maxY=Math.max(...ys);ctx.strokeStyle='rgba(110,231,183,.5)';ctx.setLineDash([8,7]);ctx.strokeRect(minX-14,minY-18,maxX-minX+28,maxY-minY+36);ctx.setLineDash([]);if(label){const txt=`${label}  ${confidence}%`;ctx.font=`700 ${Math.max(14,w/55)}px system-ui`;const tw=ctx.measureText(txt).width;const x=Math.max(8,Math.min(w-tw-26,minX-12)),y=Math.max(34,minY-28);ctx.fillStyle='rgba(3,10,18,.86)';ctx.fillRect(x,y-28,tw+20,36);ctx.fillStyle='#ecfeff';ctx.fillText(txt,x+10,y-4)}ctx.restore()}

function updateMotion(lm,handIndex){const p=lm[8],w=lm[0],now=performance.now();const t=handTrails[handIndex]||[];t.push({x:p.x,y:p.y,wx:w.x,wy:w.y,t:now});handTrails[handIndex]=t.filter(a=>now-a.t<1100).slice(-40)}
function motionStats(handIndex=0){const t=handTrails[handIndex]||[];if(t.length<6)return {dx:0,dy:0,path:0,turns:0,speed:0,verticalTurns:0};let path=0,turns=0,verticalTurns=0,lastX=0,lastY=0;for(let i=1;i<t.length;i++){const dx=t[i].x-t[i-1].x,dy=t[i].y-t[i-1].y;path+=Math.hypot(dx,dy);const sx=Math.sign(dx),sy=Math.sign(dy);if(lastX&&sx&&sx!==lastX)turns++;if(lastY&&sy&&sy!==lastY)verticalTurns++;if(sx)lastX=sx;if(sy)lastY=sy}const first=t[0],last=t[t.length-1],dt=Math.max(1,last.t-first.t);return {dx:last.x-first.x,dy:last.y-first.y,path,turns,verticalTurns,speed:path/(dt/1000)}}

function features(lm){
  const wrist=lm[0], palm=dist(lm[5],lm[17])||.08;
  const fingers=[
    {tip:8,pip:6,mcp:5}, {tip:12,pip:10,mcp:9}, {tip:16,pip:14,mcp:13}, {tip:20,pip:18,mcp:17}
  ];
  const curls=fingers.map(f=>angle(lm[f.mcp],lm[f.pip],lm[f.tip]));
  const extended=curls.map(a=>a>150);
  const [index,middle,ring,pinky]=extended;
  const thumbAngle=angle(lm[2],lm[3],lm[4]);
  const thumbOpen=dist(lm[4],lm[5])>palm*.62 && thumbAngle>135;
  const thumbIndex=dist(lm[4],lm[8])/palm;
  const pinch=thumbIndex<.34;
  const palmOpen=extended.filter(Boolean).length;
  const palmAngle=Math.atan2(lm[17].y-lm[5].y,lm[17].x-lm[5].x)*180/Math.PI;
  const thumbVector={x:lm[4].x-lm[2].x,y:lm[4].y-lm[2].y};
  const thumbUp=thumbVector.y<-.055 && Math.abs(thumbVector.y)>Math.abs(thumbVector.x)*.75;
  const thumbDown=thumbVector.y>.055 && Math.abs(thumbVector.y)>Math.abs(thumbVector.x)*.75;
  const spread=dist(lm[8],lm[20])/palm;
  const fist=palmOpen===0 && !thumbOpen;
  const flat=palmOpen===4 && spread>1.45;
  return {lm,palm,wrist,index,middle,ring,pinky,extended,curls,thumbOpen,thumbIndex,pinch,palmOpen,palmAngle,thumbUp,thumbDown,spread,fist,flat};
}

function scoreSingle(f,m){
  const candidates=[];
  // Expression: I LOVE YOU — index + pinky extended, middle/ring folded, thumb open.
  if(f.index&&f.pinky&&!f.middle&&!f.ring&&f.thumbOpen){const shape=96-clamp(Math.abs(f.curls[1]-90),0,20)*.2;candidates.push({label:'MAHAL KITA',confidence:Math.round(shape),type:'Expression',dynamic:false})}
  // GOOD/BAD: thumb direction plus a compact fist prevents random open-hand matches.
  if(f.thumbUp&&f.palmOpen<=1&&f.thumbOpen)candidates.push({label:'MABUTI',confidence:91,type:'Expression',dynamic:false});
  if(f.thumbDown&&f.palmOpen<=1&&f.thumbOpen)candidates.push({label:'MASAMA',confidence:89,type:'Expression',dynamic:false});
  // HELLO: requires an open hand and a real lateral wave with multiple direction changes.
  if(f.flat&&m.path>.13&&m.turns>=2&&Math.abs(m.dx)<.16)candidates.push({label:'KUMUSTA',confidence:93,type:'Common word',dynamic:true});
  // STOP: open palm must be settled and not in the middle of a wave.
  if(f.flat&&m.path<.045&&m.speed<.08)candidates.push({label:'TIGIL',confidence:90,type:'Common word',dynamic:false});
  // YOU: one index extended, others folded, motion nearly still.
  if(f.index&&!f.middle&&!f.ring&&!f.pinky&&!f.thumbOpen&&m.path<.06)candidates.push({label:'IKAW',confidence:86,type:'Common word',dynamic:false});
  // SORRY: fist making a small circular/alternating movement.
  if(f.fist&&m.path>.12&&m.turns>=1&&m.verticalTurns>=1)candidates.push({label:'PAUMANHIN',confidence:82,type:'Expression',dynamic:true});
  // THANK YOU: open palm moving forward/downward.
  if(f.flat&&m.path>.08&&m.dy>.045&&m.turns<=1)candidates.push({label:'SALAMAT',confidence:82,type:'Expression',dynamic:true});
  // YES: closed hand with a short vertical nod/bounce.
  if(f.fist&&m.path>.07&&Math.abs(m.dy)>.025&&m.verticalTurns>=1)candidates.push({label:'OO',confidence:84,type:'Common word',dynamic:true});
  // NO: thumb/index close together while remaining fingers are mostly folded; small repeated motion helps disambiguate.
  if(f.pinch&&f.palmOpen<=2&&m.path>.035)candidates.push({label:'HINDI',confidence:83,type:'Common word',dynamic:true});
  // WHERE: one upright index with a small side-to-side movement.
  if(f.index&&!f.middle&&!f.ring&&!f.pinky&&m.path>.055&&m.turns>=1)candidates.push({label:'SAAN',confidence:85,type:'Question',dynamic:true});
  // EAT / DRINK: compact pinch shapes; motion direction separates the two reference gestures.
  if(f.pinch&&f.palmOpen<=1&&m.path>.035&&m.dy<-.015)candidates.push({label:'KAIN',confidence:80,type:'Common word',dynamic:true});
  if(f.pinch&&f.palmOpen<=1&&m.path>.035&&m.dy>.015)candidates.push({label:'INOM',confidence:80,type:'Common word',dynamic:true});
  // WHEN: upright index with a circular/turning path.
  if(f.index&&!f.middle&&!f.ring&&!f.pinky&&m.path>.08&&m.turns>=1&&m.verticalTurns>=1)candidates.push({label:'KAILAN',confidence:82,type:'Question',dynamic:true});
  return candidates.sort((a,b)=>b.confidence-a.confidence)[0]||null;
}

function scoreTwoHand(featuresArr){
  if(featuresArr.length<2)return null;
  const a=featuresArr[0],b=featuresArr[1];
  const wa=a.wrist,wb=b.wrist,centerDist=dist(wa,wb)/Math.max(.06,(a.palm+b.palm)/2);
  const verticalGap=Math.abs(wa.y-wb.y), horizontalGap=Math.abs(wa.x-wb.x);
  // HELP: one compact fist positioned just above an open supporting palm.
  const fist=a.fist?a:(b.fist?b:null), open=a.flat?a:(b.flat?b:null);
  if(fist&&open){const above=fist.wrist.y<open.wrist.y+.08;const aligned=Math.abs(fist.wrist.x-open.wrist.x)<.18;if(above&&aligned&&centerDist<3.8)return {label:'TULONG',confidence:91,type:'Common word',dynamic:false,twoHand:true}}
  // MORE: both hands use a pinch/closed fingertip shape and are brought close together.
  if(a.pinch&&b.pinch&&horizontalGap<.28&&verticalGap<.22)return {label:'DAGDAG',confidence:88,type:'Common word',dynamic:false,twoHand:true};
  // WE: both index fingers extended and hands held apart at similar height.
  const oneIndex=f=>f.index&&!f.middle&&!f.ring&&!f.pinky;
  if(oneIndex(a)&&oneIndex(b)&&verticalGap<.18&&horizontalGap>.18)return {label:'TAYO',confidence:84,type:'Common word',dynamic:false,twoHand:true};
  // PLEASE: two open hands close together, approximately parallel.
  if(a.flat&&b.flat&&centerDist<3.4&&verticalGap<.20)return {label:'PAKIUSAP',confidence:82,type:'Expression',dynamic:false,twoHand:true};
  // PAIN: two compact/index-oriented hands facing each other near the same height.
  if(oneIndex(a)&&oneIndex(b)&&horizontalGap<.30&&verticalGap<.18)return {label:'SAKIT',confidence:82,type:'Common word',dynamic:false,twoHand:true};
  // WHAT: both open hands separated and held at similar height.
  if(a.flat&&b.flat&&horizontalGap>.20&&verticalGap<.20)return {label:'ANO',confidence:81,type:'Question',dynamic:false,twoHand:true};
  // HOW: two compact hands close together.
  if(a.fist&&b.fist&&horizontalGap<.28&&verticalGap<.20)return {label:'PAANO',confidence:81,type:'Question',dynamic:false,twoHand:true};
  return null;
}

const SIGN_TEXT={
  'KUMUSTA':{fil:'Kumusta!',en:'Hello!'},
  'SALAMAT':{fil:'Salamat!',en:'Thank you!'},
  'PAUMANHIN':{fil:'Paumanhin.',en:'Sorry.'},
  'MAHAL KITA':{fil:'Mahal kita.',en:'I love you.'},
  'MABUTI':{fil:'Mabuti.',en:'Good.'},
  'MASAMA':{fil:'Masama.',en:'Bad.'},
  'TULONG':{fil:'Tulong.',en:'Help.'},
  'TIGIL':{fil:'Tigil.',en:'Stop.'},
  'IKAW':{fil:'Ikaw.',en:'You.'},
  'TAYO':{fil:'Tayo.',en:'We.'},
  'DAGDAG':{fil:'Dagdag.',en:'More.'},
  'OO':{fil:'Oo.',en:'Yes.'},
  'HINDI':{fil:'Hindi.',en:'No.'},
  'PAKIUSAP':{fil:'Pakiusap.',en:'Please.'},
  'SAKIT':{fil:'Masakit.',en:'Pain.'},
  'KAIN':{fil:'Kain.',en:'Eat.'},
  'INOM':{fil:'Inom.',en:'Drink.'},
  'PAANO':{fil:'Paano?',en:'How?'},
  'ANO':{fil:'Ano?',en:'What?'},
  'SAAN':{fil:'Saan?',en:'Where?'},
  'KAILAN':{fil:'Kailan?',en:'When?'}
};
const SIGN_CATEGORIES={
  'KUMUSTA':'common','SALAMAT':'common','OO':'common','HINDI':'common','TULONG':'common','TIGIL':'common','DAGDAG':'common','IKAW':'common','TAYO':'common','MABUTI':'expression','MASAMA':'expression','MAHAL KITA':'expression','PAKIUSAP':'expression','PAUMANHIN':'expression','SAKIT':'common','KAIN':'common','INOM':'common','PAANO':'question','ANO':'question','SAAN':'question','KAILAN':'question'
};
const SIGN_TIPS={
  'KUMUSTA':'Supported dynamic open-hand gesture.','SALAMAT':'Supported dynamic open-hand gesture.','OO':'Supported dynamic closed-hand motion.','HINDI':'Supported pinch/motion gesture.','TULONG':'Supported two-hand gesture.','MAHAL KITA':'Supported expression gesture.','PAKIUSAP':'Supported two-hand expression gesture.','PAUMANHIN':'Supported dynamic expression gesture.','TIGIL':'Supported steady open-palm gesture.','SAKIT':'Supported two-hand gesture.','KAIN':'Supported motion gesture.','INOM':'Supported motion gesture.','DAGDAG':'Supported two-hand gesture.','PAANO':'Supported two-hand question gesture.','ANO':'Supported two-hand question gesture.','SAAN':'Supported moving index gesture.','KAILAN':'Supported moving index gesture.','IKAW':'Supported single index gesture.','TAYO':'Supported two-hand gesture.','MABUTI':'Supported thumb-up expression.','MASAMA':'Supported thumb-down expression.'
};
// --- Learn FSL visual/practice layer -------------------------------------------------
const SIGN_PREVIEWS={
  'KUMUSTA':'kumusta.jpg','SALAMAT':'salamat.jpg','PAUMANHIN':'paumanhin.jpg','MAHAL KITA':'mahal_kita.jpg','MABUTI':'mabuti.jpg','MASAMA':'masama.jpg','TULONG':'tulong.jpg','TIGIL':'tigil.jpg','IKAW':'ikaw.jpg','TAYO':'tayo.jpg','DAGDAG':'dagdag.jpg','OO':'oo.jpg'
};
const FSL_SIGNBANK_URL='https://cead.benilde.edu.ph/fsl-signbank/';
const PRACTICE_GUIDES={
  'KUMUSTA':'Use the complete 2D reference, then practice the supported camera gesture.',
  'SALAMAT':'Use the complete 2D reference, then practice the supported camera gesture.',
  'PAUMANHIN':'Follow the hand shape and movement shown by the reference.',
  'MAHAL KITA':'Follow the hand configuration shown by the reference.',
  'MABUTI':'Match the hand configuration shown by the reference.',
  'MASAMA':'Match the hand configuration shown by the reference.',
  'TULONG':'Use both hands and copy their relative position from the reference.',
  'TIGIL':'Raise the hand as shown and hold the gesture clearly.',
  'IKAW':'Extend the index finger as shown by the reference.',
  'TAYO':'Use both hands and follow the relative position shown.',
  'DAGDAG':'Use both hands and follow the demonstrated movement.',
  'OO':'Copy the hand configuration and movement shown.',
  'HINDI':'Verify the exact FSL form in the SignBank before practicing.',
  'PAKIUSAP':'Verify the exact two-hand FSL form in the SignBank before practicing.',
  'SAKIT':'Verify the exact FSL form in the SignBank before practicing.',
  'KAIN':'Verify the exact FSL form in the SignBank before practicing.',
  'INOM':'Verify the exact FSL form in the SignBank before practicing.',
  'PAANO':'Verify the exact two-hand FSL form in the SignBank before practicing.',
  'ANO':'Verify the exact FSL form in the SignBank before practicing.',
  'SAAN':'Verify the exact FSL form in the SignBank before practicing.',
  'KAILAN':'Verify the exact FSL form in the SignBank before practicing.'
};
let activePracticeSign='KUMUSTA';
let practice3DState=null;
let practiceCameraStream=null;

function previewAsset(label){
  return SIGN_PREVIEWS[label] ? `assets/sign-previews/${SIGN_PREVIEWS[label]}` : '';
}
function renderSignPreview(label){
  const asset=previewAsset(label);
  if(!asset){
    return `<div class="fsl-source-preview"><strong>Authentic FSL reference</strong><span>This sign needs verification from the Benilde FSL SignBank rather than an approximate or generated hand.</span><a href="${FSL_SIGNBANK_URL}" target="_blank" rel="noopener">Open FSL SignBank ↗</a></div>`;
  }
  return `<div class="sign-card-preview"><img src="${asset}" alt="${label} Filipino Sign Language gesture reference" loading="lazy"><span class="preview-badge">2D guide</span></div>`;
}

// Procedural 3D hand: a lightweight offline-friendly model built from palm, joints and finger bones.
// The poses are visual guides for the prototype vocabulary; they are not a claim of complete FSL coverage.
const POSE_TYPES={
  open:'open',side:'side',thumbUp:'thumbUp',thumbDown:'thumbDown',point:'point',ilove:'ilove',fist:'fist',pinch:'pinch',two:'two'
};
function handPose(type='open', side=0){
  const wrist={x:0,y:-1.55,z:0};
  const mcp=[
    {x:-.72,y:.05,z:0},{x:-.24,y:.18,z:0},{x:.25,y:.18,z:0},{x:.68,y:.06,z:0}
  ];
  const dirs=[[-.20,.92,.02],[-.07,1.04,.02],[.06,1.08,.02],[.20,.96,.02]];
  const lengths=[.72,.90,.86,.72];
  const pts=[wrist,{x:-.86,y:-.42,z:.04},{x:-.38,y:-.12,z:.03},{x:.18,y:-.12,z:.03},{x:.74,y:-.38,z:.04}];
  for(let i=0;i<4;i++){
    const base=mcp[i], d=dirs[i], L=lengths[i];
    let p1={x:base.x+d[0]*L*.38,y:base.y+d[1]*L*.38,z:d[2]};
    let p2={x:base.x+d[0]*L*.70,y:base.y+d[1]*L*.70,z:d[2]*1.4};
    let p3={x:base.x+d[0]*L,y:base.y+d[1]*L,z:d[2]*1.7};
    if(type==='fist'){
      p1={x:base.x+d[0]*.08,y:base.y-.12,z:.12};p2={x:base.x+d[0]*.16,y:base.y-.30,z:.18};p3={x:base.x+d[0]*.10,y:base.y-.45,z:.22};
    } else if(type==='point' && i!==0){
      p1={x:base.x+d[0]*.20,y:base.y-.20,z:.10};p2={x:base.x+d[0]*.34,y:base.y-.32,z:.14};p3={x:base.x+d[0]*.25,y:base.y-.46,z:.18};
    } else if(type==='ilove' && (i===1||i===2)){
      p1={x:base.x+d[0]*.10,y:base.y-.20,z:.08};p2={x:base.x+d[0]*.18,y:base.y-.35,z:.12};p3={x:base.x+d[0]*.14,y:base.y-.47,z:.15};
    } else if(type==='pinch' && i===0){
      p1={x:.05,y:.35,z:.18};p2={x:.34,y:.48,z:.20};p3={x:.50,y:.40,z:.20};
    } else if(type==='thumbUp' && i!==0){
      p1={x:base.x+d[0]*.06,y:base.y-.22,z:.10};p2={x:base.x+d[0]*.12,y:base.y-.38,z:.14};p3={x:base.x+d[0]*.08,y:base.y-.48,z:.16};
    } else if(type==='thumbDown' && i!==0){
      p1={x:base.x+d[0]*.05,y:base.y-.10,z:.10};p2={x:base.x+d[0]*.10,y:base.y-.18,z:.14};p3={x:base.x+d[0]*.05,y:base.y-.26,z:.16};
    }
    pts.push(base,p1,p2,p3);
  }
  // thumb: index 0 is wrist, then thumb chain 1-4 in this custom pose representation
  let thumb=[{x:-.78,y:-.55,z:.12},{x:-1.10,y:-.22,z:.15},{x:-1.22,y:.18,z:.18},{x:-.98,y:.50,z:.20}];
  if(type==='thumbUp') thumb=[{x:-.78,y:-.55,z:.12},{x:-.92,y:-.05,z:.16},{x:-.90,y:.48,z:.20},{x:-.82,y:.90,z:.24}];
  if(type==='thumbDown') thumb=[{x:-.78,y:-.55,z:.12},{x:-.92,y:-.72,z:.16},{x:-.86,y:-1.10,z:.20},{x:-.80,y:-1.48,z:.24}];
  if(type==='pinch') thumb=[{x:-.78,y:-.55,z:.12},{x:-.50,y:-.18,z:.17},{x:-.12,y:.28,z:.20},{x:.50,y:.40,z:.20}];
  if(type==='fist') thumb=[{x:-.72,y:-.45,z:.20},{x:-.35,y:-.30,z:.25},{x:-.05,y:-.48,z:.27},{x:.18,y:-.62,z:.25}];
  pts.push(...thumb);
  const ordered=[pts[0], ...thumb, ...pts.slice(1)];
  // Return standard 21-ish landmarks used only for visualization.
  return {wrist,thumb, fingers:[0,1,2,3].map(i=>[mcp[i],pts[5+i*4+1],pts[5+i*4+2],pts[5+i*4+3]])};
}
function poseTypeForSign(label){
  if(['MABUTI'].includes(label))return 'thumbUp';
  if(['MASAMA'].includes(label))return 'thumbDown';
  if(['IKAW','SAAN','KAILAN'].includes(label))return 'point';
  if(['MAHAL KITA'].includes(label))return 'ilove';
  if(['OO','PAANO'].includes(label))return 'fist';
  if(['HINDI','KAIN','INOM'].includes(label))return 'pinch';
  if(['TULONG','TAYO','DAGDAG','PAKIUSAP','SAKIT','ANO'].includes(label))return 'two';
  if(['SALAMAT'].includes(label))return 'side';
  if(['TIGIL'].includes(label))return 'open';
  return 'open';
}
function v3(x,y,z){return new THREE.Vector3(x,y,z)}
function makeBone(a,b,r1=.095,r2=.075,material){
  const start=v3(a.x,a.y,a.z),end=v3(b.x,b.y,b.z),mid=start.clone().add(end).multiplyScalar(.5);let len=start.distanceTo(end)||.01;
  const geo=new THREE.CylinderGeometry(r2,r1,len,14,1);
  const mesh=new THREE.Mesh(geo,material);mesh.position.copy(mid);mesh.quaternion.setFromUnitVectors(new THREE.Vector3(0,1,0),end.clone().sub(start).normalize());return mesh;
}
function addHandModel(group,type='open',offsetX=0){
  const material=new THREE.MeshStandardMaterial({color:0xe5a27b,roughness:.72,metalness:.03});
  const jointMat=new THREE.MeshStandardMaterial({color:0xf0b58f,roughness:.7});
  const palm=new THREE.Mesh(new THREE.SphereGeometry(1,28,20),material);palm.scale.set(1.05,1.42,.38);palm.position.set(offsetX,-.18,0);group.add(palm);
  const wrist=new THREE.Mesh(new THREE.CylinderGeometry(.38,.48,1.05,24),material);wrist.position.set(offsetX,-1.48,0);group.add(wrist);
  const pose=handPose(type);
  const addChain=(points)=>{for(let i=0;i<points.length-1;i++){const a=points[i],b=points[i+1];const bone=makeBone(a,b,.12,.095,material);bone.position.x+=offsetX;group.add(bone);const j=new THREE.Mesh(new THREE.SphereGeometry(.13,14,10),jointMat);j.position.set(b.x+offsetX,b.y,b.z);group.add(j)}};
  addChain(pose.thumb);pose.fingers.forEach(addChain);
  const nailMat=new THREE.MeshStandardMaterial({color:0xffe7da,roughness:.5});
  pose.fingers.forEach(chain=>{const p=chain[3],n=new THREE.Mesh(new THREE.SphereGeometry(.10,12,8),nailMat);n.position.set(p.x+offsetX,p.y,p.z+.035);n.scale.set(.9,.65,.35);group.add(n)});
  return group;
}
function initPractice3D(label){
  const host=$('practice3D');if(!host||typeof THREE==='undefined')return;
  if(practice3DState){host.innerHTML='';practice3DState.renderer.dispose();practice3DState=null;}
  const scene=new THREE.Scene();scene.background=new THREE.Color(0xeaf4ff);
  const camera=new THREE.PerspectiveCamera(28,Math.max(1,host.clientWidth)/Math.max(1,host.clientHeight),.1,100);camera.position.set(0,0,8.5);
  const renderer=new THREE.WebGLRenderer({antialias:true,alpha:true});renderer.setPixelRatio(Math.min(2,window.devicePixelRatio||1));renderer.setSize(Math.max(1,host.clientWidth),Math.max(1,host.clientHeight));renderer.outputColorSpace=THREE.SRGBColorSpace;host.appendChild(renderer.domElement);
  scene.add(new THREE.HemisphereLight(0xffffff,0x8fb7d9,2.2));const key=new THREE.DirectionalLight(0xffffff,2.2);key.position.set(3,4,6);scene.add(key);const fill=new THREE.DirectionalLight(0x9dc8ff,1.1);fill.position.set(-4,1,2);scene.add(fill);
  const root=new THREE.Group();scene.add(root);
  const type=poseTypeForSign(label);
  if(type==='two'){addHandModel(root,'open',-.9);const g2=new THREE.Group();addHandModel(g2,'open',0);g2.scale.setScalar(.92);g2.rotation.y=Math.PI*.22;root.add(g2);}else addHandModel(root,type,0);
  root.rotation.y=.12;root.rotation.x=-.05;
  let dragging=false,lastX=0,lastY=0,zoom=8.5;
  const pointerDown=e=>{dragging=true;lastX=e.clientX;lastY=e.clientY;host.setPointerCapture?.(e.pointerId)};
  const pointerMove=e=>{if(!dragging)return;root.rotation.y+=(e.clientX-lastX)*.012;root.rotation.x+=(e.clientY-lastY)*.009;root.rotation.x=clamp(root.rotation.x,-.9,.9);lastX=e.clientX;lastY=e.clientY};
  const pointerUp=()=>{dragging=false};
  host.onpointerdown=pointerDown;host.onpointermove=pointerMove;host.onpointerup=pointerUp;host.onpointerleave=pointerUp;host.onwheel=e=>{e.preventDefault();zoom=clamp(zoom+e.deltaY*.008,5.2,12);camera.position.z=zoom};
  const reset=()=>{root.rotation.set(-.05,.12,0);zoom=8.5;camera.position.set(0,0,zoom)};
  $('reset3DBtn')?.addEventListener('click',reset,{once:true});
  const resize=()=>{if(!practice3DState)return;const w=Math.max(1,host.clientWidth),h=Math.max(1,host.clientHeight);camera.aspect=w/h;camera.updateProjectionMatrix();renderer.setSize(w,h)};
  const tick=()=>{if(!practice3DState)return;renderer.render(scene,camera);requestAnimationFrame(tick)};
  practice3DState={renderer,scene,camera,root,resize};resize();tick();
}
function openPractice(label,shouldScroll=true){
  activePracticeSign=label;const panel=$('practicePanel');if(!panel)return;
  panel.hidden=false;$('practiceTitle').textContent=`Practice: ${label}`;$('practiceSignName').textContent=label;$('practiceSignMeaning').textContent=SIGN_TEXT[label]?.fil||label;$('practiceTip').textContent=PRACTICE_GUIDES[label]||'Follow the 3D model and hold the gesture clearly.';$('instructionTwo').textContent=PRACTICE_GUIDES[label]||'Place your hand in front of the camera and hold the gesture clearly.';
  document.querySelectorAll('.practice-tab').forEach(b=>b.classList.toggle('active',b.dataset.practiceTab==='model'));
  $('practiceModelWrap').hidden=false;$('practiceCameraWrap').hidden=true;$('practiceInstructions').hidden=true;$('practiceGoCameraBtn').hidden=false;
  initPractice3D(label);if(shouldScroll)panel.scrollIntoView({behavior:'smooth',block:'start'});
}
function closePractice(){const panel=$('practicePanel');if(panel)panel.hidden=true;stopPracticeCamera();}
function selectPracticeTab(tab){
  document.querySelectorAll('.practice-tab').forEach(b=>b.classList.toggle('active',b.dataset.practiceTab===tab));
  $('practiceModelWrap').hidden=tab!=='model';$('practiceCameraWrap').hidden=tab!=='camera';$('practiceInstructions').hidden=tab!=='instructions';$('practiceGoCameraBtn').hidden=tab!=='model';
  if(tab==='model')initPractice3D(activePracticeSign);if(tab==='camera')startPracticeCamera();
}
async function startPracticeCamera(){
  const msg=$('practiceCameraMessage');try{if(!window.isSecureContext)throw new Error('Camera practice requires HTTPS.');practiceCameraStream?.getTracks().forEach(t=>t.stop());practiceCameraStream=await navigator.mediaDevices.getUserMedia({video:{facingMode:{ideal:facingMode},width:{ideal:1280},height:{ideal:720}},audio:false});const v=$('practiceVideo');v.srcObject=practiceCameraStream;await v.play();msg.textContent=`Try ${activePracticeSign} while comparing your hand with the 3D guide.`;$('startPracticeCameraBtn').disabled=true;$('stopPracticeCameraBtn').disabled=false;}catch(e){msg.textContent=e.message||'Camera access is unavailable.';showToast(msg.textContent)}}
function stopPracticeCamera(){if(practiceCameraStream)practiceCameraStream.getTracks().forEach(t=>t.stop());practiceCameraStream=null;const v=$('practiceVideo');if(v)v.srcObject=null;const a=$('startPracticeCameraBtn'),b=$('stopPracticeCameraBtn');if(a)a.disabled=false;if(b)b.disabled=true}
function renderSignChart(){
  if(!signChart)return;
  const q=(signSearch?.value||'').trim().toLowerCase(),cat=signCategory?.value||'all';
  const items=Object.keys(SIGN_TEXT).filter(label=>{const text=(label+' '+SIGN_TEXT[label].fil+' '+SIGN_TEXT[label].en).toLowerCase();return(!q||text.includes(q))&&(cat==='all'||SIGN_CATEGORIES[label]===cat)});
  signChart.innerHTML=items.length?items.map(label=>`<button class="sign-card" data-sign="${label.replace(/"/g,'&quot;')}" aria-label="Practice ${label}">${renderSignPreview(label)}<span class="sign-card-name">${label}</span><span class="sign-card-meaning">${SIGN_TEXT[label].fil}</span><span class="sign-card-tip">${PRACTICE_GUIDES[label]||SIGN_TIPS[label]||'Supported prototype sign.'}</span><span class="practice-link">Practice in camera →</span></button>`).join(''):'<div class="empty-chart">No supported sign matches your search.</div>';
  signChart.querySelectorAll('[data-sign]').forEach(btn=>btn.addEventListener('click',()=>{sentenceTokens=[];normalizeSentence();const label=btn.dataset.sign;setState('READY','confirmed');detectedSign.textContent=label;translationText.textContent=meaning(label)+' Ready for camera practice.';setARFeedback(label,meaning(label));openPractice(label)}));
}

function setARFeedback(label='Waiting for a sign', text='Show a supported Filipino Sign Language gesture.'){if(arSign)arSign.textContent=label;if(arMeaning)arMeaning.textContent=text;if(arOverlay)arOverlay.classList.toggle('active',label!=='Waiting for a sign')}

function speechLanguage(){return voiceLanguage?.value==='en'?'en':'fil'}
function meaning(label,lang=speechLanguage()){return SIGN_TEXT[label]?.[lang]||label||''}
function sentenceText(lang='fil'){
  if(!sentenceTokens.length)return '';
  const compact=[]; for(const token of sentenceTokens){if(compact[compact.length-1]!==token)compact.push(token)}
  let text=compact.map(t=>(SIGN_TEXT[t]?.[lang]||t).replace(/[.!?]+$/,'')).join(' ').replace(/\s+/g,' ').trim();
  text=text.charAt(0).toUpperCase()+text.slice(1);
  if(!/[.!?]$/.test(text)) text += ['KUMUSTA','MAHAL KITA','SALAMAT'].includes(compact[compact.length-1])?'!':'.';
  return text;
}
function rawText(){return sentenceTokens.join(' | ')}
function normalizeSentence(){rawOutput.textContent=sentenceTokens.length?rawText():'Wala pang nakumpirmang senyas.';sentenceOutput.textContent=sentenceTokens.length?sentenceText('fil'):'Gumawa ng isang senyas, hawakan ito nang maayos, saka i-relax ang kamay bago ang susunod.'}
function addToSentence(label){if(sentenceTokens[sentenceTokens.length-1]===label)return false;sentenceTokens.push(label);normalizeSentence();if(autoSpeakToggle.checked)speakText(meaning(label),speechLanguage());return true}
function speakText(text,lang=speechLanguage()){if(!('speechSynthesis' in window)){showToast('Speech is not supported in this browser.');return}speechSynthesis.cancel();const u=new SpeechSynthesisUtterance(text);u.lang=lang==='en'?'en-PH':'fil-PH';u.rate=.92;u.pitch=1;speechSynthesis.speak(u)}
function commitRecognition(result,confidence){if(confidence>=82)learnExample(result.label,latestFeatureVectors);gestureLocked=true;lockedLabel=result.label;releaseFrames=0;candidateLabel='';candidateType='';candidateFrames=0;candidateScoreSum=0;addToSentence(result.label);setState('CONFIRMED','confirmed');detectedSign.textContent=`${result.label} ✓`;confidenceValue.textContent=`${confidence}%`;confidenceBar.style.width='100%';translationText.textContent=meaning(result.label)+' Awtomatikong binibigkas kapag naka-on ang voice output.';setARFeedback(result.label,meaning(result.label));typePill.textContent=result.model?'Adaptive AI + landmarks':result.twoHand?'Two-hand FSL sign':result.dynamic?'Motion FSL sign':'FSL sign';showToast(`${result.label} captured.`)}

function updateRecognition(result, hasHands){
  if(gestureLocked){
    const sameHeld=result&&result.label===lockedLabel;
    if(!result || !sameHeld)releaseFrames++; else releaseFrames=0;
    setState('LOCKED','locked');detectedSign.textContent=`${lockedLabel} ✓`;confidenceValue.textContent='100%';confidenceBar.style.width='100%';translationText.textContent=releaseFrames>=REQUIRED_RELEASE_FRAMES?'Ready for the next gesture.':'Gesture captured. Relax/change your hand.';typePill.textContent='Gesture complete';
    if(releaseFrames>=REQUIRED_RELEASE_FRAMES){gestureLocked=false;lockedLabel='';releaseFrames=0;setState('READY','confirmed');translationText.textContent='Ready for the next gesture.'}
    return;
  }
  if(!hasHands){candidateLabel='';candidateType='';candidateFrames=0;candidateScoreSum=0;setState('SEARCHING');detectedSign.textContent='Waiting...';confidenceValue.textContent='0%';confidenceBar.style.width='0%';translationText.textContent='Show a supported gesture.';typePill.textContent='Automatic';setARFeedback();return}
  if(!result){candidateLabel='';candidateType='';candidateFrames=0;candidateScoreSum=0;ambiguousFrames++;setState('DETECTED');detectedSign.textContent='Analyzing…';confidenceValue.textContent='0%';confidenceBar.style.width='0%';translationText.textContent='Hand detected. Adjust the gesture and hold it clearly.';typePill.textContent='Automatic';setARFeedback('Analyzing…','Hand detected. Hold the gesture clearly.');return}
  ambiguousFrames=0;
  if(result.label===candidateLabel&&result.type===candidateType){candidateFrames++;candidateScoreSum+=result.confidence}else{candidateLabel=result.label;candidateType=result.type;candidateFrames=1;candidateScoreSum=result.confidence}
  const progress=clamp(candidateFrames/REQUIRED_STABLE_FRAMES,0,1);const mean=Math.round(candidateScoreSum/candidateFrames);const verified=Math.round(mean*(.72+.28*progress));
  setState(candidateFrames===1?'DETECTED':'VERIFYING',candidateFrames>1?'verifying':'');detectedSign.textContent=result.label;confidenceValue.textContent=`${verified}%`;confidenceBar.style.width=`${Math.round(progress*100)}%`;translationText.textContent=`Hold steady… ${Math.round(progress*100)}% verified`;typePill.textContent=result.model?'Adaptive AI + landmarks':result.twoHand?'Two-hand sign':result.dynamic?'Motion sign':result.type;setARFeedback(result.label,meaning(result.label));
  if(candidateFrames>=REQUIRED_STABLE_FRAMES&&verified>=MIN_COMMIT_CONFIDENCE)commitRecognition(result,verified)
}

function onHandsResults(results){
  resizeCanvas();ctx.clearRect(0,0,canvas.width,canvas.height);
  const arr=results.multiHandLandmarks||[];const hasHands=arr.length>0;
  trackingBadge.classList.add('show');trackingBadge.classList.toggle('detected',hasHands);trackingBadge.textContent=hasHands?`HANDS DETECTED: ${arr.length}`:'HANDS: SEARCHING';
  if(!hasHands){handTrails=[[],[]];updateRecognition(null,false);return}
  latestFeatureVectors=arr.map(normalizedVector);
  const feats=arr.map((lm,i)=>{updateMotion(lm,i);return features(lm)});
  const twoHand=scoreTwoHand(feats);
  const singles=feats.map((f,i)=>scoreSingle(f,motionStats(i)));
  let best=twoHand||singles.filter(Boolean).sort((a,b)=>b.confidence-a.confidence)[0]||null;
  best=hybridize(best,latestFeatureVectors);
  arr.forEach((lm,i)=>{drawTrail(handTrails[i]||[]);const local=twoHand||singles[i];drawHand(lm,i,local?.label||'',local?.confidence||0)});
  updateRecognition(best,true);
}

async function initHandTracker(){if(hands)return true;if(typeof Hands==='undefined'){showToast('Recognition library did not load. Connect to the internet and refresh once.');throw new Error('MediaPipe Hands unavailable')}hands=new Hands({locateFile:file=>`${MP_BASE}/${file}`});hands.setOptions({maxNumHands:2,modelComplexity:1,minDetectionConfidence:.62,minTrackingConfidence:.62});hands.onResults(onHandsResults);if(typeof hands.initialize==='function')await hands.initialize();return true}
async function trackingLoop(){if(!stream)return;if(video.readyState>=2&&video.currentTime!==lastVideoTime&&!processingFrame&&hands){lastVideoTime=video.currentTime;processingFrame=true;try{await hands.send({image:video})}catch(e){console.error(e);trackingBadge.textContent='RECOGNITION ERROR'}finally{processingFrame=false}}trackingLoopId=requestAnimationFrame(trackingLoop)}
function cameraErrorMessage(e){if(!window.isSecureContext)return 'Camera requires HTTPS. Open the deployed Netlify URL.';if(e.name==='NotAllowedError')return 'Camera permission was denied. Allow camera access in browser settings.';if(e.name==='NotFoundError'||e.name==='OverconstrainedError')return 'Requested camera is unavailable on this device.';if(e.name==='NotReadableError')return 'Camera is busy in another app.';return `Camera error: ${e.message||e.name||'unknown error'}`}
async function startCamera(){try{if(!window.isSecureContext)throw new Error('HTTPS is required');if(!navigator.mediaDevices?.getUserMedia)throw new Error('Camera API unavailable');stopCamera(false);cameraStatus.classList.remove('online');cameraStatus.innerHTML='<span></span> Loading recognition…';await initHandTracker();const constraints={video:{facingMode:{ideal:facingMode},width:{ideal:1280},height:{ideal:720}},audio:false};stream=await navigator.mediaDevices.getUserMedia(constraints);video.srcObject=stream;await video.play();setMirror();resizeCanvas();video.classList.add('active');cameraPlaceholder.style.display='none';liveBadge.classList.add('show');trackingBadge.classList.add('show');cameraStatus.classList.add('online');cameraStatus.innerHTML='<span></span> AR camera on';startBtn.disabled=true;stopBtn.disabled=false;switchBtn.textContent=facingMode==='user'?'Use back camera':'Use front camera';lastVideoTime=-1;trackingLoopId=requestAnimationFrame(trackingLoop)}catch(e){console.error(e);stopCamera(false);const msg=cameraErrorMessage(e);showToast(msg);cameraStatus.classList.remove('online');cameraStatus.innerHTML='<span></span> Camera unavailable';translationText.textContent=msg}}
function stopCamera(update=true){if(trackingLoopId)cancelAnimationFrame(trackingLoopId);trackingLoopId=null;processingFrame=false;if(stream)stream.getTracks().forEach(t=>t.stop());stream=null;video.srcObject=null;video.classList.remove('active');cameraPlaceholder.style.display='flex';liveBadge.classList.remove('show');trackingBadge.classList.remove('show','detected');handTrails=[[],[]];ctx.clearRect(0,0,canvas.width,canvas.height);startBtn.disabled=false;stopBtn.disabled=true;gestureLocked=false;candidateLabel='';candidateFrames=0;if(update){cameraStatus.classList.remove('online');cameraStatus.innerHTML='<span></span> Camera off';resetResult()}}

navButtons.forEach(btn=>btn.addEventListener('click',()=>{const target=btn.dataset.target;navButtons.forEach(b=>b.classList.toggle('active',b===btn));pageSections.forEach(sec=>sec.classList.toggle('active-section',sec.id===target));if(target==='learnSection')renderSignChart()}));
signSearch?.addEventListener('input',renderSignChart);signCategory?.addEventListener('change',renderSignChart);
startBtn.addEventListener('click',startCamera);stopBtn.addEventListener('click',()=>stopCamera(true));
switchBtn.addEventListener('click',async()=>{facingMode=facingMode==='user'?'environment':'user';if(stream)await startCamera();else{setMirror();switchBtn.textContent=facingMode==='user'?'Use back camera':'Use front camera';showToast(`${facingMode==='user'?'Front':'Back'} camera selected.`)}});
fullscreenBtn.addEventListener('click',async()=>{try{if(!document.fullscreenElement)await cameraStage.requestFullscreen();else await document.exitFullscreen()}catch{showToast('Fullscreen is not available in this browser.')}});
deleteBtn.addEventListener('click',()=>{sentenceTokens.pop();normalizeSentence()});clearSentenceBtn.addEventListener('click',()=>{sentenceTokens=[];normalizeSentence()});speakBtn.addEventListener('click',()=>{const lang=speechLanguage(),text=sentenceText(lang);text?speakText(text,lang):showToast('No sentence to speak yet.')});voiceLanguage.addEventListener('change',()=>{translationText.textContent='Voice language: '+(speechLanguage()==='en'?'English':'Filipino')+'.';});
window.addEventListener('resize',resizeCanvas);window.addEventListener('orientationchange',()=>setTimeout(resizeCanvas,250));window.addEventListener('beforeunload',()=>stopCamera(false));
window.addEventListener('beforeinstallprompt',e=>{e.preventDefault();deferredInstallPrompt=e;installBtn.hidden=false});installBtn.addEventListener('click',async()=>{if(!deferredInstallPrompt)return;deferredInstallPrompt.prompt();await deferredInstallPrompt.userChoice;deferredInstallPrompt=null;installBtn.hidden=true});
if('serviceWorker' in navigator)window.addEventListener('load',()=>navigator.serviceWorker.register('./sw.js').catch(console.warn));
switchBtn.textContent='Use back camera';setMirror();normalizeSentence();resetResult();renderSignChart();
