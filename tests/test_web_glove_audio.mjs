// Test the exact controller shipped in the inline collection module.
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import vm from 'node:vm';
const html=readFileSync(new URL('../src/dexstation/server/web/index.html',import.meta.url),'utf8');
const source=html.slice(html.indexOf('function createGloveAudio('),html.indexOf('const gloveAudio=createGloveAudio'));
assert.ok(source.includes('function createGloveAudio'),'Missing audio transition controller');
const create=vm.runInNewContext(source+';createGloveAudio');
const calls=[],store=new Map();
class AC{constructor(){this.destination={};this.currentTime=0;}resume(){return Promise.resolve();}createOscillator(){return {frequency:{set value(v){calls.push(['pitch',v]);}},connect(){},start(){calls.push(['beep']);},stop(){}};}createGain(){return {gain:{},connect(){}};}}
const host={AudioContext:AC,SpeechSynthesisUtterance:class{constructor(text){this.text=text;}},localStorage:{getItem:k=>store.get(k),setItem:(k,v)=>store.set(k,v)},speechSynthesis:{getVoices:()=>[{lang:'en-US'}],cancel(){},speak:u=>calls.push(['speak',u.text])}};
const passive=create(host);passive.update({id:'passive',phase:'countdown',remaining_s:5});assert.equal(calls.length,0,'SSE must not trigger an autoplay permission bar');
const audio=create(host);assert.equal(audio.enabled,true);audio.unlock();
for(const remaining_s of [5,5,4,4,3,2,1])audio.update({id:'a',phase:'countdown',remaining_s});
assert.equal(calls.filter(c=>c[0]==='beep').length,5);
for(const step of [1,1,2,2,3])audio.update({id:'a',phase:'recording',step,prompt:step===1?'Open: hold':step===2?'Fist: repeat':'Pinch index: repeat'});
assert.deepEqual(calls.filter(c=>c[0]==='speak').map(c=>c[1]),['Open','Fist','Pinch index']);
assert.ok(calls.some(c=>c[0]==='pitch'&&c[1]===1046));
audio.update({id:'a',phase:'completed'});audio.update({id:'a',phase:'completed'});
assert.equal(calls.at(-1)[1],'Done');
audio.setEnabled(false);const n=calls.length;audio.update({id:'b',phase:'countdown',remaining_s:5});assert.equal(calls.length,n);assert.equal(create(host).enabled,false);
audio.setEnabled(true);host.speechSynthesis.getVoices=()=>[];
audio.update({id:'b',phase:'recording',step:2,prompt:'Fist: repeat'});assert.equal(calls.at(-1)[0],'beep');
const fallbackBeepCount=calls.filter(c=>c[0]==='beep').length;audio.update({id:'c',phase:'recording',step:1,prompt:'Open: hold'});
assert.equal(calls.filter(c=>c[0]==='beep').length,fallbackBeepCount+1);assert.equal(calls.filter(c=>c[0]==='pitch').at(-1)[1],1046);
const unsupported=create({localStorage:host.localStorage});assert.doesNotThrow(()=>unsupported.update({id:'c',phase:'recording',step:1,prompt:'Open'}));
console.log('glove audio transitions: passed');
