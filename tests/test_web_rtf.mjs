// Exercise the exact status renderer shipped in the collection page.
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import vm from 'node:vm';
const html=readFileSync(new URL('../src/dexstation/server/web/index.html',import.meta.url),'utf8');
const source=html.slice(html.indexOf('function showRTF('),html.indexOf('function showMotion('));
const element={textContent:'',style:{}};
const show=vm.runInNewContext(source+';showRTF',{$:id=>{assert.equal(id,'rtf');return element;}});
for(const [value,text,color] of [[1.02,'1.02',''],[.98,'0.98',''],[.979,'0.98','#ffbf47'],[.9,'0.90','#ffbf47'],[.899,'0.90','#ef5350'],[.8,'0.80','#ef5350'],[null,'—',''],[undefined,'—','']]){
 show({realtime_metrics:{rtf:value}});
 assert.equal(element.textContent,text);
 assert.equal(element.style.color,color);
}
show({});assert.equal(element.textContent,'—');assert.equal(element.style.color,'');
console.log('RTF number and threshold colors: passed');
