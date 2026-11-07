const fs = require('node:fs');
const path = require('node:path');
const root = path.resolve('target/concurrency-20260923');
const number = (line, key) => Number(line.match(new RegExp(`(?:^| )${key}=([0-9.]+)`))?.[1]);
const stamp = row => Date.parse(row.line.slice(0, 27)) || row.at;
const round = x => Number.isFinite(x) ? +x.toFixed(3) : null;
const fields = line => Object.fromEntries([...line.matchAll(/(?:^| )(\w+)=([0-9.]+)/g)].map(m => [m[1], +m[2]]));

function tokensAt(points, at) {
  if (!points.length) return 0;
  if (at <= points[0].at) return points[0].n;
  for (let i = 1; i < points.length; i++) {
    if (at <= points[i].at) {
      const a = points[i - 1], b = points[i];
      return a.n + (b.n - a.n) * (at - a.at) / Math.max(1, b.at - a.at);
    }
  }
  return points.at(-1).n;
}
const rate = (points, a, b) => b > a ? (tokensAt(points, b) - tokensAt(points, a)) * 1000 / (b - a) : null;
const slices = (points, a, b) => [0, 1, 2].map(i => round(rate(points, a + (b - a) * i / 3, a + (b - a) * (i + 1) / 3)));

for (const label of process.argv.slice(2)) {
  const data = JSON.parse(fs.readFileSync(path.join(root, label + '.json'), 'utf8'));
  const summary = { label, mode: data.mode, profile: data.profile, exeSha256: data.exeSha256, cases: [] };
  for (const c of data.cases) {
    const lines = data.lines.filter(x => x.case === c.name);
    const records = c.requests.map(r => {
      const final = lines.find(x => /request done/.test(x.line) && number(x.line, 'prompt_tokens') === r.usage?.prompt_tokens);
      const req = final ? number(final.line, 'req') : null;
      let points = req === null ? [] : lines.filter(x => /request progress/.test(x.line) && number(x.line, 'req') === req && /phase="decode"/.test(x.line)).map(x => ({ at: stamp(x), n: number(x.line, 'gen_tokens') }));
      const cycles = lines.filter(x => /qwen4 mtp cycle/.test(x.line) && stamp(x) >= r.sentAt && stamp(x) <= r.doneAt && (r.firstAt === null || stamp(x) >= r.firstAt - 1000));
      // A serialized second request's sentAt precedes the first one's completion. Its
      // first text boundary removes the other request's cycles from this interval.
      if (!points.length && cycles.length) {
        let n = 0;
        points = cycles.map(x => { n += number(x.line, 'committed'); return { at: stamp(x), n: Math.min(n, r.usage?.completion_tokens || n) }; });
      }
      if (final && points.length) points.push({ at: stamp(final), n: r.usage.completion_tokens });
      const begin = points[0]?.at ?? r.firstAt;
      const end = points.at(-1)?.at ?? r.doneAt;
      const gaps = r.events.slice(1).map((x, i) => ({ ms: x.at - r.events[i].at, at: r.events[i].at }));
      const longest = gaps.sort((a,b) => b.ms-a.ms)[0];
      const cycleFields=cycles.map(x=>fields(x.line));
      const total=key=>cycleFields.reduce((sum,x)=>sum+(x[key]||0),0);
      const mtp=cycles.length?{cycles:cycles.length,alpha:round(total('accepted')/total('drafted')),emittedPerCycle:round(r.usage?.completion_tokens/cycles.length),meanMs:Object.fromEntries(['draft','verify','catchup'].map(key=>[key,round(total(key)/cycles.length)]))}:null;
      return { lane: r.lane, req, prompt: r.usage?.prompt_tokens ?? r.plannedTokens, cached: r.timings?.cached_n, output: r.usage?.completion_tokens, ttftSec: r.firstAt===null?null:round((r.firstAt-r.sentAt)/1000), wallSec: round((r.doneAt-r.sentAt)/1000), decodeTps: round(r.timings?.predicted_per_second), prefillSec: round(r.timings?.prompt_ms/1000), thirdsTps: points.length > 1 ? slices(points, begin, end) : null, longestTextGapSec: round(longest?.ms/1000), longestTextGapStart: longest?.at, outputSha256: r.outputSha256, mtp, error: r.error, points, begin, end, sentAt:r.sentAt };
    });
    const start = Math.max(...records.map(x => x.begin));
    const end = Math.min(...records.map(x => x.end));
    const overlap = end > start && records.every(x => x.points.length > 1) ? { seconds: round((end-start)/1000), perLaneTps: records.map(x=>round(rate(x.points,start,end))), totalTps: round(records.reduce((sum,x)=>sum+rate(x.points,start,end),0)), thirdsTotalTps:[0,1,2].map(i=>round(records.reduce((sum,x)=>sum+rate(x.points,start+(end-start)*i/3,start+(end-start)*(i+1)/3),0))) } : null;
    const profiles = [];
    for (let i=0;i<lines.length;i++) {
      if (!lines[i].line.includes('[parallel-token-profile]')) continue;
      const item = { at:stamp(lines[i]), ...fields(lines[i].line) };
      const next = lines.slice(i+1,i+6);
      const pagerLine = next.find(x=>x.line.includes('[parallel-token-pager]'))?.line || '';
      item.pager = fields(pagerLine);
      const push = pagerLine.match(/push=([0-9.]+)MiB\/([0-9.]+)ms/);
      if (push) Object.assign(item.pager, { pushMiB:+push[1], pushMs:+push[2] });
      item.gap = fields(next.find(x=>x.line.includes('[parallel-token-gap]'))?.line || '');
      // Percentages are recomputed from totals, not summed across cohorts.
      for (const key of ['hit_rate', 'dma_hidden', 'push']) delete item.pager[key];
      delete item.gap.closure;
      profiles.push(item);
    }
    const byLanes = {};
    for(const p of profiles){
      const key = p.lanes;
      const bucket=byLanes[key] ||= { cohorts:0,steps:0,rows:0,wall:0,phase:{},pager:{},gap:{} };
      bucket.cohorts++; bucket.steps+=p.steps; bucket.rows+=p.decode_rows; bucket.wall+=p.wall;
      for(const key of ['once','front','layer0','ple_wait','ple_upload','main_setup','main_execute','tail','teardown','unaccounted']) bucket.phase[key]=(bucket.phase[key]||0)+(p[key]||0);
      for(const family of ['pager','gap']) for(const [k,v] of Object.entries(p[family])) bucket[family][k]=(bucket[family][k]||0)+v;
    }
    for(const b of Object.values(byLanes)) {
      b.rowRate=round(b.rows*1000/b.wall);
      b.hitRate=round(b.pager.gpu_hits/(b.pager.gpu_hits+b.pager.gpu_misses));
      b.dmaHiddenRate=round(b.pager.overlap/b.pager.gpu_dma);
      b.hostClosureRate=round(b.gap.host_accounted/b.gap.backend);
      for(const family of ['phase','pager','gap']) b[family+'PerStep']=Object.fromEntries(Object.entries(b[family]).map(([k,v])=>[k,round(v/b.steps)]));
    }
    const first=records[0], second=records[1];
    const live = lines.filter(x=>/serve stats/.test(x.line)).map(x=>({at:stamp(x),...fields(x.line)}));
    summary.cases.push({name:c.name,wallSec:round((c.doneAt-c.startAt)/1000),requests:records.map(({points,begin,end,sentAt,...x})=>x),overlap,soloBeforeSecondTps:first.points.length>1?round(rate(first.points,first.begin,Math.min(second.sentAt,first.end))):null,soloTailTps:records.map(x=>overlap&&x.end>end&&x.points.length>1?round(rate(x.points,end,x.end)):null),completedTokensPerWallSec:round(records.reduce((sum,x)=>sum+(x.output||0),0)*1000/(c.doneAt-c.startAt)),byLanes,zeroDecodeActiveWindows:live.filter(x=>x.active===2&&x.decode_tps===0).length});
  }
  fs.writeFileSync(path.join(root,label+'.analysis.json'),JSON.stringify(summary,null,2));
  console.log(JSON.stringify(summary,null,2));
}
