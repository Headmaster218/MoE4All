const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const { spawn } = require('node:child_process');
const { pathToFileURL } = require('node:url');

if (!process.env.TRACE_ANALYSIS_ROOT) throw new Error('TRACE_ANALYSIS_ROOT must name the new chart');
const root = path.resolve(process.env.TRACE_ANALYSIS_ROOT);
const gpuOnly = process.env.GPU_ONLY_TRACE === '1';
const profile = path.join(root, `edge-cdp-${Date.now()}`);
const child = spawn('C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe', [
  '--headless=new', '--disable-gpu', '--disable-extensions', '--disable-background-networking',
  '--no-first-run', '--no-default-browser-check', '--remote-debugging-port=0',
  `--user-data-dir=${profile}`, 'about:blank',
], { windowsHide: true, stdio: 'ignore' });
const closed = new Promise(resolve => child.once('exit', resolve));
let socket, nextId = 1;
const pending = new Map();
const delay = ms => new Promise(resolve => setTimeout(resolve, ms));
function send(method, params = {}) {
  const id = nextId++;
  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => { pending.delete(id); reject(new Error(`CDP timeout: ${method}`)); }, 15000);
    pending.set(id, { resolve, reject, timer });
    socket.send(JSON.stringify({ id, method, params }));
  });
}
async function evaluate(expression) {
  const result = await send('Runtime.evaluate', { expression, awaitPromise: true, returnByValue: true });
  assert.equal(result.exceptionDetails, undefined);
  return result.result.value;
}

async function checkGpuOnly(name, width, height) {
  const metrics = await evaluate(`window.timelineReady.then(api => ({
    width: innerWidth, documentWidth: document.documentElement.scrollWidth,
    cases: api.getCases().length, steps: document.getElementById('step').options.length,
    rows: api.getRows().length, misses: document.getElementById('misses').rows.length,
    boundaries: document.getElementById('boundaries').rows.length,
    canvasWidth: document.getElementById('timeline').clientWidth
  }))`);
  assert.equal(metrics.width, width);
  assert.ok(metrics.documentWidth <= width, 'GPU-only page overflow');
  assert.equal(metrics.cases, 2); assert.equal(metrics.steps, 12); assert.equal(metrics.rows, 6);
  assert.equal(metrics.boundaries, 3);
  assert.ok(metrics.misses > 0 && metrics.canvasWidth > 200);
  const colored = await evaluate(`(() => {
    const c = document.getElementById('timeline'), p = c.getContext('2d').getImageData(0,0,c.width,c.height).data;
    let count = 0; for (let i=0;i<p.length;i+=4) if (p[i+3] && Math.max(p[i],p[i+1],p[i+2])-Math.min(p[i],p[i+1],p[i+2])>20) count++;
    return count;
  })()`);
  assert.ok(colored > 1000, 'GPU-only blank canvas');
  await evaluate("document.getElementById('timeline').scrollIntoView({block:'center'})");
  const original = await evaluate('window.timelineReady.then(api=>api.getViewport())');
  const at = await evaluate(`(() => { const r=document.getElementById('timeline').getBoundingClientRect(); return {x:r.left+110+(r.width-124)/2,y:r.top+70};})()`);
  await send('Input.dispatchMouseEvent', { type: 'mouseWheel', ...at, deltaX: 0, deltaY: -400 });
  await delay(100);
  const zoomed = await evaluate('window.timelineReady.then(api=>api.getViewport())');
  assert.ok(zoomed.span < original.span && zoomed.start > 0);
  await send('Input.dispatchMouseEvent', { type: 'mousePressed', ...at, button:'left',buttons:1,clickCount:1 });
  await send('Input.dispatchMouseEvent', { type:'mouseMoved',x:at.x-30,y:at.y,button:'left',buttons:1 });
  await send('Input.dispatchMouseEvent', { type:'mouseReleased',x:at.x-30,y:at.y,button:'left',buttons:0,clickCount:1 });
  const dragged = await evaluate('window.timelineReady.then(api=>api.getViewport())');
  assert.ok(dragged.start > zoomed.start);
  await evaluate("document.getElementById('reset').click()");
  for (let i=0;i<2;i++) {
    const rows = await evaluate(`window.timelineReady.then(api=> { document.getElementById('case').value=${i};api.choose();return api.getRows(); })`);
    const firstLayer = rows[2].some(e=>e.layer===0) ? 0 : 1;
    assert.deepEqual(rows[2].map(e=>e.layer).sort((a,b)=>a-b), Array.from({length:48-firstLayer},(_,j)=>j+firstLayer));
    assert.ok(rows[4].length > 0);
  }
  const capture = await send('Page.captureScreenshot', {format:'png',captureBeyondViewport:false});
  fs.writeFileSync(path.join(root,`timeline-${name}.png`),Buffer.from(capture.data,'base64'));
  return {name,...metrics,colored,original,zoomed,dragged};
}

(async () => {
  try {
    const portFile = path.join(profile, 'DevToolsActivePort');
    for (let i = 0; !fs.existsSync(portFile) && i < 150; i++) await delay(100);
    const port = fs.readFileSync(portFile, 'utf8').split(/\r?\n/)[0];
    const targets = await (await fetch(`http://127.0.0.1:${port}/json/list`)).json();
    const target = targets.find(value => value.type === 'page');
    assert.ok(target);
    socket = new WebSocket(target.webSocketDebuggerUrl);
    socket.addEventListener('message', event => {
      const message = JSON.parse(event.data);
      const waiter = pending.get(message.id);
      if (!waiter) return;
      pending.delete(message.id);
      clearTimeout(waiter.timer);
      if (message.error) waiter.reject(new Error(JSON.stringify(message.error)));
      else waiter.resolve(message.result);
    });
    await new Promise((resolve, reject) => {
      socket.addEventListener('open', resolve, { once: true });
      socket.addEventListener('error', reject, { once: true });
    });
    await send('Page.enable');
    const results = [];
    for (const [name, width, height] of [['desktop', 1440, 1000], ['mobile', 390, 1000]]) {
      await send('Emulation.setDeviceMetricsOverride', { width, height, deviceScaleFactor: 1, mobile: name === 'mobile' });
      await send('Page.navigate', { url: pathToFileURL(path.join(root, 'timeline.html')).href });
      let ready = false;
      for (let i = 0; !ready && i < 100; i++) {
        await delay(100);
        ready = await evaluate('window.timelineReady ? window.timelineReady.then(() => true) : false');
      }
      assert.ok(ready, 'timeline did not render');
      if (gpuOnly) {
        results.push(await checkGpuOnly(name, width, height));
        continue;
      }
      const metrics = await evaluate(`window.timelineReady.then(api => {
        document.getElementById('case').value = 2; api.choose();
        return { width: innerWidth, documentWidth: document.documentElement.scrollWidth,
          cases: document.getElementById('case').options.length,
          canvasWidth: document.getElementById('timeline').clientWidth,
          layers: api.getLayers().length,
          gpuLayers: api.getGpuLayers().length,
          summary: document.getElementById('detail').textContent };
      })`);
      assert.equal(metrics.width, width);
      assert.ok(metrics.documentWidth <= width, `page overflow: ${JSON.stringify(metrics)}`);
      assert.equal(metrics.cases, 4);
      assert.ok(metrics.canvasWidth > 200);
      assert.equal(metrics.layers, 48);
      assert.equal(metrics.gpuLayers, 48);
      const pixels = await evaluate(`['timeline', 'histogram'].map(id => {
        const canvas = document.getElementById(id);
        const pixels = canvas.getContext('2d').getImageData(0, 0, canvas.width, canvas.height).data;
        let colored = 0;
        for (let i = 0; i < pixels.length; i += 4) {
          if (pixels[i + 3] > 0 && Math.max(pixels[i], pixels[i + 1], pixels[i + 2])
            - Math.min(pixels[i], pixels[i + 1], pixels[i + 2]) > 20) colored++;
        }
        return { id, colored };
      })`);
      assert.ok(pixels.every(p => p.colored > 1000), 'blank chart');
      await evaluate("document.getElementById('timeline').scrollIntoView({block:'center'})");
      const initialView = await evaluate('window.timelineReady.then(api => api.getViewport())');
      const plot = await evaluate(`(() => { const r = document.getElementById('timeline').getBoundingClientRect();
        return { x: r.left + 125 + (r.width - 145) / 2, y: r.top + 80 }; })()`);
      await send('Input.dispatchMouseEvent', { type: 'mouseWheel', x: plot.x, y: plot.y, deltaX: 0, deltaY: -400 });
      await delay(150);
      const zoomed = await evaluate('window.timelineReady.then(api => api.getViewport())');
      assert.ok(zoomed.span < initialView.span && zoomed.start > 0, 'real wheel zoom failed');
      await send('Input.dispatchMouseEvent', { type: 'mousePressed', x: plot.x, y: plot.y, button: 'left', buttons: 1, clickCount: 1 });
      await send('Input.dispatchMouseEvent', { type: 'mouseMoved', x: plot.x - 45, y: plot.y, button: 'left', buttons: 1 });
      await send('Input.dispatchMouseEvent', { type: 'mouseReleased', x: plot.x - 45, y: plot.y, button: 'left', buttons: 0, clickCount: 1 });
      const dragged = await evaluate('window.timelineReady.then(api => api.getViewport())');
      assert.ok(dragged.start > zoomed.start, 'real pointer drag failed');
      await evaluate("document.getElementById('reset').click()");
      const resetView = await evaluate('window.timelineReady.then(api => api.getViewport())');
      assert.ok(Math.abs(resetView.span - initialView.span) < 1e-8 && resetView.start === 0);
      const layerPoint = await evaluate(`window.timelineReady.then(api => {
        const l = api.getLayers()[0], r = document.getElementById('timeline').getBoundingClientRect();
        return { x: r.left + 125 + (l.start + l.end) / 2 / api.getViewport().span * (r.width - 145), y: r.top + 65 };
      })`);
      await send('Input.dispatchMouseEvent', { type: 'mouseMoved', ...layerPoint });
      const layerHover = await evaluate("document.getElementById('hover').textContent");
      assert.ok(layerHover.includes('L0') && layerHover.includes('非 GPU'), 'layer hover missing');
      const gpuHover = await evaluate(`window.timelineReady.then(api => {
        const l = api.getGpuLayers()[5], el = document.getElementById('timeline'), r = el.getBoundingClientRect();
        el.dispatchEvent(new PointerEvent('pointermove', { clientX: r.left + 125 +
          (l.start + l.end) / 2 / api.getViewport().span * (r.width - 145), clientY: r.top + 255 }));
        return document.getElementById('hover').textContent;
      })`);
      assert.ok(gpuHover.includes('L5') && gpuHover.includes('非精确测量') && gpuHover.includes('定位范围'), 'GPU estimate hover missing');
      await evaluate("document.querySelector('details').open = true");
      assert.ok(await evaluate('document.documentElement.scrollWidth <= innerWidth'), 'definitions overflow on mobile');
      await evaluate("document.querySelector('details').open = false");
      for (let index = 0; index < 4; index++) {
        const counts = await evaluate(`window.timelineReady.then(api => {
          document.getElementById('case').value = ${index}; api.choose();
          return { layers: api.getLayers().length, gpuLayers: api.getGpuLayers().length,
            steps: document.getElementById('step').options.length };
        })`);
        assert.equal(counts.layers, index % 2 === 0 ? 48 : 0);
        assert.equal(counts.gpuLayers, index % 2 === 0 ? 48 : 0);
        assert.equal(counts.steps, 12);
      }
      const capture = await send('Page.captureScreenshot', { format: 'png', captureBeyondViewport: false });
      const bytes = Buffer.from(capture.data, 'base64');
      assert.equal(bytes.readUInt32BE(16), width);
      assert.equal(bytes.readUInt32BE(20), height);
      assert.ok(bytes.length > 25000);
      fs.writeFileSync(path.join(root, `timeline-${name}.png`), bytes);
      results.push({ name, ...metrics, pixels, initialView, zoomed, dragged, resetView, layerHover, gpuHover, screenshotBytes: bytes.length });
    }
    fs.writeFileSync(path.join(root, 'browser-check.json'), JSON.stringify(results, null, 2));
    console.log(JSON.stringify(results));
  } finally {
    if (socket?.readyState === WebSocket.OPEN) {
      try { await send('Browser.close'); } catch (_) { /* Browser close may retire the websocket first. */ }
      socket.close();
    }
    if (await Promise.race([closed.then(() => true), delay(5000).then(() => false)]) === false) child.kill();
  }
})().catch(error => { console.error(error); process.exitCode = 1; });
