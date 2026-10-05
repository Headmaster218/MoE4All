import { createRequire } from 'node:module'

const require = createRequire(new URL('../dsh/apps/web/package.json', import.meta.url))
const { chromium } = require('playwright')
const url = process.argv[2] ?? 'http://127.0.0.1:17373/'
const browser = await chromium.launch({ channel: 'msedge', headless: true, args: ['--disable-gpu'] })

try {
  const page = await browser.newPage({ viewport: { width: 1280, height: 800 } })
  await page.goto(url, { waitUntil: 'domcontentloaded' })
  await page.waitForTimeout(5_000)
  const start = page.getByRole('button', { name: '开始使用' })
  if (await start.count()) await start.click()
  const close = page.getByRole('button', { name: '关闭' })
  if (await close.count()) await close.click()
  await page.waitForTimeout(1_000)

  const inspect = () => page.evaluate(() => {
    const footers = [...document.querySelectorAll('[class*="footerActions"]')]
    const footer = footers.find(node => node.getBoundingClientRect().width > 0)
    if (!(footer instanceof HTMLElement)) throw new Error('Sidebar footer actions are missing')
    const phone = document.querySelector('button[aria-label="远程访问"]')
    const clock = document.querySelector('[title="定时任务"]')
      ?? [...document.querySelectorAll('*')].find(node => node.textContent?.trim() === '定时任务' && node.children.length === 0)
    const rectOf = (node) => {
      if (!node) return null
      const rect = node.getBoundingClientRect()
      return { x: rect.x, y: rect.y, width: rect.width, height: rect.height }
    }
    return {
      direction: getComputedStyle(footer).flexDirection,
      footer: rectOf(footer),
      clock: rectOf(clock),
      phone: rectOf(phone),
      phoneText: phone?.textContent?.trim() ?? '',
    }
  })

  const wide = await inspect()
  if (process.argv[3]) await page.screenshot({ path: process.argv[3] })
  await page.getByRole('button', { name: '收起侧边栏' }).click()
  await page.waitForTimeout(500)
  const collapsed = await inspect()
  if (process.argv[4]) await page.screenshot({ path: process.argv[4] })
  console.log(JSON.stringify({ wide, collapsed }, null, 2))
  for (const state of [wide, collapsed]) {
    if (state.direction !== 'column' || !state.clock || !state.phone
      || state.phone.y < state.clock.y + state.clock.height) process.exitCode = 1
  }
  if (Math.abs(collapsed.clock.x + collapsed.clock.width / 2 - collapsed.phone.x - collapsed.phone.width / 2) > 3) {
    process.exitCode = 1
  }
  if (wide.phoneText !== '远程访问' || collapsed.phoneText !== '') process.exitCode = 1
} finally {
  await browser.close()
}
