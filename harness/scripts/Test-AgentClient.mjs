import { createRequire } from 'node:module'

const require = createRequire(new URL('../dsh/apps/web/package.json', import.meta.url))
const { chromium } = require('playwright')
const url = process.argv[2] ?? 'http://127.0.0.1:17373/'
const browser = await chromium.launch({ channel: 'msedge', headless: true, args: ['--disable-gpu'] })

try {
  const page = await browser.newPage()
  const errors = []
  page.on('pageerror', error => errors.push(error.message))
  page.on('console', message => {
    if (message.type() === 'error') errors.push(message.text())
  })
  const response = await page.goto(url, { waitUntil: 'domcontentloaded', timeout: 30_000 })
  await page.waitForTimeout(8_000)
  const body = (await page.locator('body').innerText()).slice(0, 1200)
  const failed = errors.length > 0 || /Failed to load plugins|failed to import loader entry|require\("[^\"]+"\) missed the module table/i.test(body)
  console.log(JSON.stringify({ status: response?.status(), title: await page.title(), body, errors }, null, 2))
  if (response?.status() !== 200 || failed) process.exitCode = 1
} finally {
  await browser.close()
}
