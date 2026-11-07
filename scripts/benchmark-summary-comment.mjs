// Used only by the trusted pull_request_target/manual workflow, never PR code.
export const MARKER = '<!-- moe4all-benchmark-review:v1 -->';
const sha = value => typeof value === 'string' && /^[a-f0-9]{40}$/.test(value);
const number = value => /^\d+$/.test(String(value)) && Number.isSafeInteger(Number(value)) && Number(value) > 0;

export async function prepare({ github, context, core, prNumber }) {
  const value = context.eventName === 'workflow_dispatch' ? prNumber : context.payload.pull_request?.number;
  if (!number(value)) throw new Error('Expected a positive PR number');
  const pullNumber = Number(value);
  const { data: pull } = await github.rest.pulls.get({ ...context.repo, pull_number: pullNumber });
  if (pull.base?.repo?.full_name !== `${context.repo.owner}/${context.repo.repo}` || pull.base.ref !== 'main') {
    core.setOutput('enabled', 'false');
    return;
  }
  // An old queued event must not render or overwrite the summary of a newer head.
  if (context.eventName !== 'workflow_dispatch' && (pull.state !== 'open' || pull.head.sha !== context.payload.pull_request.head.sha)) {
    core.setOutput('enabled', 'false');
    return;
  }
  let base = pull.base.sha;
  if (pull.state !== 'open') {
    if (!pull.merged || !sha(pull.merge_commit_sha)) throw new Error('Only open or merged PRs can be summarized');
    const { data: merge } = await github.rest.git.getCommit({ ...context.repo, commit_sha: pull.merge_commit_sha });
    base = merge.parents[0]?.sha;
  }
  const { data: main } = await github.rest.git.getRef({ ...context.repo, ref: 'heads/main' });
  const baseline = main.object.sha;
  if (![base, pull.head.sha, baseline].every(sha)) throw new Error('Expected immutable Git commit IDs');
  for (const [key, value] of Object.entries({ enabled: 'true', number: pullNumber, base, head: pull.head.sha, baseline })) core.setOutput(key, String(value));
}

export async function publish({ github, context, core, result, prNumber, expectedHead, write = false }) {
  if (!number(prNumber) || !sha(expectedHead) || result.head_sha !== expectedHead) throw new Error('Summary identity mismatch');
  const skipped = result.status === 'skipped';
  if (!skipped && (!['ready', 'invalid'].includes(result.status) || typeof result.body !== 'string' || !result.body.startsWith(`${MARKER}\n`) || Buffer.byteLength(result.body, 'utf8') > 60000)) throw new Error('Invalid summary output');
  const pullNumber = Number(prNumber);
  const args = { ...context.repo, pull_number: pullNumber };
  const fresh = async () => {
    const { data: pull } = await github.rest.pulls.get(args);
    return pull.head.sha === expectedHead && pull.base?.repo?.full_name === `${context.repo.owner}/${context.repo.repo}` && pull.base.ref === 'main' && (pull.state === 'open' || context.eventName === 'workflow_dispatch' && pull.merged);
  };
  if (!await fresh()) {
    core.info('PR changed or closed; skipping stale summary');
    return 'stale';
  }
  if (!skipped) await core.summary.addRaw(result.body).write();
  if (!write) return skipped ? 'skipped' : 'preview';
  const comments = await github.paginate(github.rest.issues.listComments, { ...context.repo, issue_number: pullNumber, per_page: 100 });
  const previous = comments.find(comment => comment.user?.type === 'Bot' && comment.user.login === 'github-actions[bot]' && comment.body?.startsWith(`${MARKER}\n`));
  if (skipped && !previous) return 'skipped';
  if (!await fresh()) return 'stale';
  if (previous) {
    const body = skipped ? `${MARKER}\n本次提交不再包含独立 Benchmark 报告，旧摘要已撤下。\n\n检查 \`${expectedHead.slice(0, 12)}\`\n` : result.body;
    await github.rest.issues.updateComment({ ...context.repo, comment_id: previous.id, body });
    return 'updated';
  }
  await github.rest.issues.createComment({ ...context.repo, issue_number: pullNumber, body: result.body });
  return 'created';
}
