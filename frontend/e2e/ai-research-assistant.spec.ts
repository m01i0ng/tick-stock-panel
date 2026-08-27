import { expect, test, type Page, type Route } from '@playwright/test'

type SessionStatus =
  | 'awaiting_approval' | 'running' | 'paused' | 'stopping'
  | 'stopped' | 'completed' | 'failed' | 'interrupted'

type Session = ReturnType<typeof makeSession>

const proposal = {
  title: '量价互补研究计划',
  hypothesis: '动量与换手率可能提供互补信息',
  rationale: '比较两个目录内因子的样本外表现',
  factor_names: ['momentum_5d', 'turnover_rate'],
  strategy_ids: [],
  expected_outcome: '样本外 Sharpe 改善且回撤受控',
  risks: ['过拟合', '市场环境漂移'],
  digest: 'catalog-digest-1',
}

function makeSession(
  id: string,
  status: SessionStatus = 'awaiting_approval',
  title = proposal.title,
) {
  const completed = status === 'completed'
  const activeTrial = status === 'running' || status === 'paused' || status === 'stopping'
  const trial = completed || activeTrial ? {
    index: 1,
    proposal: { ...proposal, title },
    mining_run_id: 'run-1',
    status: completed ? 'completed' : 'running',
    started_at: '2026-08-26T01:00:00Z',
    finished_at: completed ? '2026-08-26T01:01:00Z' : null,
    evidence: completed ? {
      oos_sharpe: -1,
      oos_max_drawdown: -0.1118,
      oos_positive_fold_ratio: 0,
      oos_n_trades: 32,
      valid_folds: 1,
      qualified: false,
      gate_reasons: ['样本外 Sharpe 低于 0.5'],
    } : null,
    improved: completed,
    error: null,
  } : null
  return {
    session_id: id,
    status,
    goal: '研究量价因子稳定性',
    asset_type: 'stock',
    start: null,
    end: null,
    budget_profile: 'exploratory',
    commission_pct: 0.0002,
    stamp_tax_pct: 0.0005,
    slippage_bps: 5,
    correlation_threshold: 0.75,
    max_combination_factors: 4,
    beam_width: 12,
    max_finalists: 8,
    max_trials: 1,
    max_wall_minutes: 10,
    patience: 1,
    created_at: '2026-08-26T00:00:00Z',
    updated_at: '2026-08-26T01:01:00Z',
    started_at: status === 'awaiting_approval' ? null : '2026-08-26T01:00:00Z',
    finished_at: completed || status === 'stopped' ? '2026-08-26T01:01:00Z' : null,
    approved_at: status === 'awaiting_approval' ? null : '2026-08-26T01:00:00Z',
    current_trial: trial ? 1 : null,
    active_run_id: activeTrial ? 'run-1' : null,
    completed_trials: completed ? 1 : 0,
    no_improvement_trials: completed ? 1 : 0,
    data_snapshot_digest: trial ? 'snapshot-1' : null,
    adaptive_end: '2024-09-30',
    holdout: {
      status: 'reserved',
      start: '2024-10-08',
      end: '2025-01-01',
      bars: 63,
      mining_run_id: null,
      signature: null,
      sealed_at: null,
      sharpe: null,
      max_drawdown: null,
      n_trades: null,
      total_return: null,
      qualified: null,
      gate_reasons: null,
      error: null,
    },
    stop_reason: completed ? 'max_trials' : status === 'stopped' ? 'user_requested' : null,
    error: null,
    initial_proposal: { ...proposal, title },
    trials: trial ? [trial] : [],
    leaderboard: completed ? [{
      trial_index: 1,
      mining_run_id: 'run-1',
      proposal_digest: proposal.digest,
      title,
      oos_sharpe: -1,
      oos_max_drawdown: -0.1118,
      oos_positive_fold_ratio: 0,
      oos_n_trades: 32,
      qualified: false,
    }] : [],
  }
}

class ResearchApi {
  configured = true
  completeOnApprove = true
  preflightError = ''
  createdPayloads: Record<string, unknown>[] = []
  sessions: Session[] = []

  async install(page: Page) {
    await page.route('**/api/settings', route => json(route, { onboarding_completed: true }))
    await page.route('**/api/strategies/ai/status', route => json(route, {
      configured: this.configured,
      provider: 'test',
      model: 'test-model',
    }))
    await page.route('**/api/backtest/autoresearch/sessions**', route => this.handle(route))
  }

  private async handle(route: Route) {
    const request = route.request()
    const url = new URL(request.url())
    const path = url.pathname
    if (path.endsWith('/events')) {
      const id = path.split('/').at(-2)!
      const session = this.required(id)
      return route.fulfill({
        status: 200,
        contentType: 'text/event-stream',
        body: `event: snapshot\ndata: ${JSON.stringify(session)}\n\n`,
      })
    }
    if (path === '/api/backtest/autoresearch/sessions') {
      if (request.method() === 'GET') return json(route, { items: this.sessions })
      const payload = request.postDataJSON() as Record<string, unknown>
      this.createdPayloads.push(payload)
      if (this.preflightError) {
        return json(route, { detail: this.preflightError }, 400, {
          'X-Mining-Preflight-Code': 'enriched_insufficient',
        })
      }
      const session = makeSession(`session-${this.sessions.length + 1}`)
      Object.assign(session, payload)
      this.sessions.unshift(session)
      return json(route, session)
    }
    const match = path.match(/\/sessions\/([^/]+)(?:\/(approve|pause|resume|stop|seal))?$/)
    if (!match) return json(route, { detail: 'not found' }, 404)
    const session = this.required(decodeURIComponent(match[1]))
    const action = match[2]
    if (!action) return json(route, session)
    if (action === 'approve') {
      Object.assign(session, makeSession(
        session.session_id,
        this.completeOnApprove ? 'completed' : 'running',
        session.initial_proposal.title,
      ))
    } else if (action === 'pause') {
      session.status = 'paused'
    } else if (action === 'resume') {
      session.status = 'running'
    } else if (action === 'seal') {
      session.holdout = {
        ...(session.holdout || {}),
        status: 'sealed',
        mining_run_id: 'run-1',
        signature: 'sig-1',
        sealed_at: '2026-08-26T01:02:00Z',
        sharpe: -0.4,
        max_drawdown: -0.2,
        n_trades: 20,
        total_return: -0.05,
        qualified: false,
        gate_reasons: ['requires a holdout Sharpe of at least 0.5'],
        error: null,
      }
    } else {
      session.status = 'stopped'
      session.active_run_id = null
      session.stop_reason = 'user_requested'
      if (session.trials[0]) session.trials[0].status = 'stopped'
    }
    return json(route, session)
  }

  private required(id: string) {
    const session = this.sessions.find(item => item.session_id === id)
    if (!session) throw new Error(`missing mock session: ${id}`)
    return session
  }
}

function json(
  route: Route,
  body: unknown,
  status = 200,
  headers: Record<string, string> = {},
) {
  return route.fulfill({
    status,
    contentType: 'application/json',
    headers: { 'Cache-Control': 'no-store', ...headers },
    body: JSON.stringify(body),
  })
}

test('creates, approves and renders a completed Catalog research session', async ({ page }) => {
  const api = new ResearchApi()
  await api.install(page)
  await page.goto('/mining?mode=auto')

  await expect(page.getByRole('button', { name: '自动研究' })).toHaveAttribute('aria-current', 'page')
  await page.getByRole('button', { name: '新建会话' }).click()
  await expect(page.getByRole('heading', { name: 'Catalog 实验编排' })).toBeVisible()

  await page.getByRole('button', { name: '生成研究计划' }).click()
  await expect(page.getByText('请至少用 4 个字描述研究问题')).toBeVisible()
  expect(api.createdPayloads).toHaveLength(0)

  await page.getByLabel('研究问题').fill('研究量价因子稳定性')
  await page.getByLabel('验证档位').selectOption('exploratory')
  await page.getByLabel('最多 Trial').fill('1')
  await page.getByLabel('最多分钟').fill('10')
  await page.getByLabel('无改善停止').fill('1')
  await page.getByRole('button', { name: '生成研究计划' }).click()

  const assistant = page.getByLabel('Catalog 实验编排')
  await expect(assistant.getByText(proposal.title)).toBeVisible()
  await expect(assistant.getByText('momentum_5d')).toBeVisible()
  await assistant.getByRole('button', { name: '批准并开始' }).click()

  await expect(page).toHaveURL(/mode=auto.*session=session-1/)
  await expect(page.getByText('已完成', { exact: true }).last()).toBeVisible()
  await expect(page.getByText('Trial 1')).toBeVisible()
  await expect(page.getByText('未通过研究门槛')).toBeVisible()
  await expect(page.getByText('Leaderboard')).toBeVisible()
  await expect(page.getByText('停止原因：max_trials')).toBeVisible()
  await expect(page.getByText(/留出集 2024-10-08 ~ 2025-01-01/)).toBeVisible()
  await page.getByRole('button', { name: '密封最终留出集' }).click()
  await expect(page.getByText('密封最终留出集未通过, 只能保存待定候选。')).toBeVisible()
  await expect(page.getByRole('link', { name: '查看完整挖掘结果' })).toHaveAttribute(
    'href',
    '/mining?mode=manual&run=run-1',
  )
  expect(api.createdPayloads[0]).toMatchObject({
    goal: '研究量价因子稳定性',
    budget_profile: 'exploratory',
    commission_pct: 0.0002,
    stamp_tax_pct: 0.0005,
    slippage_bps: 5,
    max_trials: 1,
    max_wall_minutes: 10,
    patience: 1,
  })
})

test('restores an active session and controls pause, resume, stop and history selection', async ({ page }) => {
  const api = new ResearchApi()
  api.completeOnApprove = false
  api.sessions = [
    makeSession('session-active', 'running', '正在运行的研究'),
    makeSession('session-history', 'completed', '历史研究'),
  ]
  await api.install(page)
  await page.addInitScript(() => localStorage.setItem('autoresearch_active_session_id', 'session-active'))
  await page.goto('/mining?mode=auto')

  await expect(page).toHaveURL(/session=session-active/)
  await expect(page.getByText('正在运行的研究').first()).toBeVisible()
  await page.getByRole('button', { name: '本轮后暂停' }).click()
  await expect(page.getByRole('button', { name: '恢复' })).toBeVisible()

  await page.reload()
  await expect(page.getByRole('button', { name: '恢复' })).toBeVisible()
  await page.getByRole('button', { name: '恢复' }).click()
  await expect(page.getByRole('button', { name: '本轮后暂停' })).toBeVisible()

  page.once('dialog', dialog => dialog.accept())
  await page.getByRole('button', { name: '停止', exact: true }).click()
  await expect(page.getByText('停止原因：user_requested')).toBeVisible()
  await expect.poll(() => page.evaluate(() => localStorage.getItem('autoresearch_active_session_id'))).toBeNull()

  await page.getByRole('button', { name: '历史研究' }).click()
  await expect(page).toHaveURL(/session=session-history/)
  await expect(page.getByText('Trial 1')).toBeVisible()
})

test('shows a preflight failure without creating a session', async ({ page }) => {
  const api = new ResearchApi()
  api.preflightError = 'balanced mining requires at least 786 enriched trading bars; effective range has 243'
  await api.install(page)
  await page.goto('/mining?mode=auto')
  await page.getByRole('button', { name: '新建会话' }).click()
  await page.getByLabel('研究问题').fill('研究均衡档数据是否充足')
  await page.getByRole('button', { name: '生成研究计划' }).click()
  await expect(page.getByLabel('Catalog 实验编排').getByText(api.preflightError)).toBeVisible()
  await expect(page.getByRole('button', { name: '生成研究计划' })).toBeEnabled()
  expect(api.sessions).toHaveLength(0)
})

test('routes an unconfigured user to AI settings', async ({ page }) => {
  const api = new ResearchApi()
  api.configured = false
  await api.install(page)
  await page.goto('/mining?mode=auto')
  await page.getByRole('button', { name: '新建会话' }).click()
  await expect(page.getByText('AI 尚未配置，无法创建 Catalog 实验。')).toBeVisible()
  await expect(page.getByRole('button', { name: '前往 AI 设置' })).toBeVisible()
})
