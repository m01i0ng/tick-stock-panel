import { useEffect, useRef, useState } from 'react'
import { LoaderCircle, Play, Settings, Sparkles, X } from 'lucide-react'
import { useNavigate } from 'react-router-dom'
import { Modal } from '@/components/Modal'
import { api, type AutoresearchSession, type MiningBudgetProfile } from '@/lib/api'
import { approveAutoresearch, startAutoresearch, useAutoresearchTask } from '@/lib/autoresearchTask'

interface Props {
  onClose: () => void
  onCreated: (sessionId: string) => void
}

const INPUT = 'h-8 w-full rounded-input border border-border bg-base px-2 text-xs text-foreground outline-none focus:border-accent'

export function ResearchAssistantDialog({ onClose, onCreated }: Props) {
  const navigate = useNavigate()
  const task = useAutoresearchTask()
  const inputRef = useRef<HTMLTextAreaElement>(null)
  const [goal, setGoal] = useState('')
  const [assetType, setAssetType] = useState<'stock' | 'etf'>('stock')
  const [profile, setProfile] = useState<MiningBudgetProfile>('balanced')
  const [start, setStart] = useState('')
  const [end, setEnd] = useState('')
  const [commissionBps, setCommissionBps] = useState('2')
  const [stampTaxBps, setStampTaxBps] = useState('5')
  const [slippageBps, setSlippageBps] = useState('5')
  const [maxTrials, setMaxTrials] = useState('5')
  const [maxWallMinutes, setMaxWallMinutes] = useState('60')
  const [patience, setPatience] = useState('3')
  const [configured, setConfigured] = useState<boolean | null>(null)
  const [pending, setPending] = useState(false)
  const [error, setError] = useState('')
  const [result, setResult] = useState<AutoresearchSession | null>(null)

  useEffect(() => {
    api.strategyAiStatus()
      .then(value => setConfigured(value.configured))
      .catch(() => setConfigured(false))
  }, [])

  const generate = async () => {
    const value = goal.trim()
    if (value.length < 4) {
      setError('请至少用 4 个字描述研究问题')
      return
    }
    if (start && end && start > end) {
      setError('开始日期不能晚于结束日期')
      return
    }
    const costs = [commissionBps, stampTaxBps, slippageBps].map(Number)
    if (costs.some(item => !Number.isFinite(item) || item < 0) || costs[0] > 500 || costs[1] > 500 || costs[2] > 1000) {
      setError('佣金和印花税需为 0–500bp，滑点需为 0–1000bp')
      return
    }
    const budgets = [maxTrials, maxWallMinutes, patience].map(Number)
    if (!Number.isInteger(budgets[0]) || budgets[0] < 1 || budgets[0] > 10
      || !Number.isInteger(budgets[1]) || budgets[1] < 1 || budgets[1] > 360
      || !Number.isInteger(budgets[2]) || budgets[2] < 1 || budgets[2] > budgets[0]) {
      setError('Trial 数需为 1–10，运行分钟需为 1–360，耐心轮数不能超过 Trial 数')
      return
    }
    setPending(true)
    setError('')
    setResult(null)
    try {
      const session = await startAutoresearch({
        goal: value,
        asset_type: assetType,
        budget_profile: profile,
        start: start || null,
        end: end || null,
        commission_pct: costs[0] / 10000,
        stamp_tax_pct: costs[1] / 10000,
        slippage_bps: costs[2],
        max_trials: budgets[0],
        max_wall_minutes: budgets[1],
        patience: budgets[2],
      })
      setResult(session)
      onCreated(session.session_id)
    } catch (reason) {
      setError(String((reason as Error).message || reason))
    } finally {
      setPending(false)
    }
  }

  const approve = async () => {
    if (!result || task.commandPending) return
    const session = await approveAutoresearch()
    if (!session) return
    setResult(session)
    onCreated(session.session_id)
    onClose()
  }

  return (
    <Modal
      onClose={onClose}
      labelledBy="research-assistant-title"
      initialFocusRef={inputRef}
      closeOnBackdrop={!pending}
      panelClassName="flex max-h-[88vh] w-[94vw] max-w-2xl flex-col overflow-hidden rounded-card border border-border bg-surface shadow-2xl"
    >
      <header className="flex items-center gap-3 border-b border-border px-4 py-3">
        <div className="grid h-8 w-8 shrink-0 place-items-center rounded-btn bg-accent/10 text-accent"><Sparkles className="h-4 w-4" /></div>
        <div className="min-w-0 flex-1">
          <h2 id="research-assistant-title" className="text-sm font-semibold text-foreground">Catalog 实验编排</h2>
          <p className="mt-0.5 text-[10px] text-muted">从现有因子和策略目录生成有预算的连续试验，不会自动发布。不是选股页的策略代码生成器。</p>
        </div>
        <button type="button" aria-label="关闭" disabled={pending} onClick={onClose} className="grid h-8 w-8 place-items-center rounded-btn text-muted hover:bg-elevated hover:text-foreground disabled:opacity-40"><X className="h-4 w-4" /></button>
      </header>

      <div className="min-h-0 flex-1 overflow-y-auto p-4">
        {configured === false ? (
          <div className="rounded-card border border-warning/30 bg-warning/5 p-4 text-xs text-warning">
            <div>AI 尚未配置，无法创建 Catalog 实验。</div>
            <button type="button" onClick={() => { onClose(); navigate('/settings?tab=ai') }} className="mt-3 inline-flex h-8 items-center gap-1.5 rounded-btn border border-warning/40 px-3 font-medium hover:bg-warning/10"><Settings className="h-3.5 w-3.5" />前往 AI 设置</button>
          </div>
        ) : (
          <>
            <label className="block">
              <span className="mb-1 block text-[10px] font-medium text-secondary">研究问题</span>
              <textarea
                ref={inputRef}
                value={goal}
                onChange={event => setGoal(event.target.value)}
                disabled={pending}
                maxLength={2000}
                rows={4}
                placeholder="例如：研究低波动与20日动量是否互补，并用趋势突破策略作为对照"
                className="w-full resize-y rounded-input border border-border bg-base px-3 py-2 text-xs leading-5 text-foreground outline-none focus:border-accent disabled:opacity-60"
              />
            </label>
            <div className="mt-3 grid grid-cols-2 gap-3">
              <label><span className="mb-1 block text-[10px] font-medium text-secondary">资产</span><select className={INPUT} value={assetType} disabled={pending} onChange={event => setAssetType(event.target.value as 'stock' | 'etf')}><option value="stock">股票</option><option value="etf">ETF</option></select></label>
              <label><span className="mb-1 block text-[10px] font-medium text-secondary">验证档位</span><select className={INPUT} value={profile} disabled={pending} onChange={event => setProfile(event.target.value as MiningBudgetProfile)}><option value="exploratory">探索</option><option value="balanced">均衡</option><option value="strict">严格</option></select></label>
              <label><span className="mb-1 block text-[10px] font-medium text-secondary">开始日期（可选）</span><input type="date" className={INPUT} value={start} disabled={pending} onChange={event => setStart(event.target.value)} /></label>
              <label><span className="mb-1 block text-[10px] font-medium text-secondary">结束日期（可选）</span><input type="date" className={INPUT} value={end} disabled={pending} onChange={event => setEnd(event.target.value)} /></label>
            </div>
            <div className="mt-3 grid grid-cols-3 gap-3">
              <label><span className="mb-1 block text-[10px] font-medium text-secondary">佣金 bp</span><input inputMode="decimal" className={INPUT} value={commissionBps} disabled={pending} onChange={event => setCommissionBps(event.target.value)} /></label>
              <label><span className="mb-1 block text-[10px] font-medium text-secondary">印花税 bp</span><input inputMode="decimal" className={INPUT} value={stampTaxBps} disabled={pending} onChange={event => setStampTaxBps(event.target.value)} /></label>
              <label><span className="mb-1 block text-[10px] font-medium text-secondary">滑点 bp</span><input inputMode="decimal" className={INPUT} value={slippageBps} disabled={pending} onChange={event => setSlippageBps(event.target.value)} /></label>
            </div>

            <div className="mt-3 grid grid-cols-3 gap-3">
              <label><span className="mb-1 block text-[10px] font-medium text-secondary">最多 Trial</span><input inputMode="numeric" className={INPUT} value={maxTrials} disabled={pending} onChange={event => setMaxTrials(event.target.value)} /></label>
              <label><span className="mb-1 block text-[10px] font-medium text-secondary">最多分钟</span><input inputMode="numeric" className={INPUT} value={maxWallMinutes} disabled={pending} onChange={event => setMaxWallMinutes(event.target.value)} /></label>
              <label><span className="mb-1 block text-[10px] font-medium text-secondary">无改善停止</span><input inputMode="numeric" className={INPUT} value={patience} disabled={pending} onChange={event => setPatience(event.target.value)} /></label>
            </div>

            <button type="button" disabled={pending || configured !== true || !!result} onClick={() => void generate()} className="mt-4 inline-flex h-9 w-full items-center justify-center gap-2 rounded-btn bg-accent px-4 text-xs font-semibold text-white disabled:cursor-not-allowed disabled:opacity-50">
              {pending ? <LoaderCircle className="h-4 w-4 animate-spin" /> : <Sparkles className="h-4 w-4" />}
              {pending ? '正在创建研究会话' : '生成研究计划'}
            </button>

            {(error || task.error) && <div className="mt-3 whitespace-pre-wrap rounded-btn border border-danger/30 bg-danger/5 px-3 py-2 text-[10px] leading-5 text-danger">{error || task.error}</div>}

            {result && (
              <section className="mt-4 overflow-hidden rounded-card border border-border">
                <div className="border-b border-border bg-base/50 px-3 py-2">
                  <div className="text-xs font-semibold text-foreground">{result.initial_proposal?.title || '研究计划'}</div>
                  <div className="mt-1 text-[10px] leading-5 text-secondary">{result.initial_proposal?.hypothesis}</div>
                </div>
                <div className="space-y-3 p-3 text-[10px] leading-5">
                  <div><span className="font-medium text-secondary">研究理由：</span><span className="text-muted">{result.initial_proposal?.rationale}</span></div>
                  <div><span className="font-medium text-secondary">支持或否定条件：</span><span className="text-muted">{result.initial_proposal?.expected_outcome}</span></div>
                  <div><div className="mb-1 font-medium text-secondary">因子</div><div className="flex flex-wrap gap-1">{result.initial_proposal?.factor_names.map(id => <span key={id} className="rounded-full bg-accent/10 px-2 py-0.5 font-mono text-accent">{id}</span>)}</div></div>
                  {!!result.initial_proposal?.strategy_ids.length && <div><div className="mb-1 font-medium text-secondary">对照策略</div><div className="flex flex-wrap gap-1">{result.initial_proposal.strategy_ids.map(id => <span key={id} className="rounded-full bg-elevated px-2 py-0.5 font-mono text-secondary">{id}</span>)}</div></div>}
                  {!!result.initial_proposal?.risks.length && <div><div className="mb-1 font-medium text-secondary">主要风险</div><ul className="list-disc pl-4 text-muted">{result.initial_proposal.risks.map(risk => <li key={risk}>{risk}</li>)}</ul></div>}
                </div>
              </section>
            )}
          </>
        )}
      </div>

      {result && configured !== false && (
        <footer className="flex items-center justify-between gap-3 border-t border-border bg-base/40 px-4 py-3">
          <span className="text-[9px] text-muted">批准后才会执行 Trial；可在工作台暂停或停止。</span>
          <button type="button" disabled={!!task.commandPending || result.status !== 'awaiting_approval'} onClick={() => void approve()} className="inline-flex h-8 shrink-0 items-center gap-1.5 rounded-btn bg-accent px-3 text-xs font-semibold text-white disabled:opacity-50"><Play className="h-3.5 w-3.5" />{task.commandPending === 'approve' ? '正在批准' : '批准并开始'}</button>
        </footer>
      )}
    </Modal>
  )
}
