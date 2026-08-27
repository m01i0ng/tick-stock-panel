import { useEffect, useMemo } from 'react'
import { useQuery, useQueryClient } from '@tanstack/react-query'
import { Link, useSearchParams } from 'react-router-dom'
import {
  AlertTriangle, CheckCircle2, CirclePause, CirclePlay, Clock3, FlaskConical,
  LoaderCircle, Lock, Pause, Play, Plus, RefreshCw, Square, Trophy,
} from 'lucide-react'
import { EmptyState } from '@/components/EmptyState'
import { api, type AutoresearchSession, type AutoresearchSessionStatus, type AutoresearchTrial } from '@/lib/api'
import {
  approveAutoresearch, attachAutoresearchSession, pauseAutoresearch,
  resumeAutoresearch, sealAutoresearch, stopAutoresearch, tryReconnectAutoresearch, useAutoresearchTask,
} from '@/lib/autoresearchTask'
import { QK } from '@/lib/queryKeys'

const TERMINAL = new Set<AutoresearchSessionStatus>(['completed', 'failed', 'stopped', 'interrupted'])
const STATUS_LABELS: Record<AutoresearchSessionStatus, string> = {
  awaiting_approval: '等待批准',
  running: '研究中',
  paused: '本轮完成后暂停',
  stopping: '正在停止',
  stopped: '已停止',
  completed: '已完成',
  failed: '失败',
  interrupted: '启动中断',
}
const TRIAL_LABELS: Record<string, string> = {
  running: '运行中', completed: '已评估', failed: '失败', stopped: '已停止',
}

function holdoutNotice(session: AutoresearchSession) {
  const status = session.holdout?.status
  if (status === 'sealing') return '正在评估密封最终留出集…'
  if (status === 'sealed' && session.holdout?.qualified) return '密封最终留出集已通过; 发布以留出集指标为准。'
  if (status === 'sealed') return '密封最终留出集未通过, 只能保存待定候选。'
  if (status === 'failed') return '密封最终留出集失败, 可在修复数据后重试。'
  return '当前排名来自反复使用的自适应验证, 不是密封最终留出集; 完成后需密封留出集才能发布。'
}

function formatNumber(value: number | null | undefined, digits = 2) {
  return typeof value === 'number' && Number.isFinite(value) ? value.toFixed(digits) : '—'
}
function formatPct(value: number | null | undefined, digits = 1) {
  return typeof value === 'number' && Number.isFinite(value) ? `${(value * 100).toFixed(digits)}%` : '—'
}
function statusIcon(status: AutoresearchSessionStatus) {
  if (status === 'running' || status === 'stopping') return LoaderCircle
  if (status === 'completed') return CheckCircle2
  if (status === 'paused') return CirclePause
  if (status === 'awaiting_approval') return Clock3
  return AlertTriangle
}

function SessionRail({
  sessions,
  selectedId,
  active,
  loading,
  error,
  refreshing,
  onSelect,
  onNew,
  onRefresh,
}: {
  sessions: AutoresearchSession[]
  selectedId: string
  active: boolean
  loading: boolean
  error: boolean
  refreshing: boolean
  onSelect: (id: string) => void
  onNew: () => void
  onRefresh: () => void
}) {
  return (
    <aside className="border-b border-border bg-base/25 xl:border-b-0 xl:border-r">
      <div className="flex items-center gap-2 border-b border-border px-3 py-2">
        <div className="min-w-0 flex-1">
          <div className="text-xs font-semibold text-foreground">研究会话</div>
          <div className="mt-0.5 text-[9px] text-muted">一次仅运行一个 Catalog 会话</div>
        </div>
        <button type="button" title="刷新会话" onClick={onRefresh} className="grid h-7 w-7 place-items-center rounded-btn text-muted hover:bg-elevated hover:text-accent"><RefreshCw className={`h-3.5 w-3.5 ${refreshing ? 'animate-spin' : ''}`} /></button>
      </div>
      <div className="p-3">
        <button type="button" disabled={active} onClick={onNew} title={active ? '请先完成或停止当前会话' : '新建研究会话'} className="inline-flex h-8 w-full items-center justify-center gap-1.5 rounded-btn bg-accent text-xs font-semibold text-white disabled:cursor-not-allowed disabled:opacity-40"><Plus className="h-3.5 w-3.5" />新建会话</button>
      </div>
      <div className="max-h-72 overflow-y-auto border-t border-border xl:max-h-[calc(100vh-16rem)]">
        {loading && <div className="px-3 py-5 text-center text-[10px] text-muted">加载会话…</div>}
        {!loading && error && <div className="px-3 py-5 text-center text-[10px] text-danger">会话列表加载失败，请刷新重试</div>}
        {!loading && !error && !sessions.length && <div className="px-3 py-5 text-center text-[10px] text-muted">暂无研究会话</div>}
        {sessions.map(session => {
          const Icon = statusIcon(session.status)
          return (
            <button key={session.session_id} type="button" onClick={() => onSelect(session.session_id)} className={`block w-full border-b border-border/60 px-3 py-2.5 text-left hover:bg-elevated/60 ${selectedId === session.session_id ? 'bg-accent/10' : ''}`}>
              <div className="flex items-center gap-2">
                <Icon className={`h-3.5 w-3.5 shrink-0 ${session.status === 'running' || session.status === 'stopping' ? 'animate-spin text-accent' : session.status === 'completed' ? 'text-success' : 'text-muted'}`} />
                <span className="min-w-0 flex-1 truncate text-[11px] font-medium text-foreground">{session.initial_proposal?.title || session.goal}</span>
              </div>
              <div className="mt-1 flex items-center justify-between gap-2 pl-5 text-[9px] text-muted">
                <span>{STATUS_LABELS[session.status]}</span>
                <span className="font-mono">{session.completed_trials}/{session.max_trials}</span>
              </div>
            </button>
          )
        })}
      </div>
    </aside>
  )
}

function TrialTimeline({ trials, selectedIndex, onSelect }: { trials: AutoresearchTrial[]; selectedIndex: number | null; onSelect: (index: number) => void }) {
  return (
    <section className="border-b border-border">
      <div className="flex items-center justify-between px-3 py-2">
        <h2 className="text-xs font-semibold text-foreground">Trial 时间线</h2>
        <span className="text-[9px] text-muted">每轮仅调整一个研究方向</span>
      </div>
      <div className="overflow-x-auto border-t border-border">
        <div className="flex min-w-max gap-2 p-3">
          {trials.map(trial => (
            <button key={trial.index} type="button" onClick={() => onSelect(trial.index)} className={`w-48 shrink-0 rounded-card border p-2 text-left ${selectedIndex === trial.index ? 'border-accent bg-accent/10' : 'border-border bg-base/30 hover:border-accent/40'}`}>
              <div className="flex items-center justify-between gap-2">
                <span className="font-mono text-[9px] text-muted">Trial {trial.index}</span>
                <span className={`text-[9px] ${trial.status === 'completed' ? 'text-success' : trial.status === 'failed' ? 'text-danger' : trial.status === 'running' ? 'text-accent' : 'text-muted'}`}>{TRIAL_LABELS[trial.status] || trial.status}</span>
              </div>
              <div className="mt-1.5 truncate text-[10px] font-medium text-foreground" title={trial.proposal.title}>{trial.proposal.title}</div>
              <div className="mt-1 truncate text-[9px] text-muted" title={trial.proposal.hypothesis}>{trial.proposal.hypothesis}</div>
            </button>
          ))}
          {!trials.length && <div className="px-2 py-4 text-[10px] text-muted">批准计划后开始生成 Trial。</div>}
        </div>
      </div>
    </section>
  )
}

export function AutoresearchWorkbench({ onNewSession }: { onNewSession: () => void }) {
  const queryClient = useQueryClient()
  const [searchParams, setSearchParams] = useSearchParams()
  const task = useAutoresearchTask()
  const sessionId = searchParams.get('session') || ''
  const requestedTrial = Number(searchParams.get('trial'))
  const sessionsQuery = useQuery({
    queryKey: QK.autoresearchSessions,
    queryFn: api.autoresearchSessions,
    refetchInterval: 5000,
  })

  useEffect(() => {
    if (sessionId) {
      const creatingSession = task.isPending && task.sessionId === null
      if (!creatingSession && task.sessionId !== sessionId) void attachAutoresearchSession(sessionId)
      return
    }
    if (task.isPending && task.sessionId === null) return
    if (!tryReconnectAutoresearch()) return
  }, [sessionId, task.isPending, task.sessionId])

  useEffect(() => {
    if (!sessionId && task.sessionId) {
      const next = new URLSearchParams(searchParams)
      next.set('mode', 'auto')
      next.set('session', task.sessionId)
      setSearchParams(next, { replace: true })
    }
  }, [sessionId, searchParams, setSearchParams, task.sessionId])

  useEffect(() => {
    if (!task.session) return
    queryClient.setQueryData<{ items: AutoresearchSession[] }>(QK.autoresearchSessions, previous => {
      if (!previous) return { items: [task.session!] }
      return { items: [task.session!, ...previous.items.filter(item => item.session_id !== task.session!.session_id)] }
    })
    queryClient.setQueryData(QK.autoresearchSession(task.session.session_id), task.session)
  }, [queryClient, task.session])

  const sessions = sessionsQuery.data?.items ?? []
  const hasActiveSession = sessions.some(item => !TERMINAL.has(item.status))
  const session = task.sessionId === sessionId ? task.session : null
  const trials = useMemo(() => {
    if (!session) return []
    return session.trials ?? []
  }, [session])
  const defaultTrial = session?.current_trial ?? trials.at(-1)?.index ?? null
  const selectedIndex = Number.isInteger(requestedTrial) && trials.some(item => item.index === requestedTrial)
    ? requestedTrial
    : defaultTrial
  const selectedTrial = trials.find(item => item.index === selectedIndex) ?? null

  const selectSession = (id: string) => {
    if (id === sessionId) void attachAutoresearchSession(id)
    const next = new URLSearchParams(searchParams)
    next.set('mode', 'auto')
    next.set('session', id)
    next.delete('trial')
    setSearchParams(next, { replace: true })
  }
  const selectTrial = (index: number) => {
    const next = new URLSearchParams(searchParams)
    next.set('trial', String(index))
    setSearchParams(next, { replace: true })
  }
  const stop = () => {
    if (window.confirm('确认停止当前自动研究？已完成的 Trial 和结果会保留。')) void stopAutoresearch()
  }

  return (
    <div className="grid min-h-[calc(100vh-9rem)] grid-cols-1 overflow-hidden rounded-card border border-border bg-surface xl:grid-cols-[17rem_minmax(0,1fr)]">
      <SessionRail
        sessions={sessions}
        selectedId={sessionId}
        active={task.isPending || hasActiveSession}
        loading={sessionsQuery.isLoading}
        error={sessionsQuery.isError}
        refreshing={sessionsQuery.isFetching}
        onSelect={selectSession}
        onNew={onNewSession}
        onRefresh={() => void sessionsQuery.refetch()}
      />

      <main className="min-w-0 xl:max-h-[calc(100vh-9rem)] xl:overflow-y-auto">
        {!session && (task.reconnecting || task.isPending) && <div className="grid min-h-[32rem] place-items-center"><div className="flex items-center gap-2 text-xs text-muted"><LoaderCircle className="h-4 w-4 animate-spin" />读取研究会话…</div></div>}
        {!session && !task.isPending && !task.reconnecting && (
          <div className="min-h-[32rem]"><EmptyState icon={FlaskConical} title={task.error ? '研究会话加载失败' : '尚未选择研究会话'} hint={task.error || '新建会话，从当前因子和策略目录编排有预算的连续试验，不会自动发布。'} /></div>
        )}
        {session && (
          <>
            <header className="border-b border-border bg-base/20 px-3 py-2.5">
              <div className="flex flex-wrap items-start gap-3">
                <div className="min-w-0 flex-1">
                  <div className="flex items-center gap-2"><span className="truncate text-sm font-semibold text-foreground">{session.initial_proposal?.title || session.goal}</span><span className="shrink-0 rounded-full bg-elevated px-2 py-0.5 text-[9px] text-secondary">{STATUS_LABELS[session.status]}</span></div>
                  <div className="mt-1 text-[10px] leading-4 text-muted">{session.initial_proposal?.hypothesis || session.goal}</div>
                </div>
                <div className="flex shrink-0 items-center gap-1.5">
                  {session.status === 'awaiting_approval' && <button type="button" disabled={!!task.commandPending} onClick={() => void approveAutoresearch()} className="inline-flex h-8 items-center gap-1.5 rounded-btn bg-accent px-3 text-xs font-semibold text-white disabled:opacity-50"><Play className="h-3.5 w-3.5" />批准执行</button>}
                  {session.status === 'running' && <button type="button" disabled={!!task.commandPending} onClick={() => void pauseAutoresearch()} className="inline-flex h-8 items-center gap-1.5 rounded-btn border border-border px-2.5 text-xs text-secondary hover:border-accent/40 hover:text-accent disabled:opacity-50"><Pause className="h-3.5 w-3.5" />本轮后暂停</button>}
                  {session.status === 'paused' && <button type="button" disabled={!!task.commandPending} onClick={() => void resumeAutoresearch()} className="inline-flex h-8 items-center gap-1.5 rounded-btn bg-accent px-3 text-xs font-semibold text-white disabled:opacity-50"><CirclePlay className="h-3.5 w-3.5" />恢复</button>}
                  {TERMINAL.has(session.status) && session.holdout?.status === 'sealing' && <button type="button" disabled className="inline-flex h-8 items-center gap-1.5 rounded-btn bg-accent px-3 text-xs font-semibold text-white opacity-50"><LoaderCircle className="h-3.5 w-3.5 animate-spin" />密封中…</button>}
                  {TERMINAL.has(session.status) && (session.holdout?.status === 'reserved' || session.holdout?.status === 'failed') && session.leaderboard.length > 0 && <button type="button" disabled={!!task.commandPending} onClick={() => void sealAutoresearch()} className="inline-flex h-8 items-center gap-1.5 rounded-btn bg-accent px-3 text-xs font-semibold text-white disabled:opacity-50"><Lock className="h-3.5 w-3.5" />密封最终留出集</button>}
                  {!TERMINAL.has(session.status) && session.status !== 'stopping' && <button type="button" disabled={!!task.commandPending} onClick={stop} className="inline-flex h-8 items-center gap-1.5 rounded-btn border border-danger/40 px-2.5 text-xs text-danger hover:bg-danger/10 disabled:opacity-50"><Square className="h-3.5 w-3.5" />停止</button>}
                </div>
              </div>
              <div className="mt-2 grid grid-cols-2 gap-px overflow-hidden rounded-btn border border-border bg-border sm:grid-cols-4">
                {[
                  ['Trial 预算', `${session.completed_trials}/${session.max_trials}`],
                  ['时长预算', `${session.max_wall_minutes} 分钟`],
                  ['无改善', `${session.no_improvement_trials}/${session.patience}`],
                  ['验证档位', session.budget_profile],
                ].map(([label, value]) => <div key={label} className="bg-surface px-2.5 py-2"><div className="text-[9px] text-muted">{label}</div><div className="mt-0.5 font-mono text-[11px] font-semibold text-foreground">{value}</div></div>)}
              </div>
              {(session.holdout?.start || session.adaptive_end) && (
                <div className="mt-2 text-[9px] text-secondary">
                  自适应截止 {session.adaptive_end || '—'} · 留出集 {session.holdout?.start || '—'} ~ {session.holdout?.end || '—'}
                  {session.holdout?.bars != null ? ` (${session.holdout.bars} 根)` : ''}
                </div>
              )}
              <div className="mt-2 text-[9px] text-warning">{holdoutNotice(session)}</div>
              {session.holdout?.status === 'sealed' && (
                <div className="mt-2 grid grid-cols-2 gap-px overflow-hidden rounded-btn border border-border bg-border sm:grid-cols-4">
                  {[
                    ['留出集 Sharpe', formatNumber(session.holdout.sharpe)],
                    ['留出集回撤', formatPct(session.holdout.max_drawdown)],
                    ['留出集交易数', session.holdout.n_trades == null ? '—' : String(session.holdout.n_trades)],
                    ['留出集门槛', session.holdout.qualified ? '通过' : '未通过'],
                  ].map(([label, value]) => <div key={label} className="bg-surface px-2.5 py-2"><div className="text-[9px] text-muted">{label}</div><div className="mt-0.5 font-mono text-[11px] font-semibold text-foreground">{value}</div></div>)}
                </div>
              )}
              {!!session.holdout?.gate_reasons?.length && session.holdout.status === 'sealed' && !session.holdout.qualified && (
                <ul className="mt-2 list-disc pl-4 text-[9px] text-warning">{session.holdout.gate_reasons.map(reason => <li key={reason}>{reason}</li>)}</ul>
              )}
              {session.holdout?.status === 'failed' && session.holdout.error && <div className="mt-2 text-[9px] text-danger">{session.holdout.error}</div>}
              {(task.reconnecting || task.error || session.stop_reason) && <div className={`mt-2 text-[9px] ${task.error ? 'text-danger' : 'text-muted'}`}>{task.reconnecting ? '事件流已断开，正在轮询恢复…' : task.error || `停止原因：${session.stop_reason}`}</div>}
            </header>

            <TrialTimeline trials={trials} selectedIndex={selectedIndex} onSelect={selectTrial} />

            <div className="grid grid-cols-1 2xl:grid-cols-[minmax(0,1fr)_22rem]">
              <section className="min-w-0 border-b border-border 2xl:border-b-0 2xl:border-r">
                <h2 className="border-b border-border px-3 py-2 text-xs font-semibold text-foreground">Trial 详情</h2>
                {selectedTrial ? (
                  <div className="space-y-3 p-3 text-[10px] leading-5">
                    <div><span className="font-medium text-secondary">假设：</span><span className="text-muted">{selectedTrial.proposal.hypothesis}</span></div>
                    <div><span className="font-medium text-secondary">理由：</span><span className="text-muted">{selectedTrial.proposal.rationale}</span></div>
                    <div className="flex flex-wrap gap-1">{selectedTrial.proposal.factor_names.map(id => <span key={id} className="rounded-full bg-accent/10 px-2 py-0.5 font-mono text-accent">{id}</span>)}</div>
                    {selectedTrial.error && <div className="rounded-btn border border-danger/30 bg-danger/5 px-3 py-2 text-danger"><div className="mb-0.5 font-medium">Trial 失败</div>{selectedTrial.error}</div>}
                    {!!selectedTrial.evidence?.gate_reasons?.length && <div className="rounded-btn border border-warning/30 bg-warning/5 px-3 py-2 text-warning"><div className="mb-0.5 font-medium">未通过研究门槛</div><ul className="list-disc pl-4">{selectedTrial.evidence.gate_reasons.map(reason => <li key={reason}>{reason}</li>)}</ul></div>}
                    {selectedTrial.evidence && <div className="grid grid-cols-2 gap-px overflow-hidden rounded-btn border border-border bg-border sm:grid-cols-5">{[
                      ['自适应 Sharpe', formatNumber(selectedTrial.evidence.oos_sharpe)],
                      ['最大回撤', formatPct(selectedTrial.evidence.oos_max_drawdown)],
                      ['自适应正收益折', formatPct(selectedTrial.evidence.oos_positive_fold_ratio)],
                      ['交易数', selectedTrial.evidence.oos_n_trades == null ? '—' : String(selectedTrial.evidence.oos_n_trades)],
                      ['自适应门槛', selectedTrial.evidence.qualified == null ? '—' : selectedTrial.evidence.qualified ? '通过' : '未通过'],
                    ].map(([label, value]) => <div key={label} className="bg-surface px-2 py-2"><div className="text-[9px] text-muted">{label}</div><div className="mt-0.5 font-mono text-[10px] font-semibold text-foreground">{value}</div></div>)}</div>}
                    {selectedTrial.mining_run_id && <Link to={`/mining?mode=manual&run=${encodeURIComponent(selectedTrial.mining_run_id)}`} className="inline-flex items-center gap-1 text-accent hover:underline">查看完整挖掘结果</Link>}
                  </div>
                ) : <div className="px-3 py-10 text-center text-[10px] text-muted">选择一个 Trial 查看详情</div>}
              </section>

              <section className="min-w-0">
                <div className="flex items-center gap-1.5 border-b border-border px-3 py-2"><Trophy className="h-3.5 w-3.5 text-warning" /><h2 className="text-xs font-semibold text-foreground">Leaderboard</h2></div>
                <div>
                  {session.leaderboard.map((row, index) => (
                    <button key={`${row.trial_index}-${row.proposal_digest}`} type="button" onClick={() => selectTrial(row.trial_index)} className="grid w-full grid-cols-[24px_minmax(0,1fr)_52px] items-center gap-2 border-b border-border/60 px-3 py-2 text-left hover:bg-elevated/50">
                      <span className="font-mono text-[10px] text-muted">#{index + 1}</span>
                      <span className="min-w-0"><span className="block truncate text-[10px] font-medium text-foreground">{row.title}</span><span className={`mt-0.5 block text-[9px] ${row.qualified ? 'text-success' : 'text-warning'}`}>{row.qualified ? '自适应通过' : '自适应未通过'} · 回撤 {formatPct(row.oos_max_drawdown)}</span></span>
                      <span className="text-right font-mono text-[10px] text-secondary">{formatNumber(row.oos_sharpe)}</span>
                    </button>
                  ))}
                  {!session.leaderboard.length && <div className="px-3 py-10 text-center text-[10px] text-muted">暂无成功 Trial</div>}
                </div>
              </section>
            </div>
          </>
        )}
      </main>
    </div>
  )
}
