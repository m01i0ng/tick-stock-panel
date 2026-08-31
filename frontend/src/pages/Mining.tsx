import { useState } from 'react'
import { BookmarkCheck, Pickaxe, Sparkles } from 'lucide-react'
import { useSearchParams } from 'react-router-dom'
import { PageHeader } from '@/components/PageHeader'
import { AutoresearchWorkbench } from './backtest/AutoresearchWorkbench'
import { MiningWorkbench } from './backtest/MiningWorkbench'
import { ResearchCandidatesDialog } from './backtest/ResearchCandidatesDialog'
import { ResearchAssistantDialog } from './backtest/ResearchAssistantDialog'

export function Mining() {
  const [searchParams, setSearchParams] = useSearchParams()
  const [candidatesOpen, setCandidatesOpen] = useState(false)
  const [assistantOpen, setAssistantOpen] = useState(false)
  const mode = searchParams.get('mode') === 'auto' || searchParams.has('session') ? 'auto' : 'manual'

  const changeMode = (nextMode: 'manual' | 'auto') => {
    const next = new URLSearchParams(searchParams)
    next.set('mode', nextMode)
    if (nextMode === 'manual') {
      next.delete('session')
      next.delete('trial')
    } else {
      next.delete('run')
      next.delete('candidate')
    }
    setSearchParams(next, { replace: true })
  }

  const showSession = (sessionId: string) => {
    const next = new URLSearchParams(searchParams)
    next.set('mode', 'auto')
    next.set('session', sessionId)
    next.delete('trial')
    next.delete('run')
    next.delete('candidate')
    setSearchParams(next, { replace: true })
  }

  return (
    <div className="flex min-h-full flex-col bg-base">
      <PageHeader
        title="挖掘"
        subtitle={<span className="hidden md:inline">{mode === 'auto' ? '有预算的 Catalog 实验编排（发布需密封留出集）' : '嵌套样本外因子与策略挖掘'}</span>}
        className="shrink-0 flex-wrap gap-x-4 gap-y-2 bg-base/95 px-3 lg:flex-nowrap lg:px-5"
        right={(<div className="flex items-center gap-2">
          <nav className="inline-flex rounded-btn border border-border bg-surface/80 p-0.5" aria-label="挖掘模式">
            {([['manual', '手动挖掘', Pickaxe], ['auto', '自动研究', Sparkles]] as const).map(([value, label, Icon]) => (
              <button key={value} type="button" aria-current={mode === value ? 'page' : undefined} onClick={() => changeMode(value)} className={`inline-flex h-7 items-center gap-1.5 rounded-[5px] px-2.5 text-xs font-medium transition-colors ${mode === value ? 'bg-accent text-white shadow-sm' : 'text-secondary hover:bg-elevated hover:text-foreground'}`}>
                <Icon className="h-3.5 w-3.5" />{label}
              </button>
            ))}
          </nav>
          <button
            type="button"
            onClick={() => setCandidatesOpen(true)}
            aria-label="打开候选方案"
            title="候选方案"
            className="inline-flex h-8 shrink-0 items-center gap-1.5 rounded-btn border border-border bg-surface px-2 text-[11px] font-medium text-secondary transition-colors hover:border-accent/40 hover:text-accent sm:px-2.5 sm:text-xs"
          >
            <BookmarkCheck className="h-3.5 w-3.5" />
            <span>候选方案</span>
          </button>
        </div>)}
      />

      <main className="min-h-0 flex-1 px-3 pb-3 pt-3 lg:px-4 lg:pb-4">
        {mode === 'manual'
          ? <MiningWorkbench />
          : <AutoresearchWorkbench onNewSession={() => setAssistantOpen(true)} />}
      </main>

      {candidatesOpen && <ResearchCandidatesDialog onClose={() => setCandidatesOpen(false)} />}
      {assistantOpen && <ResearchAssistantDialog onClose={() => setAssistantOpen(false)} onCreated={showSession} />}
    </div>
  )
}
