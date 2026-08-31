import { useSyncExternalStore } from 'react'
import { api, type AutoresearchSession, type AutoresearchSessionCreate } from './api'

export type AutoresearchCommand = 'approve' | 'pause' | 'resume' | 'stop' | 'seal'

export interface AutoresearchTask {
  sessionId: string | null
  session: AutoresearchSession | null
  isPending: boolean
  reconnecting: boolean
  commandPending: AutoresearchCommand | null
  error: string | null
}

const ACTIVE_SESSION_KEY = 'autoresearch_active_session_id'
const TERMINAL = new Set(['completed', 'failed', 'stopped', 'interrupted'])
const POLL_INTERVAL_MS = 2000

let current: AutoresearchTask = {
  sessionId: null,
  session: null,
  isPending: false,
  reconnecting: false,
  commandPending: null,
  error: null,
}
let eventSource: EventSource | null = null
let connectionToken = 0
let pollTimer: ReturnType<typeof setTimeout> | null = null
let refreshInFlight: { sessionId: string; promise: Promise<void> } | null = null
const listeners = new Set<() => void>()

function emit() { listeners.forEach(listener => listener()) }
function subscribe(listener: () => void) {
  listeners.add(listener)
  return () => listeners.delete(listener)
}
function update(patch: Partial<AutoresearchTask>) {
  current = { ...current, ...patch }
  emit()
}
function isTerminal(session: AutoresearchSession) {
  return TERMINAL.has(session.status)
}
function holdoutSealing(session: AutoresearchSession | null) {
  return session?.holdout?.status === 'sealing'
}
function stopPolling() {
  if (pollTimer) clearTimeout(pollTimer)
  pollTimer = null
}
function closeConnection() {
  connectionToken += 1
  stopPolling()
  eventSource?.close()
  eventSource = null
}
function applySession(session: AutoresearchSession) {
  if (current.sessionId !== session.session_id) return
  const terminal = isTerminal(session)
  const sealing = holdoutSealing(session)
  if (sealing) localStorage.setItem(ACTIVE_SESSION_KEY, session.session_id)
  else if (terminal && localStorage.getItem(ACTIVE_SESSION_KEY) === session.session_id) {
    localStorage.removeItem(ACTIVE_SESSION_KEY)
  }
  update({
    session,
    isPending: !terminal,
    reconnecting: false,
    commandPending: sealing ? 'seal' : null,
    error: session.error || null,
  })
  if (!terminal) return
  if (eventSource) {
    eventSource.close()
    eventSource = null
    connectionToken += 1
  }
  if (sealing) startHoldoutPolling(session.session_id)
  else stopPolling()
}

function startHoldoutPolling(sessionId: string) {
  if (pollTimer) return
  const poll = async () => {
    pollTimer = null
    if (current.sessionId !== sessionId || !holdoutSealing(current.session)) return
    try {
      const session = await api.autoresearchSession(sessionId)
      if (current.sessionId !== sessionId) return
      applySession(session)
    } catch (error) {
      if (current.sessionId !== sessionId) return
      update({ error: String((error as Error).message || error) })
      pollTimer = setTimeout(() => void poll(), POLL_INTERVAL_MS)
    }
  }
  pollTimer = setTimeout(() => void poll(), POLL_INTERVAL_MS)
}

async function refreshSession(sessionId: string, token: number) {
  if (refreshInFlight?.sessionId === sessionId) return refreshInFlight.promise
  const promise = api.autoresearchSession(sessionId)
    .then(session => {
      if (token === connectionToken && current.sessionId === sessionId) applySession(session)
    })
    .catch(error => {
      if (token === connectionToken && current.sessionId === sessionId) {
        update({ error: String((error as Error).message || error) })
      }
    })
    .finally(() => {
      if (refreshInFlight?.promise === promise) refreshInFlight = null
    })
  refreshInFlight = { sessionId, promise }
  return promise
}

function startPolling(sessionId: string, token: number) {
  if (pollTimer) return
  const poll = async () => {
    pollTimer = null
    if (token !== connectionToken || current.sessionId !== sessionId || !current.isPending) return
    await refreshSession(sessionId, token)
    if (token !== connectionToken || current.sessionId !== sessionId || !current.isPending) return
    pollTimer = setTimeout(() => void poll(), POLL_INTERVAL_MS)
  }
  pollTimer = setTimeout(() => void poll(), POLL_INTERVAL_MS)
}

function connect(sessionId: string) {
  closeConnection()
  const token = connectionToken
  const source = new EventSource(`/api/backtest/autoresearch/sessions/${encodeURIComponent(sessionId)}/events`)
  eventSource = source

  source.onopen = () => {
    if (token !== connectionToken) return
    update({ reconnecting: false })
    stopPolling()
  }
  source.addEventListener('snapshot', event => {
    if (token !== connectionToken) return
    try {
      applySession(JSON.parse((event as MessageEvent).data) as AutoresearchSession)
    } catch { /* polling remains the fallback for malformed events */ }
  })

  const refresh = () => {
    if (token === connectionToken) void refreshSession(sessionId, token)
  }
  source.onmessage = refresh
  for (const type of [
    'created', 'approved', 'trial_started', 'trial_progress', 'trial_completed', 'trial_failed',
    'paused', 'resumed', 'stopping', 'completed', 'error', 'failed', 'stopped', 'interrupted',
    'holdout_sealing', 'holdout_sealed', 'holdout_failed',
  ]) source.addEventListener(type, refresh)

  source.onerror = () => {
    if (token !== connectionToken || current.sessionId !== sessionId || !current.isPending) return
    update({ reconnecting: true })
    startPolling(sessionId, token)
  }
}

export async function startAutoresearch(payload: AutoresearchSessionCreate) {
  closeConnection()
  const token = connectionToken
  update({
    sessionId: null,
    session: null,
    isPending: true,
    reconnecting: false,
    commandPending: null,
    error: null,
  })
  try {
    const session = await api.autoresearchCreate(payload)
    if (token !== connectionToken || current.sessionId !== null) return session
    localStorage.setItem(ACTIVE_SESSION_KEY, session.session_id)
    update({ sessionId: session.session_id, session, isPending: !isTerminal(session) })
    if (isTerminal(session)) localStorage.removeItem(ACTIVE_SESSION_KEY)
    else connect(session.session_id)
    return session
  } catch (error) {
    if (token === connectionToken && current.sessionId === null) {
      update({ isPending: false, error: String((error as Error).message || error) })
    }
    throw error
  }
}

export async function attachAutoresearchSession(sessionId: string) {
  closeConnection()
  const token = connectionToken
  update({
    sessionId,
    session: null,
    isPending: true,
    reconnecting: true,
    commandPending: null,
    error: null,
  })
  try {
    const session = await api.autoresearchSession(sessionId)
    if (token !== connectionToken || current.sessionId !== sessionId) return null
    applySession(session)
    if (!isTerminal(session)) {
      localStorage.setItem(ACTIVE_SESSION_KEY, sessionId)
      connect(sessionId)
    }
    return session
  } catch (error) {
    if (token === connectionToken && current.sessionId === sessionId) {
      update({
        isPending: false,
        reconnecting: false,
        error: String((error as Error).message || error),
      })
    }
    return null
  }
}

async function runCommand(command: AutoresearchCommand) {
  const sessionId = current.sessionId
  if (!sessionId || current.commandPending) return null
  update({ commandPending: command, error: null })
  try {
    const session = await ({
      approve: api.autoresearchApprove,
      pause: api.autoresearchPause,
      resume: api.autoresearchResume,
      stop: api.autoresearchStop,
      seal: api.autoresearchSeal,
    } as const)[command](sessionId)
    if (current.sessionId === sessionId) applySession(session)
    return session
  } catch (error) {
    if (current.sessionId === sessionId) {
      update({ commandPending: null, error: String((error as Error).message || error) })
    }
    return null
  }
}

export const approveAutoresearch = () => runCommand('approve')
export const pauseAutoresearch = () => runCommand('pause')
export const resumeAutoresearch = () => runCommand('resume')
export const stopAutoresearch = () => runCommand('stop')
export const sealAutoresearch = () => runCommand('seal')

export function tryReconnectAutoresearch() {
  const sessionId = localStorage.getItem(ACTIVE_SESSION_KEY)
  if (!sessionId) return false
  if (current.sessionId === sessionId && (current.session || current.isPending)) return true
  void attachAutoresearchSession(sessionId)
  return true
}

export function useAutoresearchTask() {
  return useSyncExternalStore(subscribe, () => current, () => current)
}
