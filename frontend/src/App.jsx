import { useState, useEffect } from 'react'
import PDFUploader from './components/PDFUploader'
import ChatWindow from './components/ChatWindow'

const API_BASE = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000'

export default function App() {
  useEffect(() => {
    // Wake up Render backend on app load
    fetch(`${API_BASE}/health`).catch(() => {})
  }, [])

  // uploadedPdfs: [{sessionId, filename, chunkCount}, ...]
  const [uploadedPdfs, setUploadedPdfs]       = useState([])
  const [activeSessionId, setActiveSessionId] = useState(null)
  // messagesMap: {sessionId: Message[]}
  const [messagesMap, setMessagesMap]         = useState({})
  const [isLoading, setIsLoading]             = useState(false)

  const activeSession = uploadedPdfs.find(p => p.sessionId === activeSessionId) ?? null
  const messages      = activeSessionId ? (messagesMap[activeSessionId] ?? []) : []

  function handleUploadSuccess(newSessionId, filename, chunkCount) {
    setUploadedPdfs(prev => [...prev, { sessionId: newSessionId, filename, chunkCount }])
    setActiveSessionId(newSessionId)
    setMessagesMap(prev => ({ ...prev, [newSessionId]: [] }))
  }

  function handleSwitchSession(sid) {
    if (!isLoading) setActiveSessionId(sid)
  }

  function handleDeleteSession(sid) {
    fetch(`${API_BASE}/session/${sid}`, { method: 'DELETE' }).catch(() => {})

    const remaining = uploadedPdfs.filter(p => p.sessionId !== sid)
    setUploadedPdfs(remaining)

    if (activeSessionId === sid) {
      setActiveSessionId(remaining.length > 0 ? remaining[remaining.length - 1].sessionId : null)
    }

    setMessagesMap(prev => {
      const next = { ...prev }
      delete next[sid]
      return next
    })
  }

  async function handleSendMessage(text) {
    const trimmed = text.trim()
    if (!trimmed || isLoading || !activeSessionId) return

    const sid = activeSessionId

    // Push the user turn AND an empty streaming assistant placeholder. Tokens
    // from the NDJSON stream are appended into the placeholder as they arrive.
    setMessagesMap(prev => ({
      ...prev,
      [sid]: [
        ...(prev[sid] ?? []),
        { role: 'user', content: trimmed },
        { role: 'assistant', content: '', cited_pages: [], is_refusal: false, streaming: true },
      ],
    }))
    setIsLoading(true)

    // Patch the last message (the streaming assistant bubble) for this session.
    const patchLast = updater =>
      setMessagesMap(prev => {
        const list = prev[sid] ? [...prev[sid]] : []
        const i = list.length - 1
        if (i >= 0) list[i] = updater(list[i])
        return { ...prev, [sid]: list }
      })

    try {
      const res = await fetch(`${API_BASE}/chat/stream`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ session_id: sid, message: trimmed }),
      })

      if (!res.ok) {
        let detail = `Server error ${res.status}`
        try { const e = await res.json(); detail = e.detail || detail } catch { /* non-JSON */ }
        throw new Error(detail)
      }

      // Read the NDJSON stream line-by-line.
      const reader  = res.body.getReader()
      const decoder = new TextDecoder()
      let buffer = ''

      while (true) {
        const { value, done } = await reader.read()
        if (done) break
        buffer += decoder.decode(value, { stream: true })

        let nl
        while ((nl = buffer.indexOf('\n')) >= 0) {
          const line = buffer.slice(0, nl).trim()
          buffer = buffer.slice(nl + 1)
          if (!line) continue

          let evt
          try { evt = JSON.parse(line) } catch { continue }

          if (evt.type === 'token') {
            patchLast(m => ({ ...m, content: m.content + evt.text }))
          } else if (evt.type === 'done') {
            patchLast(m => ({
              ...m,
              content: evt.answer ?? m.content,
              cited_pages: evt.cited_pages ?? [],
              is_refusal: evt.is_refusal ?? false,
              streaming: false,
            }))
          } else if (evt.type === 'error') {
            patchLast(m => ({ ...m, content: evt.text, is_error: true, streaming: false }))
          }
        }
      }

      // Clear the streaming flag in case the stream ended without a done event.
      patchLast(m => (m.streaming ? { ...m, streaming: false } : m))
    } catch (err) {
      patchLast(m => ({
        ...m,
        content: `Something went wrong: ${err.message}`,
        cited_pages: [],
        is_refusal: false,
        is_error: true,
        streaming: false,
      }))
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <div
      className="flex flex-col h-screen overflow-hidden"
      style={{background: 'linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #0f172a 100%)'}}
    >

      {/* ── Top bar ── */}
      <header
        className="flex-none h-14 flex items-center px-5 gap-3 z-10"
        style={{
          background: 'rgba(15, 23, 42, 0.9)',
          borderBottom: '1px solid rgba(139, 92, 246, 0.3)',
          backdropFilter: 'blur(12px)',
          boxShadow: '0 1px 30px rgba(139,92,246,0.15)',
        }}
      >
        <div
          className="flex items-center justify-center w-7 h-7 rounded-md shrink-0"
          style={{background: 'linear-gradient(135deg, #8b5cf6, #6366f1)'}}
        >
          <DocumentIcon className="w-4 h-4 text-white" />
        </div>
        <div className="flex items-baseline gap-2">
          <span className="text-sm font-semibold tracking-tight text-white">PDF Chat Agent</span>
          <span className="hidden sm:block text-xs" style={{color: '#64748b'}}>
            — every answer cited from your document
          </span>
        </div>
      </header>

      {/* ── Two-panel body ── */}
      <div className="flex flex-1 overflow-hidden">

        {/* Left sidebar — 300 px fixed, scrollable */}
        <aside
          className="w-[300px] shrink-0 flex flex-col overflow-y-auto"
          style={{
            background: 'rgba(15, 23, 42, 0.8)',
            borderRight: '1px solid rgba(139, 92, 246, 0.2)',
          }}
        >
          <PDFUploader
            apiBase={API_BASE}
            uploadedPdfs={uploadedPdfs}
            activeSessionId={activeSessionId}
            onUploadSuccess={handleUploadSuccess}
            onSwitchSession={handleSwitchSession}
            onDeleteSession={handleDeleteSession}
          />
        </aside>

        {/* Right — chat fills remaining width */}
        <main className="flex-1 overflow-hidden">
          <ChatWindow
            messages={messages}
            onSendMessage={handleSendMessage}
            disabled={!activeSessionId}
            isLoading={isLoading}
            pdfName={activeSession?.filename ?? ''}
          />
        </main>

      </div>
    </div>
  )
}

function DocumentIcon({ className }) {
  return (
    <svg className={className} fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
      <path strokeLinecap="round" strokeLinejoin="round"
        d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
    </svg>
  )
}
