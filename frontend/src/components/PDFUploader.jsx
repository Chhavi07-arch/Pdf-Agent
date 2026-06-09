import { useEffect, useRef, useState } from 'react'

const MAX_FILE_MB = 20

const UPLOAD_STAGES = [
  'Parsing pages…',
  'Creating embeddings…',
  'Indexing content…',
  'Almost ready…',
]

export default function PDFUploader({
  apiBase,
  uploadedPdfs,
  activeSessionId,
  onUploadSuccess,
  onSwitchSession,
  onDeleteSession,
}) {
  // No 'success' phase needed — success is reflected in the uploadedPdfs list
  const [phase, setPhase]           = useState('idle')  // 'idle' | 'uploading' | 'error'
  const [errorMsg, setErrorMsg]     = useState('')
  const [dragActive, setDragActive] = useState(false)
  const [stageIdx, setStageIdx]     = useState(0)

  useEffect(() => {
    if (phase !== 'uploading') return
    const timer = setInterval(() => {
      setStageIdx(i => Math.min(i + 1, UPLOAD_STAGES.length - 1))
    }, 6000)
    return () => clearInterval(timer)
  }, [phase])

  const fileInputRef = useRef(null)
  const dragCounter  = useRef(0)

  // ── Drag handlers ──────────────────────────────────────────────────────────
  function onDragEnter(e) {
    e.preventDefault()
    dragCounter.current++
    setDragActive(true)
  }
  function onDragLeave(e) {
    e.preventDefault()
    dragCounter.current--
    if (dragCounter.current === 0) setDragActive(false)
  }
  function onDragOver(e) { e.preventDefault() }
  function onDrop(e) {
    e.preventDefault()
    dragCounter.current = 0
    setDragActive(false)
    const file = e.dataTransfer.files?.[0]
    if (file) processFile(file)
  }

  // ── File validation + upload ───────────────────────────────────────────────
  function processFile(file) {
    if (!file.name.toLowerCase().endsWith('.pdf')) {
      setPhase('error')
      setErrorMsg('Only PDF files are supported.')
      return
    }
    if (file.size > MAX_FILE_MB * 1024 * 1024) {
      setPhase('error')
      setErrorMsg(`File exceeds the ${MAX_FILE_MB} MB limit.`)
      return
    }
    uploadFile(file)
  }

  // Poll the status endpoint until the session reports "ready".
  // Embedding is synchronous server-side, so this returns on the first poll —
  // it is a safety net that also lets the UI confirm indexing before unlocking.
  async function waitUntilReady(sessionId, { tries = 20, intervalMs = 500 } = {}) {
    for (let i = 0; i < tries; i++) {
      try {
        const res = await fetch(`${apiBase}/session/${sessionId}/status`)
        if (res.ok) {
          const s = await res.json()
          if (s.status === 'ready') return s
        }
      } catch { /* transient — retry */ }
      await new Promise(r => setTimeout(r, intervalMs))
    }
    return null  // timed out — proceed anyway; chat will surface any real error
  }

  async function uploadFile(file) {
    setPhase('uploading')
    setStageIdx(0)
    setErrorMsg('')

    const form = new FormData()
    form.append('file', file)

    try {
      const res = await fetch(`${apiBase}/upload`, { method: 'POST', body: form })
      const data = await res.json()

      if (!res.ok) {
        throw new Error(data.detail || `Upload failed (${res.status})`)
      }

      // Confirm the session is indexed before unlocking chat.
      await waitUntilReady(data.session_id)

      // Notify parent to add to list and make active, then reset to idle
      onUploadSuccess(data.session_id, data.filename, data.chunk_count)
      setPhase('idle')
    } catch (err) {
      setPhase('error')
      setErrorMsg(err.message)
    }
  }

  const hasPdfs = uploadedPdfs.length > 0

  return (
    <div className="flex flex-col gap-4 p-5">

      {/* ── Uploaded PDFs list ─────────────────────────────────────────────── */}
      {hasPdfs && (
        <div className="flex flex-col gap-2">
          <SectionLabel>Documents</SectionLabel>
          <div className="flex flex-col gap-1.5">
            {uploadedPdfs.map(pdf => (
              <PdfCard
                key={pdf.sessionId}
                pdf={pdf}
                isActive={pdf.sessionId === activeSessionId}
                onSwitch={() => onSwitchSession(pdf.sessionId)}
                onDelete={() => onDeleteSession(pdf.sessionId)}
              />
            ))}
          </div>
        </div>
      )}

      {/* ── Upload zone ────────────────────────────────────────────────────── */}
      <div className="flex flex-col gap-3">
        <SectionLabel>{hasPdfs ? 'Add another PDF' : 'Document'}</SectionLabel>

        {phase === 'uploading' ? (
          <div
            className="rounded-xl p-6 flex flex-col items-center gap-3"
            style={{
              background: 'rgba(30, 27, 75, 0.6)',
              border: '1px solid rgba(139, 92, 246, 0.2)',
            }}
          >
            <Spinner />
            <p className="text-sm font-medium" style={{color: '#a78bfa'}}>Processing document…</p>
            <p className="text-xs" style={{color: '#64748b'}}>{UPLOAD_STAGES[stageIdx]}</p>

            {/* Indeterminate progress bar — embedding time is not reported
                incrementally, so a sliding bar conveys "working" honestly. */}
            <div
              className="w-full mt-1 overflow-hidden rounded-full"
              style={{height: '4px', background: 'rgba(139, 92, 246, 0.15)'}}
            >
              <div
                style={{
                  width: '40%',
                  height: '100%',
                  borderRadius: '9999px',
                  background: 'linear-gradient(90deg, #8b5cf6, #6366f1)',
                  animation: 'progressSlide 1.4s ease-in-out infinite',
                }}
              />
            </div>
          </div>
        ) : (
          <>
            <div
              onDragEnter={onDragEnter}
              onDragLeave={onDragLeave}
              onDragOver={onDragOver}
              onDrop={onDrop}
              onClick={() => fileInputRef.current?.click()}
              className="rounded-xl p-6 flex flex-col items-center gap-3 cursor-pointer transition-all duration-200 select-none"
              style={dragActive ? {
                background: 'rgba(139, 92, 246, 0.1)',
                border: '2px dashed #8b5cf6',
                boxShadow: '0 0 30px rgba(139,92,246,0.2)',
              } : {
                background: 'rgba(30, 27, 75, 0.5)',
                border: '2px dashed rgba(139, 92, 246, 0.3)',
              }}
            >
              <span
                className="flex items-center justify-center w-10 h-10 rounded-xl transition-colors duration-200"
                style={{background: dragActive ? 'rgba(139, 92, 246, 0.25)' : 'rgba(139, 92, 246, 0.1)'}}
              >
                <UploadIcon
                  className="w-5 h-5"
                  style={{color: dragActive ? '#a78bfa' : '#64748b'}}
                />
              </span>

              <div className="text-center">
                <p className="text-sm font-medium" style={{color: '#94a3b8'}}>
                  {dragActive ? 'Drop to upload' : 'Drag & drop a PDF'}
                </p>
                <p className="text-xs mt-0.5" style={{color: '#64748b'}}>or click to browse</p>
              </div>

              <span className="text-xs" style={{color: '#64748b'}}>PDF · max {MAX_FILE_MB} MB</span>
            </div>

            {/* Hidden file input */}
            <input
              ref={fileInputRef}
              type="file"
              accept=".pdf"
              className="hidden"
              onChange={e => {
                const file = e.target.files?.[0]
                if (file) processFile(file)
                e.target.value = ''
              }}
            />

            {/* Error state */}
            {phase === 'error' && (
              <div
                className="flex items-start gap-2 rounded-lg px-3 py-2.5"
                style={{
                  background: 'rgba(127, 29, 29, 0.3)',
                  border: '1px solid rgba(239, 68, 68, 0.3)',
                }}
              >
                <AlertIcon className="w-4 h-4 shrink-0 mt-0.5" style={{color: '#f87171'}} />
                <div className="flex-1 min-w-0">
                  <p className="text-xs font-medium" style={{color: '#fca5a5'}}>{errorMsg}</p>
                  <button
                    className="text-xs underline mt-0.5 hover:no-underline"
                    style={{color: '#f87171'}}
                    onClick={() => setPhase('idle')}
                  >
                    Try again
                  </button>
                </div>
              </div>
            )}
          </>
        )}
      </div>

      {/* Footer hint — only shown before first upload */}
      {!hasPdfs && (
        <div className="pt-1" style={{borderTop: '1px solid rgba(139, 92, 246, 0.15)'}}>
          <p className="text-xs leading-relaxed" style={{color: '#64748b'}}>
            Answers are grounded strictly in the document. Every response includes
            page citations. Questions outside the document will be refused.
          </p>
        </div>
      )}
    </div>
  )
}

// ── PDF card in the list ──────────────────────────────────────────────────────

function PdfCard({ pdf, isActive, onSwitch, onDelete }) {
  return (
    <div
      onClick={onSwitch}
      className="group relative rounded-xl p-3 cursor-pointer transition-all duration-200"
      style={isActive ? {
        background: 'linear-gradient(135deg, #1e1b4b, #2e1065)',
        border: '1px solid rgba(139, 92, 246, 0.4)',
        borderLeft: '4px solid #8b5cf6',
        boxShadow: '0 4px 20px rgba(139,92,246,0.2)',
      } : {
        background: 'rgba(30, 27, 75, 0.4)',
        border: '1px solid rgba(139, 92, 246, 0.15)',
      }}
    >
      <div className="flex items-start gap-2.5 pr-6">
        <span
          className="mt-0.5 flex items-center justify-center w-6 h-6 rounded-full shrink-0"
          style={{
            background: isActive
              ? 'linear-gradient(135deg, #8b5cf6, #6366f1)'
              : 'rgba(139, 92, 246, 0.15)',
          }}
        >
          <CheckIcon className="w-3 h-3 text-white" />
        </span>
        <div className="min-w-0">
          <p
            className="text-xs font-semibold truncate"
            title={pdf.filename}
            style={{color: isActive ? '#e2e8f0' : '#94a3b8'}}
          >
            {pdf.filename}
          </p>
          <p className="text-xs mt-0.5" style={{color: '#64748b'}}>{pdf.chunkCount} sections indexed</p>
        </div>
      </div>

      <button
        onClick={e => { e.stopPropagation(); onDelete() }}
        className="absolute top-2.5 right-2.5 opacity-0 group-hover:opacity-100
                   flex items-center justify-center w-5 h-5 rounded
                   hover:bg-red-900 transition-all duration-150"
        style={{color: '#64748b'}}
        aria-label="Remove document"
      >
        <XIcon className="w-3 h-3" />
      </button>
    </div>
  )
}

// ── Tiny sub-components ───────────────────────────────────────────────────────

function SectionLabel({ children }) {
  return (
    <p className="text-xs font-bold uppercase tracking-widest" style={{color: '#a78bfa'}}>
      {children}
    </p>
  )
}

function Spinner() {
  return (
    <svg className="w-7 h-7 animate-spin" style={{color: '#8b5cf6'}} fill="none" viewBox="0 0 24 24">
      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
      <path className="opacity-75" fill="currentColor"
        d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
    </svg>
  )
}

function CheckIcon({ className }) {
  return (
    <svg className={className} fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2.5}>
      <path strokeLinecap="round" strokeLinejoin="round" d="M5 13l4 4L19 7" />
    </svg>
  )
}

function UploadIcon({ className, style }) {
  return (
    <svg className={className} style={style} fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
      <path strokeLinecap="round" strokeLinejoin="round"
        d="M3 16.5v2.25A2.25 2.25 0 005.25 21h13.5A2.25 2.25 0 0021 18.75V16.5m-13.5-9L12 3m0 0l4.5 4.5M12 3v13.5" />
    </svg>
  )
}

function AlertIcon({ className, style }) {
  return (
    <svg className={className} style={style} fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
      <path strokeLinecap="round" strokeLinejoin="round"
        d="M12 9v3.75m-9.303 3.376c-.866 1.5.217 3.374 1.948 3.374h14.71c1.73 0 2.813-1.874 1.948-3.374L13.949 3.378c-.866-1.5-3.032-1.5-3.898 0L2.697 16.126zM12 15.75h.007v.008H12v-.008z" />
    </svg>
  )
}

function XIcon({ className }) {
  return (
    <svg className={className} fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
      <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
    </svg>
  )
}
