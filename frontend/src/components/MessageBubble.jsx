// Renders a single chat turn. Handles three assistant variants:
//   normal   — white card with inline citation highlights
//   refusal  — amber left border, slightly different text
//   error    — red border, error tone

const PAGE_REGEX = /(\[Page\s+\d+\])/gi

// Splits text on [Page N] markers and wraps them in highlight spans.
function renderWithCitations(text) {
  const parts = text.split(PAGE_REGEX)
  return parts.map((part, i) => {
    const match = part.match(/^\[Page\s+(\d+)\]$/i)
    if (match) {
      return (
        <span
          key={i}
          className="inline-block align-middle mx-0.5 px-1.5 py-0.5 text-xs font-semibold rounded whitespace-nowrap"
          style={{
            background: 'rgba(139, 92, 246, 0.2)',
            color: '#c4b5fd',
            border: '1px solid rgba(139, 92, 246, 0.3)',
          }}
        >
          p.{match[1]}
        </span>
      )
    }
    return <span key={i}>{part}</span>
  })
}

export default function MessageBubble({ role, content, citedPages, isRefusal, isError }) {
  const isUser      = role === 'user'
  const isAssistant = role === 'assistant'

  // ── User bubble ────────────────────────────────────────────────────────────
  if (isUser) {
    return (
      <div className="flex justify-end items-end gap-2"
           style={{animation: 'fadeIn 0.3s ease-out'}}>
        <div className="max-w-[75%]">
          <div
            className="text-white text-sm px-4 py-2.5 rounded-2xl rounded-br-sm leading-relaxed"
            style={{
              background: 'linear-gradient(135deg, #8b5cf6, #6366f1)',
              boxShadow: '0 4px 20px rgba(139, 92, 246, 0.3)',
            }}
          >
            {content}
          </div>
        </div>
        <span className="shrink-0 flex items-center justify-center w-7 h-7 rounded-full mb-0.5" style={{background: 'rgba(139, 92, 246, 0.3)'}}>
          <span style={{color: '#e2e8f0'}}><PersonIcon className="w-3.5 h-3.5" /></span>
        </span>
      </div>
    )
  }

  // ── Assistant bubble ───────────────────────────────────────────────────────
  if (!isAssistant) return null

  // Derive class and inline style per variant
  let wrapperClass = ''
  let wrapperStyle = {
    background: 'rgba(30, 27, 75, 0.9)',
    border: '1px solid rgba(139, 92, 246, 0.2)',
    borderLeft: '3px solid rgba(139, 92, 246, 0.5)',
    color: '#e2e8f0',
  }

  if (isRefusal) {
    wrapperClass = ''
    wrapperStyle = {
      background: 'rgba(120, 53, 15, 0.4)',
      borderLeft: '4px solid #f59e0b',
      border: '1px solid rgba(245, 158, 11, 0.3)',
    }
  }
  if (isError) {
    wrapperClass = 'border-l-4 border-l-red-400 border border-red-800'
    wrapperStyle = {background: 'rgba(127, 29, 29, 0.3)'}
  }

  const textColor = isError ? '#fca5a5' : '#e2e8f0'

  return (
    <div className="flex items-start gap-2.5"
         style={{animation: 'fadeIn 0.3s ease-out'}}>

      {/* Bot avatar */}
      <span className="shrink-0 flex items-center justify-center w-7 h-7 rounded-full mt-0.5" style={{background: 'rgba(139, 92, 246, 0.2)', color: '#a78bfa'}}>
        <BotAvatarIcon className="w-3.5 h-3.5" />
      </span>

      <div className="max-w-[80%] flex flex-col gap-2">

        {/* Message card */}
        <div className={`rounded-2xl rounded-tl-sm px-4 py-3 ${wrapperClass}`}
             style={wrapperStyle}>

          {/* Refusal label */}
          {isRefusal && (
            <p className="text-xs font-semibold mb-1.5 flex items-center gap-1" style={{color: '#fbbf24'}}>
              <span style={{color: '#fbbf24'}}><WarningIcon className="w-3 h-3" /></span>
              Not in document
            </p>
          )}

          <p className="text-sm leading-relaxed whitespace-pre-wrap" style={{color: textColor}}>
            {renderWithCitations(content)}
          </p>
        </div>

        {/* Page citation badges */}
        {!isRefusal && !isError && citedPages.length > 0 && (
          <div className="flex flex-wrap items-center gap-1.5 px-1">
            <span className="text-xs mr-0.5" style={{color: '#64748b'}}>Sources:</span>
            {citedPages.map(page => (
              <span
                key={page}
                className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-xs font-medium"
                style={{
                  background: 'rgba(139, 92, 246, 0.15)',
                  color: '#a78bfa',
                  border: '1px solid rgba(139, 92, 246, 0.25)',
                }}
              >
                <BookmarkIcon className="w-2.5 h-2.5" />
                Page {page}
              </span>
            ))}
          </div>
        )}

      </div>
    </div>
  )
}

// ── Icons ────────────────────────────────────────────────────────────────────

function PersonIcon({ className }) {
  return (
    <svg className={className} fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
      <path strokeLinecap="round" strokeLinejoin="round"
        d="M15.75 6a3.75 3.75 0 11-7.5 0 3.75 3.75 0 017.5 0zM4.501 20.118a7.5 7.5 0 0114.998 0A17.933 17.933 0 0112 21.75c-2.676 0-5.216-.584-7.499-1.632z" />
    </svg>
  )
}

function BotAvatarIcon({ className }) {
  return (
    <svg className={className} fill="currentColor" viewBox="0 0 24 24">
      <path d="M12 2a2 2 0 012 2c0 .74-.4 1.39-1 1.73V7h1a7 7 0 017 7H3a7 7 0 017-7h1V5.73c-.6-.34-1-.99-1-1.73a2 2 0 012-2zM7.5 14a1.5 1.5 0 100 3 1.5 1.5 0 000-3zm9 0a1.5 1.5 0 100 3 1.5 1.5 0 000-3zM5 19.5a.5.5 0 01.5-.5h13a.5.5 0 010 1h-13a.5.5 0 01-.5-.5z" />
    </svg>
  )
}

function WarningIcon({ className }) {
  return (
    <svg className={className} fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2.5}>
      <path strokeLinecap="round" strokeLinejoin="round"
        d="M12 9v3.75m9-.75a9 9 0 11-18 0 9 9 0 0118 0zm-9 3.75h.008v.008H12v-.008z" />
    </svg>
  )
}

function BookmarkIcon({ className }) {
  return (
    <svg className={className} fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
      <path strokeLinecap="round" strokeLinejoin="round"
        d="M17.593 3.322c1.1.128 1.907 1.077 1.907 2.185V21L12 17.25 4.5 21V5.507c0-1.108.806-2.057 1.907-2.185a48.507 48.507 0 0115.186 0z" />
    </svg>
  )
}
