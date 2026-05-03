import { useEffect, useRef, useState } from 'react'
import MessageBubble from './MessageBubble'

export default function ChatWindow({ messages, onSendMessage, disabled, isLoading, pdfName }) {
  const [input, setInput]           = useState('')
  const [inputFocused, setInputFocused] = useState(false)
  const bottomRef                   = useRef(null)
  const textareaRef                 = useRef(null)

  // Auto-scroll whenever messages change or loading indicator appears
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, isLoading])

  // Auto-resize textarea as the user types (max 6 lines)
  function handleInput(e) {
    setInput(e.target.value)
    const ta = textareaRef.current
    if (ta) {
      ta.style.height = 'auto'
      ta.style.height = `${Math.min(ta.scrollHeight, 144)}px`
    }
  }

  function handleKeyDown(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      submit()
    }
  }

  function submit() {
    const text = input.trim()
    if (!text || disabled || isLoading) return
    onSendMessage(text)
    setInput('')
    // Reset height after clear
    if (textareaRef.current) textareaRef.current.style.height = 'auto'
  }

  const canSend = input.trim().length > 0 && !disabled && !isLoading

  // ── Empty / disabled state ─────────────────────────────────────────────────
  if (disabled) {
    return (
      <div className="h-full flex flex-col items-center justify-center gap-4 p-8 text-center">
        <span className="flex items-center justify-center w-16 h-16 rounded-2xl" style={{background: 'rgba(139, 92, 246, 0.1)'}}>
          <span style={{color: 'rgba(139, 92, 246, 0.4)'}}><InboxIcon className="w-8 h-8" /></span>
        </span>
        <div>
          <p className="text-base font-semibold" style={{color: '#94a3b8'}}>No document loaded</p>
          <p className="text-sm mt-1" style={{color: '#64748b'}}>Upload a PDF on the left to start chatting</p>
        </div>
      </div>
    )
  }

  // ── Active chat ────────────────────────────────────────────────────────────
  return (
    <div className="h-full flex flex-col">

      {/* Subtitle bar */}
      <div
        className="flex-none flex items-center gap-2 px-5 py-3"
        style={{
          background: 'rgba(15, 23, 42, 0.95)',
          borderBottom: '1px solid rgba(139, 92, 246, 0.2)',
        }}
      >
        <span className="flex items-center justify-center w-6 h-6 rounded-md shrink-0" style={{background: 'rgba(139, 92, 246, 0.15)'}}>
          <span style={{color: '#a78bfa'}}><DocumentTextIcon className="w-3.5 h-3.5" /></span>
        </span>
        <p className="text-sm truncate min-w-0" style={{color: '#94a3b8'}}>
          Ask anything about{' '}
          <span className="font-semibold" style={{color: '#e2e8f0'}}>{pdfName}</span>
        </p>
        <span className="ml-auto shrink-0 text-xs" style={{color: '#64748b'}}>
          {messages.length > 0 && `${Math.ceil(messages.length / 2)} exchange${messages.length > 2 ? 's' : ''}`}
        </span>
      </div>

      {/* Messages scroll area */}
      <div className="flex-1 overflow-y-auto scrollbar-thin px-5 py-5 flex flex-col gap-3">

        {/* Welcome hint when no messages yet */}
        {messages.length === 0 && (
          <div className="flex flex-col items-center justify-center h-full gap-3 text-center py-12">
            <span className="flex items-center justify-center w-12 h-12 rounded-xl" style={{background: 'rgba(139, 92, 246, 0.1)'}}>
              <span style={{color: '#8b5cf6'}}><SparklesIcon className="w-6 h-6" /></span>
            </span>
            <div>
              <p className="text-sm font-medium" style={{color: '#94a3b8'}}>Document ready</p>
              <p className="text-xs mt-1 max-w-xs" style={{color: '#64748b'}}>
                Ask any question. Answers will be cited with page numbers.
                Out-of-scope questions will be clearly refused.
              </p>
            </div>
          </div>
        )}

        {messages.map((msg, idx) => (
          <MessageBubble
            key={idx}
            role={msg.role}
            content={msg.content}
            citedPages={msg.cited_pages ?? []}
            isRefusal={msg.is_refusal ?? false}
            isError={msg.is_error ?? false}
          />
        ))}

        {/* Typing indicator — staggered purple dots */}
        {isLoading && (
          <div className="flex items-start gap-2.5">
            <span className="flex items-center justify-center w-7 h-7 rounded-full shrink-0 mt-0.5" style={{background: 'rgba(139, 92, 246, 0.2)'}}>
              <span style={{color: '#a78bfa'}}><BotIcon className="w-3.5 h-3.5" /></span>
            </span>
            <div className="flex items-center gap-1 rounded-2xl rounded-tl-sm px-4 py-3" style={{background: 'rgba(30, 27, 75, 0.9)', border: '1px solid rgba(139, 92, 246, 0.2)'}}>
              <span className="w-1.5 h-1.5 rounded-full animate-bounce [animation-delay:-0.3s]" style={{background: '#a78bfa'}} />
              <span className="w-1.5 h-1.5 rounded-full animate-bounce [animation-delay:-0.15s]" style={{background: '#8b5cf6'}} />
              <span className="w-1.5 h-1.5 rounded-full animate-bounce" style={{background: '#7c3aed'}} />
            </div>
          </div>
        )}

        {/* Scroll target */}
        <div ref={bottomRef} />
      </div>

      {/* Input bar */}
      <div
        className="flex-none px-4 py-3"
        style={{
          background: 'rgba(15, 23, 42, 0.95)',
          borderTop: '1px solid rgba(139, 92, 246, 0.2)',
        }}
      >
        <div className="flex items-end gap-2">
          <textarea
            ref={textareaRef}
            rows={1}
            value={input}
            onChange={handleInput}
            onKeyDown={handleKeyDown}
            onFocus={() => setInputFocused(true)}
            onBlur={() => setInputFocused(false)}
            disabled={disabled || isLoading}
            placeholder="Ask a question about the document…"
            className="flex-1 resize-none overflow-hidden rounded-xl px-3.5 py-2.5 text-sm
                       focus:outline-none focus:ring-2 focus:ring-violet-500 focus:border-transparent
                       disabled:opacity-50 disabled:cursor-not-allowed transition"
            style={{
              background: 'rgba(30, 27, 75, 0.8)',
              border: '1px solid rgba(139, 92, 246, 0.3)',
              color: '#e2e8f0',
              ...(inputFocused ? {boxShadow: '0 0 0 3px rgba(139,92,246,0.15)'} : {}),
            }}
          />
          <button
            onClick={submit}
            disabled={!canSend}
            aria-label="Send message"
            className="shrink-0 flex items-center justify-center w-9 h-9 rounded-xl text-white
                       disabled:opacity-40 disabled:cursor-not-allowed
                       transition-all duration-150"
            style={{
              background: 'linear-gradient(135deg, #8b5cf6, #6366f1)',
              boxShadow: canSend ? '0 4px 20px rgba(139, 92, 246, 0.4)' : 'none',
              transition: 'all 0.2s',
            }}
          >
            <SendIcon className="w-4 h-4" />
          </button>
        </div>
        <p className="text-xs mt-1.5 ml-1" style={{color: '#64748b'}}>
          Enter to send · Shift+Enter for newline
        </p>
      </div>

    </div>
  )
}

// ── Icons ────────────────────────────────────────────────────────────────────

function InboxIcon({ className }) {
  return (
    <svg className={className} fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
      <path strokeLinecap="round" strokeLinejoin="round"
        d="M2.25 13.5h3.86a2.25 2.25 0 012.012 1.244l.256.512a2.25 2.25 0 002.013 1.244h3.218a2.25 2.25 0 002.013-1.244l.256-.512a2.25 2.25 0 012.013-1.244h3.859m-19.5.338V18a2.25 2.25 0 002.25 2.25h15A2.25 2.25 0 0021.75 18v-4.162c0-.224-.034-.447-.1-.661L19.24 5.338a2.25 2.25 0 00-2.15-1.588H6.911a2.25 2.25 0 00-2.15 1.588L2.35 13.177a2.25 2.25 0 00-.1.661z" />
    </svg>
  )
}

function DocumentTextIcon({ className }) {
  return (
    <svg className={className} fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
      <path strokeLinecap="round" strokeLinejoin="round"
        d="M19.5 14.25v-2.625a3.375 3.375 0 00-3.375-3.375h-1.5A1.125 1.125 0 0113.5 7.125v-1.5a3.375 3.375 0 00-3.375-3.375H8.25m0 12.75h7.5m-7.5 3H12M10.5 2.25H5.625c-.621 0-1.125.504-1.125 1.125v17.25c0 .621.504 1.125 1.125 1.125h12.75c.621 0 1.125-.504 1.125-1.125V11.25a9 9 0 00-9-9z" />
    </svg>
  )
}

function SparklesIcon({ className }) {
  return (
    <svg className={className} fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
      <path strokeLinecap="round" strokeLinejoin="round"
        d="M9.813 15.904L9 18.75l-.813-2.846a4.5 4.5 0 00-3.09-3.09L2.25 12l2.846-.813a4.5 4.5 0 003.09-3.09L9 5.25l.813 2.846a4.5 4.5 0 003.09 3.09L15.75 12l-2.846.813a4.5 4.5 0 00-3.09 3.09zM18.259 8.715L18 9.75l-.259-1.035a3.375 3.375 0 00-2.455-2.456L14.25 6l1.036-.259a3.375 3.375 0 002.455-2.456L18 2.25l.259 1.035a3.375 3.375 0 002.456 2.456L21.75 6l-1.035.259a3.375 3.375 0 00-2.456 2.456z" />
    </svg>
  )
}

function BotIcon({ className }) {
  return (
    <svg className={className} fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
      <path strokeLinecap="round" strokeLinejoin="round"
        d="M8.625 9.75a.375.375 0 11-.75 0 .375.375 0 01.75 0zm0 0H8.25m4.125 0a.375.375 0 11-.75 0 .375.375 0 01.75 0zm0 0H12m4.125 0a.375.375 0 11-.75 0 .375.375 0 01.75 0zm0 0h-.375m-13.5 3.01c0 1.6 1.123 2.994 2.707 3.227 1.087.16 2.185.283 3.293.369V21l4.184-4.183a1.14 1.14 0 01.778-.332 48.294 48.294 0 005.83-.498c1.585-.233 2.708-1.626 2.708-3.228V6.741c0-1.602-1.123-2.995-2.707-3.228A48.394 48.394 0 0012 3c-2.392 0-4.744.175-7.043.513C3.373 3.746 2.25 5.14 2.25 6.741v6.018z" />
    </svg>
  )
}

function SendIcon({ className }) {
  return (
    <svg className={className} fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
      <path strokeLinecap="round" strokeLinejoin="round" d="M6 12L3.269 3.126A59.768 59.768 0 0121.485 12 59.77 59.77 0 013.27 20.876L5.999 12zm0 0h7.5" />
    </svg>
  )
}
