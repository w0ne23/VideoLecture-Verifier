// 긴 native select 대신 검색·스크롤을 제공하는 작은 combobox
// options: [{ value, label }]

import { useEffect, useMemo, useRef, useState } from 'react'

export default function SearchableSelect({
  value,
  onChange,
  options = [],
  placeholder = '선택',
  searchPlaceholder = '검색…',
  disabled = false,
  className = '',
}) {
  const rootRef = useRef(null)
  const searchRef = useRef(null)
  const [open, setOpen] = useState(false)
  const [query, setQuery] = useState('')

  const selected = options.find(option => option.value === value)
  // label + value 를 합쳐 부분 문자열 검색
  const filteredOptions = useMemo(() => {
    const normalizedQuery = query.trim().toLowerCase()
    if (!normalizedQuery) return options
    return options.filter(option => (
      `${option.label} ${option.value}`.toLowerCase().includes(normalizedQuery)
    ))
  }, [options, query])

  // 열려 있을 때: 바깥 클릭으로 닫기 + 검색창 포커스
  useEffect(() => {
    if (!open) return undefined
    const handlePointerDown = event => {
      if (!rootRef.current?.contains(event.target)) setOpen(false)
    }
    document.addEventListener('pointerdown', handlePointerDown)
    searchRef.current?.focus()
    return () => document.removeEventListener('pointerdown', handlePointerDown)
  }, [open])

  useEffect(() => {
    if (disabled) {
      setOpen(false)
      setQuery('')
    }
  }, [disabled])

  const selectOption = option => {
    onChange(option.value)
    setOpen(false)
    setQuery('')
  }

  return (
    <div className={`ms-search-select ${open ? 'ms-search-select--open' : ''} ${className}`.trim()} ref={rootRef}>
      <button
        type="button"
        className="ms-search-select-trigger"
        aria-haspopup="listbox"
        aria-expanded={open}
        disabled={disabled}
        onClick={() => setOpen(current => !current)}
      >
        <span className={`ms-search-select-label ${selected ? '' : 'ms-search-select-placeholder'}`.trim()}>
          {selected?.label || placeholder}
        </span>
        <span className="ms-search-select-chevron" aria-hidden="true">⌄</span>
      </button>
      {open && (
        <div className="ms-search-select-menu">
          <input
            ref={searchRef}
            className="ms-search-select-search"
            type="search"
            value={query}
            placeholder={searchPlaceholder}
            aria-label={searchPlaceholder}
            onChange={event => setQuery(event.target.value)}
            onKeyDown={event => {
              if (event.key === 'Escape') setOpen(false)
              if (event.key === 'Enter' && filteredOptions.length === 1) selectOption(filteredOptions[0])
            }}
          />
          <div className="ms-search-select-options" role="listbox">
            {filteredOptions.length ? filteredOptions.map(option => (
              <button
                type="button"
                role="option"
                aria-selected={option.value === value}
                className={`ms-search-select-option${option.value === value ? ' ms-search-select-option--selected' : ''}`}
                key={option.value}
                onClick={() => selectOption(option)}
              >
                <span className="ms-search-select-option-label">{option.label}</span>
              </button>
            )) : (
              <span className="ms-search-select-empty">검색 결과가 없어요.</span>
            )}
          </div>
        </div>
      )}
    </div>
  )
}
