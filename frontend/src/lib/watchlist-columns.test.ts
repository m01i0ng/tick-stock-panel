// @vitest-environment node
import { describe, expect, it } from 'vitest'
import { mergeColumns } from './list-columns'
import { BUILTIN_COLUMNS, COLUMN_GROUPS } from './watchlist-columns'

// #328: 自选页新增 今开/最高/最低/昨收/成交量/涨停价/跌停价 七列。
// 关键兼容契约: 老用户的持久化配置不含这些列 id, mergeColumns 必须把它们
// 补进结果且保持默认隐藏 (visible:false), 已有列的用户显隐不受影响。
describe('watchlist columns: 新增内置列向后兼容', () => {
  const NEW_KEYS = ['open', 'high', 'low', 'prev_close', 'volume', 'limit_up_price', 'limit_down_price']

  it('新增列已注册且默认隐藏, 并归入价格/成交分组', () => {
    for (const key of NEW_KEYS) {
      const col = BUILTIN_COLUMNS.find(c => c.source.type === 'builtin' && c.source.key === key)
      expect(col, `内置列 ${key} 应存在`).toBeTruthy()
      expect(col!.visible).toBe(false)
    }
    const groupKeys = new Set(COLUMN_GROUPS.flatMap(g => g.keys))
    for (const key of NEW_KEYS) {
      expect(groupKeys.has(key), `列 ${key} 应出现在 COLUMN_GROUPS 中`).toBe(true)
    }
  })

  it('老用户配置 (不含新列 id) 合并后自动补齐新列且默认隐藏', () => {
    // 模拟新列上线前保存的配置: 只有部分老列
    const saved = BUILTIN_COLUMNS
      .filter(c => ['builtin:symbol', 'builtin:price', 'builtin:pct', 'builtin:turnover'].includes(c.id))
      .map(c => ({ ...c }))

    const merged = mergeColumns(saved, BUILTIN_COLUMNS)
    const byId = new Map(merged.map(c => [c.id, c]))

    for (const key of NEW_KEYS) {
      const col = byId.get(`builtin:${key}`)
      expect(col, `合并后应包含新列 builtin:${key}`).toBeTruthy()
      expect(col!.visible).toBe(false)
    }
    // 用户可见性保留: 老列的 visible 不被新默认值覆盖
    expect(byId.get('builtin:price')!.visible).toBe(true)
    expect(byId.get('builtin:symbol')!.visible).toBe(true)
    // 用户顺序在前, 新列追加在后
    expect(merged[0].id).toBe('builtin:symbol')
    expect(merged.slice(0, 4).map(c => c.id)).toEqual(
      ['builtin:symbol', 'builtin:price', 'builtin:pct', 'builtin:turnover'],
    )
  })

  it('用户开启过的新列 (版本间来回) 显隐被保留', () => {
    const saved = BUILTIN_COLUMNS
      .filter(c => c.id === 'builtin:symbol' || c.id === 'builtin:volume')
      .map(c => ({ ...c, visible: c.id === 'builtin:volume' }))

    const merged = mergeColumns(saved, BUILTIN_COLUMNS)
    const byId = new Map(merged.map(c => [c.id, c]))
    expect(byId.get('builtin:volume')!.visible).toBe(true)
    expect(byId.get('builtin:high')!.visible).toBe(false)
  })
})
