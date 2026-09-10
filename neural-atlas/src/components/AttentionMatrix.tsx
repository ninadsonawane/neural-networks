import { useEffect, useId, useRef, useState } from 'react';
import type { KeyboardEvent } from 'react';
import './AttentionMatrix.css';

interface AttentionMatrixProps {
  matrix: number[][];
  rows: string[];
  columns: string[];
  selectedRow: number;
  onSelectRow: (row: number) => void;
  label: string;
  causal?: boolean;
}

const tokenLabel = (token: string) => token === ' ' || token === '␣' ? 'space' : token === '\n' || token === '↵' ? 'newline' : token === '\t' || token === '⇥' ? 'tab' : token;
const tokenGlyph = (token: string) => token === ' ' ? '·' : token === '\n' ? '↵' : token;

export default function AttentionMatrix({ matrix, rows, columns, selectedRow, onSelectRow, label, causal = false }: AttentionMatrixProps) {
  const id = useId();
  const gridRef = useRef<HTMLDivElement>(null);
  const pendingFocus = useRef(false);
  const [activeColumn, setActiveColumn] = useState(0);
  const [inspected, setInspected] = useState<{ row: number; column: number } | null>(null);
  const rowCount = Math.min(rows.length, matrix.length);
  const columnCount = columns.length;
  const currentRow = Math.min(Math.max(0, selectedRow), Math.max(0, rowCount - 1));
  const currentColumn = Math.min(activeColumn, Math.max(0, columnCount - 1));
  const startRow = Math.max(0, Math.min(currentRow - 4, rowCount - 9));
  const visibleRows = Array.from({ length: Math.min(9, rowCount) }, (_, index) => startRow + index);
  const detailRow = inspected && inspected.row < rowCount ? inspected.row : currentRow;
  const detailColumn = inspected && inspected.column < columnCount ? inspected.column : currentColumn;
  const probability = matrix[detailRow]?.[detailColumn] ?? 0;
  const masked = causal && detailColumn > detailRow;
  const rowTotal = (matrix[currentRow] ?? []).reduce((sum, weight) => sum + weight, 0);

  useEffect(() => {
    if (!pendingFocus.current) return;
    const cell = gridRef.current?.querySelector<HTMLButtonElement>(`[data-matrix-cell="${currentRow}-${currentColumn}"]`);
    cell?.focus({ preventScroll: true });
    cell?.scrollIntoView({ block: 'nearest', inline: 'nearest' });
    pendingFocus.current = false;
  }, [currentRow, currentColumn]);

  const moveCell = (event: KeyboardEvent<HTMLButtonElement>, row: number, column: number) => {
    let nextRow = row;
    let nextColumn = column;
    switch (event.key) {
      case 'ArrowLeft': nextColumn = Math.max(0, column - 1); break;
      case 'ArrowRight': nextColumn = Math.min(columnCount - 1, column + 1); break;
      case 'ArrowUp': nextRow = Math.max(0, row - 1); break;
      case 'ArrowDown': nextRow = Math.min(rowCount - 1, row + 1); break;
      case 'Home': nextColumn = 0; break;
      case 'End': nextColumn = columnCount - 1; break;
      default: return;
    }
    event.preventDefault();
    event.stopPropagation();
    if (nextRow === row && nextColumn === column) return;
    pendingFocus.current = true;
    setActiveColumn(nextColumn);
    setInspected({ row: nextRow, column: nextColumn });
    onSelectRow(nextRow);
  };

  if (rowCount === 0 || columnCount === 0) {
    return <div className="am-empty">No attention matrix in this trace.</div>;
  }

  return (
    <section className="am" aria-labelledby={`${id}-title`}>
      <div className="am-heading">
        <div>
          <h4 id={`${id}-title`}>{label}</h4>
          <p id={`${id}-help`}>Rows receive · columns contribute. Focus a cell; use arrow keys to explore.</p>
        </div>
        <label className="am-row-picker">
          <span>Target</span>
          <select aria-label={`${label}: selected target token`} value={currentRow} onChange={(event) => { onSelectRow(Number(event.target.value)); setInspected(null); }}>
            {rows.slice(0, rowCount).map((token, index) => <option key={index} value={index}>{index + 1}: {tokenLabel(token)}</option>)}
          </select>
        </label>
      </div>
      <div className="am-grid-scroll" ref={gridRef} role="region" aria-label={`${label} data`} aria-describedby={`${id}-help`} onMouseLeave={() => setInspected(null)} onBlur={(event) => { if (!event.currentTarget.contains(event.relatedTarget)) setInspected(null); }}>
        <table className="am-table" aria-label={`${label}: attention probability by target and source`}>
          <thead><tr>
            <th scope="col" className="am-corner">to ↓ / from →</th>
            {columns.map((token, column) => <th key={column} scope="col" title={`Source ${column + 1}: ${tokenLabel(token)}`}><span>{tokenGlyph(token)}</span><small>{column + 1}</small></th>)}
          </tr></thead>
          <tbody>
            {visibleRows.map((row) => <tr key={row} className={row === currentRow ? 'am-selected-row' : ''}>
              <th scope="row" title={`Target ${row + 1}: ${tokenLabel(rows[row])}`}><b>{tokenGlyph(rows[row])}</b><small>{row + 1}</small></th>
              {columns.map((source, column) => {
                const weight = matrix[row]?.[column] ?? 0;
                const isMasked = causal && column > row;
                const cellDescription = `Target ${row + 1}, ${tokenLabel(rows[row])}; source ${column + 1}, ${tokenLabel(source)}: ${weight} probability${isMasked ? ', future position masked' : ''}`;
                return <td key={column}>
                  <button
                    type="button"
                    className={`am-cell${isMasked ? ' am-cell-masked' : ''}${row === currentRow && column === currentColumn ? ' am-cell-active' : ''}`}
                    style={{ backgroundColor: isMasked ? undefined : `rgba(100, 220, 214, ${0.055 + Math.sqrt(Math.max(0, Math.min(1, weight))) * 0.88})` }}
                    tabIndex={row === currentRow && column === currentColumn ? 0 : -1}
                    data-matrix-cell={`${row}-${column}`}
                    aria-label={cellDescription}
                    aria-describedby={`${id}-detail`}
                    title={cellDescription}
                    onMouseEnter={() => setInspected({ row, column })}
                    onFocus={() => setInspected({ row, column })}
                    onClick={() => { setActiveColumn(column); onSelectRow(row); setInspected({ row, column }); }}
                    onKeyDown={(event) => moveCell(event, row, column)}
                  >
                    {isMasked ? <span aria-hidden="true">╱</span> : <span className="am-cell-number" aria-hidden="true">{Math.round(weight * 100)}</span>}
                  </button>
                </td>;
              })}
            </tr>)}
          </tbody>
        </table>
      </div>
      <div className="am-caption">
        <span>{rowCount > 9 ? `Targets ${startRow + 1}–${startRow + visibleRows.length} of ${rowCount}. Select a target to move the window.` : `${rowCount} targets × ${columnCount} sources`}</span>
        <span>Selected row sum <strong>{rowTotal.toFixed(3)}</strong></span>
      </div>
      <div className="am-detail" id={`${id}-detail`} role="status" aria-live="polite" aria-atomic="true">
        <span><em>Source</em> <b>{tokenLabel(columns[detailColumn])}</b> <small>#{detailColumn + 1}</small><span className="am-detail-arrow" aria-hidden="true">→</span><em>Target</em> <b>{tokenLabel(rows[detailRow])}</b> <small>#{detailRow + 1}</small></span>
        <strong className="am-probability">{(probability * 100).toFixed(2)}% <small>p = {probability.toPrecision(6)}</small></strong>
        <p>{masked ? 'Future position: the causal mask blocks this connection.' : `This source contributes ${(probability * 100).toFixed(2)}% of the attention-weighted value mixture for this target.`}</p>
      </div>
      <div className="am-legend" aria-label="Heatmap legend"><span><i className="am-low" /> Less weight</span><span><i className="am-high" /> More weight</span>{causal && <span><i className="am-mask-key">╱</i> Future masked</span>}<span>Cell numbers: rounded %</span></div>
    </section>
  );
}
