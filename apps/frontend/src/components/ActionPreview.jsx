export default function ActionPreview({ title, data, emptyLabel }) {
  return (
    <div className="action-preview">
      <div className="preview-header">
        <h3>{title}</h3>
      </div>
      {data ? (
        <pre>{JSON.stringify(data, null, 2)}</pre>
      ) : (
        <p className="preview-empty">{emptyLabel}</p>
      )}
    </div>
  );
}
