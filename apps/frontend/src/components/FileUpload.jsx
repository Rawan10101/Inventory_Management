import { useRef } from "react";

const buildPreview = (content, type) => {
  if (type === "json") {
    try {
      const parsed = JSON.parse(content);
      if (Array.isArray(parsed)) {
        return parsed.slice(0, 3);
      }
      return parsed;
    } catch (error) {
      return "Invalid JSON";
    }
  }

  if (type === "csv") {
    return content
      .split(/\r?\n/)
      .filter(Boolean)
      .slice(0, 4)
      .join("\n");
  }

  return content.slice(0, 120);
};

export default function FileUpload({ attachment, onAttachment }) {
  const inputRef = useRef(null);

  const handleFileChange = async (event) => {
    const file = event.target.files?.[0];
    if (!file) return;

    const text = await file.text();
    const extension = file.name.split(".").pop()?.toLowerCase();
    const type = extension === "json" ? "json" : "csv";
    const preview = buildPreview(text, type);

    onAttachment({
      name: file.name,
      type,
      content: text,
      preview,
    });
  };

  const clearFile = () => {
    if (inputRef.current) {
      inputRef.current.value = "";
    }
    onAttachment(null);
  };

  return (
    <div className="file-upload">
      <label className="file-button">
        Upload
        <input ref={inputRef} type="file" onChange={handleFileChange} accept=".csv,.json" />
      </label>
      {attachment && (
        <div className="file-preview">
          <div>
            <p className="file-name">{attachment.name}</p>
            <pre>{typeof attachment.preview === "string" ? attachment.preview : JSON.stringify(attachment.preview, null, 2)}</pre>
          </div>
          <button type="button" className="secondary" onClick={clearFile}>
            Clear
          </button>
        </div>
      )}
    </div>
  );
}
