import { nanoid } from "nanoid";

const conversations = new Map();

export const createConversation = () => {
  const id = nanoid(12);
  const data = {
    id,
    createdAt: new Date().toISOString(),
    lastIntent: null,
    lastDataset: null,
    lastAction: null,
  };
  conversations.set(id, data);
  return data;
};

export const getConversation = (id) => {
  if (!id) {
    return null;
  }
  return conversations.get(id) || null;
};

export const upsertConversation = (id, patch) => {
  const existing = getConversation(id) || createConversation();
  const updated = { ...existing, ...patch };
  conversations.set(updated.id, updated);
  return updated;
};
