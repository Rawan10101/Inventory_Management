import express from "express";
import { nanoid } from "nanoid";
import { z } from "zod";
import { classifyIntent } from "../services/intentClassifier.js";
import { parseDataFromInput } from "../services/dataParser.js";
import { planAction } from "../services/actionPlanner.js";
import { executeAction } from "../services/actionExecutor.js";
import { createConversation, getConversation, upsertConversation } from "../services/conversationStore.js";

const router = express.Router();
const pendingActions = new Map();

const ChatRequestSchema = z
  .object({
    conversationId: z.string().optional(),
    message: z.string().optional(),
    attachment: z
      .object({
        name: z.string().optional(),
        type: z.enum(["csv", "json"]),
        content: z.string().min(1),
      })
      .optional(),
  })
  .refine((data) => (data.message && data.message.trim().length > 0) || data.attachment, {
    message: "Provide a message or attachment.",
  });

const ConfirmRequestSchema = z.object({
  confirmationId: z.string().min(1),
  approve: z.boolean(),
});

router.post("/interpret", async (req, res, next) => {
  try {
    const parsedBody = ChatRequestSchema.safeParse(req.body || {});
    if (!parsedBody.success) {
      return res.status(400).json({ error: parsedBody.error.issues[0]?.message || "Invalid request." });
    }

    const { conversationId, message, attachment } = parsedBody.data;
    const conversation = getConversation(conversationId) || createConversation();

    const intentResult = classifyIntent(message || "");
    const parsedData = parseDataFromInput({ message: message || "", attachment });
    const plan = planAction({
      message: message || "",
      context: conversation,
      intentResult,
      parsedData,
    });

    const updatedConversation = upsertConversation(conversation.id, {
      lastIntent: plan.resolvedIntent || intentResult.intent,
      lastDataset: plan.action?.dataset || conversation.lastDataset,
      lastAction: plan.action || conversation.lastAction,
    });

    if (plan.action && !plan.requiresConfirmation) {
      const execution = await executeAction({
        action: plan.action,
        conversationId: updatedConversation.id,
        userMessage: message || "",
      });

      return res.json({
        conversationId: updatedConversation.id,
        intent: intentResult,
        resolvedIntent: plan.resolvedIntent || intentResult.intent,
        action: plan.action,
        actionPreview: plan.actionPreview,
        requiresConfirmation: false,
        assistantMessage: execution.assistantMessage,
        structuredOutput: execution.structuredOutput,
      });
    }

    let confirmationId = null;
    if (plan.action && plan.requiresConfirmation) {
      confirmationId = nanoid(10);
      pendingActions.set(confirmationId, {
        action: plan.action,
        conversationId: updatedConversation.id,
        userMessage: message || "",
        createdAt: Date.now(),
      });
    }

    return res.json({
      conversationId: updatedConversation.id,
      intent: intentResult,
      resolvedIntent: plan.resolvedIntent || intentResult.intent,
      action: plan.action,
      actionPreview: plan.actionPreview,
      requiresConfirmation: plan.requiresConfirmation,
      confirmationId,
      assistantMessage: plan.assistantMessage,
      structuredOutput: plan.action ? { action: plan.action } : null,
    });
  } catch (error) {
    return next(error);
  }
});

router.post("/confirm", async (req, res, next) => {
  try {
    const parsedBody = ConfirmRequestSchema.safeParse(req.body || {});
    if (!parsedBody.success) {
      return res.status(400).json({ error: parsedBody.error.issues[0]?.message || "Invalid request." });
    }
    const { confirmationId, approve } = parsedBody.data;
    const pending = pendingActions.get(confirmationId);

    if (!pending) {
      return res.status(404).json({ error: "Confirmation not found." });
    }

    if (!approve) {
      pendingActions.delete(confirmationId);
      return res.json({
        status: "cancelled",
        assistantMessage: "Action cancelled. No changes were made.",
      });
    }

    const execution = await executeAction({
      action: pending.action,
      conversationId: pending.conversationId,
      userMessage: pending.userMessage,
    });

    pendingActions.delete(confirmationId);

    return res.json({
      status: "executed",
      assistantMessage: execution.assistantMessage,
      structuredOutput: execution.structuredOutput,
    });
  } catch (error) {
    return next(error);
  }
});

export default router;
