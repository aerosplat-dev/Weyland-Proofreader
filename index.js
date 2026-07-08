import { eventSource, event_types, saveSettingsDebounced, updateMessageBlock, saveChatConditional, syncMesToSwipe } from '../../../script.js';
import { power_user } from '../../power-user.js';
import { parseReasoningFromString } from '../../reasoning.js';
import { getTokenCountAsync } from '../../tokenizers.js';
import { ConnectionManagerRequestService } from '../shared.js';

const { extensionSettings, renderExtensionTemplateAsync, chat } = SillyTavern.getContext();

const MODULE_NAME = 'Weyland-Proofreader';
const extensionVersion = '1.0.0';

const DEFAULT_PROMPT = `You are a precise copy editor for a single AI-generated roleplay reply. Fix only the numbered issues below -- do not otherwise rewrite, expand, shorten, summarize, or change the meaning, tone, or style.

1. Fix clear spelling mistakes, typos, and stray punctuation/capitalization errors.
2. Dialogue (spoken aloud) uses "double quotes" -- convert single-quoted dialogue to double quotes.
3. Internal thought uses [square brackets] only -- strip any asterisks or quotes wrapping the brackets themselves. Asterisks or quotes appearing INSIDE the brackets are correct and must stay.
4. Narration (actions/descriptions) uses *asterisks* -- add them around narration left unwrapped, especially beside a bracketed thought or an em dash marking interrupted dialogue.
5. Within narration already wrapped in *single asterisks*, emphasize a word or phrase with ***triple asterisks***, never nested single asterisks (wrong: *she *never* answered*; right: *she ***never*** answered*).
6. Never apply rules 2-5 to structural markup: a short, standalone line of lowercase [tag] codes (e.g. an expression/clothing footer), lines wrapped in "¦" marks, phone-message lines, relationship lines like "New Friend: {name}", or "bpm" readouts. Leave these exactly as written.
7. If private model reasoning leaked into the reply without its tags (opening, closing, or both missing), wrap that section -- and only that section -- in this app's exact reasoning tags (given below), reproducing them exactly, including line breaks. Leaked reasoning sounds like the model planning its response (analyzing the scene, deciding what the character should do), not something the character would think -- don't confuse it with a character's own [bracketed thought].
8. If a single word or short phrase is in a different language than the rest of the reply, for no narrative reason (not a character intentionally speaking another language, not a common loanword), replace it with the correct word in the reply's own language.

Output only the corrected reply: no preamble, no explanation, no wrapping quotes around the whole thing.`;

/**
 * @typedef {Object} WeylandProofreaderSettings
 * @property {boolean} enabled
 * @property {string} profileId
 * @property {string} modelId
 * @property {number} maxTokens
 * @property {number} timeoutSeconds
 * @property {string} systemPrompt
 * @property {boolean} correctSwipes
 * @property {boolean} debug
 */

/** @type {WeylandProofreaderSettings} */
const defaultSettings = {
    enabled: false,
    profileId: '',
    modelId: '',
    maxTokens: 2048,
    timeoutSeconds: 20,
    systemPrompt: DEFAULT_PROMPT,
    correctSwipes: true,
    debug: false,
};

/** @type {WeylandProofreaderSettings} */
let settings;

const messagesInFlight = new WeakSet();

// ConnectionManagerRequestService doesn't surface a finish_reason, so a truncated
// completion is indistinguishable from a genuine (shorter) correction after the fact.
// Guard against it up front instead: skip correcting a message the token ceiling can't
// safely round-trip, and treat an implausibly short result as a bad response, not a fix.
const TOKEN_CEILING_MARGIN = 150;
const MIN_CORRECTED_LENGTH_RATIO = 0.6;

function getSettings() {
    if (!extensionSettings[MODULE_NAME]) {
        extensionSettings[MODULE_NAME] = structuredClone(defaultSettings);
    }

    for (const key in defaultSettings) {
        if (extensionSettings[MODULE_NAME][key] === undefined) {
            extensionSettings[MODULE_NAME][key] = defaultSettings[key];
        }
    }

    settings = extensionSettings[MODULE_NAME];
}

function proofreaderDebug(text, error) {
    if (settings === undefined) getSettings();
    if (settings?.debug) {
        if (error) {
            console.debug(`[${MODULE_NAME}] ${text}`, error);
        } else {
            console.debug(`[${MODULE_NAME}] ${text}`);
        }
    }
}

/**
 * Builds the correction system prompt, appending the app's exact reasoning tag strings
 * (read live, since they're user-configurable) so a model asked to fix leaked, untagged
 * reasoning reproduces the tags in the exact form the native reasoning parser expects.
 * @returns {string}
 */
function buildSystemPrompt() {
    const { prefix, suffix } = power_user.reasoning ?? {};
    if (!prefix || !suffix) {
        return settings.systemPrompt;
    }
    return `${settings.systemPrompt}\n\nReasoning tags for rule 7 -- reproduce exactly, including line breaks: opening = ${JSON.stringify(prefix)}, closing = ${JSON.stringify(suffix)}`;
}

/**
 * Builds the correction request payload for the given connection profile.
 * @param {import('../connection-manager/index.js').ConnectionProfile} profile
 * @param {string} text
 * @returns {string | Array<{role: string, content: string}>}
 */
function buildPrompt(profile, text) {
    const apiMap = ConnectionManagerRequestService.validateProfile(profile);
    const systemPrompt = buildSystemPrompt();
    if (apiMap.selected === 'openai') {
        return [
            { role: 'system', content: systemPrompt },
            { role: 'user', content: text },
        ];
    }
    return `${systemPrompt}\n\n${text}`;
}

/**
 * Corrects a single chat message in place using the configured connection profile.
 * @param {number} messageId
 */
async function correctMessage(messageId) {
    if (settings === undefined) getSettings();
    if (!settings.enabled || !settings.profileId) return;
    if (typeof messageId !== 'number' || messageId < 0) return;

    const message = chat[messageId];
    if (!message || message.is_user || message.is_system) return;
    // Message 0 is the character card's permanent greeting slot, not a generated reply,
    // except for Weybot where it doubles as a real system message (mirrors Weyland-Formatter).
    if (messageId === 0 && message.name !== 'Weybot') return;

    const original = message.mes;
    if (!original || !original.trim()) return;

    // Keyed on the message object itself (not the index) so a delete/insert that shifts
    // indices while a correction is in flight can't strand or misapply the in-flight guard.
    if (messagesInFlight.has(message)) {
        proofreaderDebug(`Message ${messageId} is already being corrected, skipping`);
        return;
    }

    const context = SillyTavern.getContext();
    const profile = context.extensionSettings.connectionManager?.profiles?.find((p) => p.id === settings.profileId);
    if (!profile) {
        proofreaderDebug(`Connection profile ${settings.profileId} not found`);
        return;
    }

    const inputTokens = await getTokenCountAsync(original, 0);
    if (inputTokens + TOKEN_CEILING_MARGIN > settings.maxTokens) {
        proofreaderDebug(`Message ${messageId} (${inputTokens} tokens) is too close to the ${settings.maxTokens}-token ceiling, skipping to avoid truncation`);
        return;
    }

    let timeoutHandle;
    messagesInFlight.add(message);

    try {
        const controller = new AbortController();
        const timeoutMs = Math.max(1, settings.timeoutSeconds || defaultSettings.timeoutSeconds) * 1000;
        timeoutHandle = setTimeout(() => controller.abort(), timeoutMs);

        const prompt = buildPrompt(profile, original);
        proofreaderDebug(`Correcting message ${messageId}`);

        // The Connection Profile only supplies the backend/credentials; a Model ID here
        // overrides just the model, so correction can use a different (smaller/faster)
        // model than the profile's own, without needing a second profile to maintain.
        const overridePayload = settings.modelId ? { model: settings.modelId } : {};

        const result = await ConnectionManagerRequestService.sendRequest(
            settings.profileId,
            prompt,
            settings.maxTokens,
            { stream: false, signal: controller.signal },
            overridePayload,
        );
        const corrected = result?.content?.trim();

        if (!corrected) {
            proofreaderDebug(`Empty correction result for message ${messageId}, leaving unchanged`);
            return;
        }

        if (corrected === original) {
            proofreaderDebug(`No changes needed for message ${messageId}`);
            return;
        }

        if (corrected.length < original.length * MIN_CORRECTED_LENGTH_RATIO) {
            console.warn(`[${MODULE_NAME}] Correction for message ${messageId} is implausibly short (likely truncated or off-instruction), discarding`);
            toastr.warning('Weyland Proofreader response looked truncated or off-instruction; message left unchanged.');
            return;
        }

        if (chat[messageId] !== message || message.mes !== original) {
            proofreaderDebug(`Message ${messageId} changed during correction, discarding result`);
            return;
        }

        message.mes = corrected;
        syncMesToSwipe(messageId);
        updateMessageBlock(messageId, message);
        await saveChatConditional();

        // A fresh <think>...</think> block we just added/completed needs the native
        // reasoning parser to actually hide it -- our own write-back above only makes
        // the tags visible text; it doesn't move anything into .extra.reasoning. That
        // parser listens for MESSAGE_UPDATED, so only fire it when a tagged block was
        // actually found (parseReasoningFromString always returns a truthy object --
        // even on no match -- so a real match requires checking .content changed) and
        // only when the native parser would actually act on it (it silently refuses if
        // .extra.reasoning is already non-empty from an earlier, separately-tagged
        // block). Re-check the message wasn't touched by something else during the
        // await above before treating our own snapshot as still current.
        const parsedReasoning = parseReasoningFromString(corrected);
        const foundReasoningBlock = parsedReasoning && parsedReasoning.content !== corrected;
        if (foundReasoningBlock && !message.extra?.reasoning && chat[messageId] === message && message.mes === corrected) {
            proofreaderDebug(`Message ${messageId} contains a reasoning block, handing off to the native parser`);
            await eventSource.emit(event_types.MESSAGE_UPDATED, messageId);
        }

        if (message.extra?.token_count) {
            message.extra.token_count = await getTokenCountAsync(message.mes, 0);
        }

        proofreaderDebug(`Corrected message ${messageId}`);
    } catch (error) {
        // ConnectionManagerRequestService wraps every error, so the AbortError from our
        // own timeout ends up demoted to .cause instead of being the top-level error.
        if (error?.name === 'AbortError' || error?.cause?.name === 'AbortError') {
            toastr.warning('Weyland Proofreader timed out; message left unchanged.');
        } else {
            console.error(`[${MODULE_NAME}] Failed to correct message ${messageId}:`, error);
            toastr.error(String(error?.message ?? error), 'Weyland Proofreader failed');
        }
    } finally {
        clearTimeout(timeoutHandle);
        messagesInFlight.delete(message);
    }
}

function findLastAiMessageIndex() {
    for (let index = chat.length - 1; index >= 0; index--) {
        if (!chat[index].is_user && !chat[index].is_system) {
            return index;
        }
    }
    return -1;
}

(async function () {
    async function addExtensionSettings() {
        const template = await renderExtensionTemplateAsync(MODULE_NAME, 'settings');
        $('#extensions_settings2').append(template);

        $('#weylandProofreaderEnable').prop('checked', settings.enabled).on('input', function () {
            settings.enabled = !!$(this).prop('checked');
            proofreaderDebug(`Setting Enabled: ${settings.enabled}`);
            saveSettingsDebounced();
        });

        $('#weylandProofreaderSwipes').prop('checked', settings.correctSwipes).on('input', function () {
            settings.correctSwipes = !!$(this).prop('checked');
            proofreaderDebug(`Setting Correct Swipes: ${settings.correctSwipes}`);
            saveSettingsDebounced();
        });

        $('#weylandProofreaderDebug').prop('checked', settings.debug).on('input', function () {
            settings.debug = !!$(this).prop('checked');
            saveSettingsDebounced();
        });

        $('#weylandProofreaderModelId').val(settings.modelId).on('input', function () {
            settings.modelId = String($(this).val()).trim();
            proofreaderDebug(`Setting Model ID: ${settings.modelId || '(profile default)'}`);
            saveSettingsDebounced();
        });

        $('#weylandProofreaderMaxTokens').val(settings.maxTokens).on('input', function () {
            const value = parseInt(String($(this).val()), 10);
            settings.maxTokens = Number.isFinite(value) && value > 0 ? value : defaultSettings.maxTokens;
            saveSettingsDebounced();
        });

        $('#weylandProofreaderTimeout').val(settings.timeoutSeconds).on('input', function () {
            const value = parseInt(String($(this).val()), 10);
            settings.timeoutSeconds = Number.isFinite(value) && value > 0 ? value : defaultSettings.timeoutSeconds;
            saveSettingsDebounced();
        });

        $('#weylandProofreaderPrompt').val(settings.systemPrompt).on('input', function () {
            settings.systemPrompt = String($(this).val());
            saveSettingsDebounced();
        });

        $('#weylandProofreaderResetPromptButton').on('click', function () {
            settings.systemPrompt = DEFAULT_PROMPT;
            $('#weylandProofreaderPrompt').val(settings.systemPrompt);
            saveSettingsDebounced();
        });

        $('#weylandProofreaderTestButton').on('click', async function () {
            if (!settings.profileId) {
                toastr.warning('Select a Connection Profile first.');
                return;
            }
            const index = findLastAiMessageIndex();
            if (index < 0) {
                toastr.warning('No AI message found in this chat.');
                return;
            }
            toastr.info('Correcting last message...');
            await correctMessage(index);
        });

        try {
            ConnectionManagerRequestService.handleDropdown(
                '#weylandProofreaderProfile',
                settings.profileId,
                (profile) => {
                    settings.profileId = profile?.id ?? '';
                    proofreaderDebug(`Setting Connection Profile: ${settings.profileId}`);
                    saveSettingsDebounced();
                },
            );
        } catch (error) {
            proofreaderDebug('Connection Manager is not available', error);
        }
    }

    console.debug(`[${MODULE_NAME}] Initializing v${extensionVersion}`);

    getSettings();
    await addExtensionSettings();

    eventSource.on(event_types.CHARACTER_MESSAGE_RENDERED, (messageId) => correctMessage(messageId));
    eventSource.on(event_types.MESSAGE_SWIPED, (messageId) => {
        if (!settings.correctSwipes) return;
        return correctMessage(messageId);
    });
})();
