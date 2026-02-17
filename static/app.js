const dropZone = document.getElementById("dropZone");
const fileInput = document.getElementById("fileInput");
const syncButton = document.getElementById("syncButton");
const statusLogSync = document.getElementById("statusLogSync");
const deckNameInput = document.getElementById("deckName");
const textModelSelect = document.getElementById("textModelSelect");
const romanizedToggle = document.getElementById("romanizedToggle");

const audioDeckSelect = document.getElementById("audioDeckSelect") || document.getElementById("imageDeckSelect");
const audioModelSelect = document.getElementById("audioModelSelect");
const refreshDecksAudio = document.getElementById("refreshDecksAudio") || document.getElementById("refreshDecksImages");
const generateAudioButton = document.getElementById("generateAudio");
const statusLogMedia = document.getElementById("statusLogMedia");
const statusLogAudio = document.getElementById("statusLogAudio") || statusLogMedia;
const audioCoverageBadge = document.getElementById("audioCoverageBadge");
const mediaLogContainer = document.getElementById("mediaLogContainer");

const imageDeckSelect = document.getElementById("imageDeckSelect");
const imageModelSelect = document.getElementById("imageModelSelect");
const refreshDecksImages = document.getElementById("refreshDecksImages");
const generateImagesButton = document.getElementById("generateImages");
const statusLogImages = document.getElementById("statusLogImages") || statusLogMedia;
const imageCoverageBadge = document.getElementById("imageCoverageBadge");

const statusLogGallery = document.getElementById("statusLogGallery");
const galleryGrid = document.getElementById("galleryGrid");
const galleryPrevButton = document.getElementById("galleryPrev");
const galleryNextButton = document.getElementById("galleryNext");
const galleryPageInfo = document.getElementById("galleryPageInfo");
const browserSearchTerm = document.getElementById("browserSearchTerm");
const statusLogSearch = document.getElementById("statusLogSearch");
const chatDeckSelect = document.getElementById("chatDeckSelect");
const refreshDecksChat = document.getElementById("refreshDecksChat");
const chatModelSelect = document.getElementById("chatModelSelect");
const chatQuestion = document.getElementById("chatQuestion");
const chatAskButton = document.getElementById("chatAsk");
const chatThread = document.getElementById("chatThread");
const chatModel = document.getElementById("chatModel");

const tabButtons = document.querySelectorAll(".tab-button");
const tabPanels = document.querySelectorAll(".tab-panel");

let selectedFile = null;
let audioJobRunning = false;
let imageJobRunning = false;
let galleryPage = 1;
let galleryTotal = 0;
let chatHistory = [];
let searchDebounceTimer = null;
const SEARCH_DEBOUNCE_MS = 300;

function setStatus(element, message, append = false) {
    if (!element) return;
    if (append) {
        element.textContent += `\n${message}`;
    } else {
        element.textContent = message;
    }
}

function appendStatus(element, message) {
    if (!element) return;
    if (!element.textContent) {
        element.textContent = message;
    } else {
        element.textContent += `\n${message}`;
    }
}

function renderChatThread() {
    if (!chatThread) return;
    if (!chatHistory.length) {
        chatThread.innerHTML = '<div class="chat-message assistant">Select a deck and ask a question.</div>';
        return;
    }
    chatThread.innerHTML = chatHistory
        .map((msg) => {
            const roleClass = msg.role === "user" ? "user" : "assistant";
            if (msg.role === "assistant") {
                const safe = normalizeChatText(msg.content || "")
                    .replace(/&/g, "&amp;")
                    .replace(/</g, "&lt;")
                    .replace(/>/g, "&gt;")
                    .replace(/\n/g, "<br />")
                    .replace(/<br\s*\/?>\s*[-*]\s+/g, "<br />• ");
                return `<div class="chat-message assistant">${safe}</div>`;
            }
            const safe = (msg.content || "").replace(/</g, "&lt;").replace(/>/g, "&gt;");
            return `<div class="chat-message ${roleClass}">${safe}</div>`;
        })
        .join("");
    chatThread.scrollTop = chatThread.scrollHeight;
}

function normalizeChatText(text) {
    return (text || "")
        .replace(/\*\*/g, "")
        .replace(/__/g, "")
        .replace(/`/g, "")
        .replace(/^#+\s*/gm, "")
        .replace(/^>\s*/gm, "");
}

function showProgress(element, text) {
    const container = document.createElement("div");
    container.className = "progress";
    const spinner = document.createElement("div");
    spinner.className = "spinner";
    const label = document.createElement("span");
    label.innerHTML = text;
    container.appendChild(spinner);
    container.appendChild(label);
    element.parentNode.insertBefore(container, element);
    return container;
}

function updateProgress(container, text, etaText) {
    if (!container) return;
    const label = container.querySelector("span");
    if (!label) return;
    if (etaText) {
        label.innerHTML = `${text}<br><span class="eta-text">${etaText}</span>`;
    } else {
        label.textContent = text;
    }
}

function removeProgress(container) {
    if (container?.parentNode) {
        container.parentNode.removeChild(container);
    }
}

function streamJob(streamUrl, { onLog, onProgress, onDone, onError }) {
    const source = new EventSource(streamUrl);
    source.addEventListener("log", (event) => {
        try {
            const payload = JSON.parse(event.data);
            if (payload?.message && onLog) {
                onLog(payload.message);
            }
        } catch (error) {
            console.error("Failed to parse log event", error);
        }
    });
    source.addEventListener("progress", (event) => {
        try {
            const payload = JSON.parse(event.data);
            if (onProgress && Number.isFinite(payload.current) && Number.isFinite(payload.total)) {
                onProgress(payload.current, payload.total);
            }
        } catch (error) {
            console.error("Failed to parse progress event", error);
        }
    });
    source.addEventListener("done", (event) => {
        try {
            const payload = JSON.parse(event.data);
            if (onDone) onDone(payload);
        } catch (error) {
            if (onError) onError(error);
        } finally {
            source.close();
        }
    });
    source.onerror = (error) => {
        source.close();
        if (onError) onError(error);
    };
    return source;
}

function updateSyncButton() {
    syncButton.disabled = !selectedFile || !textModelSelect.value;
}

function handleFiles(files) {
    const file = files[0];
    if (!file) return;
    if (!file.name.toLowerCase().endsWith(".pdf")) {
        setStatus(statusLogSync, "Please choose a PDF file.");
        selectedFile = null;
        updateSyncButton();
        return;
    }
    selectedFile = file;
    setStatus(statusLogSync, `Ready to sync: ${file.name}`);
    updateSyncButton();
}

function switchTab(tabName, { updateHash = true } = {}) {
    tabButtons.forEach((button) => {
        button.classList.toggle("active", button.dataset.tab === tabName);
    });
    tabPanels.forEach((panel) => {
        panel.classList.toggle("active", panel.id === `tab-${tabName}`);
    });
    if (updateHash) {
        window.location.hash = tabName;
    }
}

tabButtons.forEach((button) => {
    button.addEventListener("click", () => switchTab(button.dataset.tab));
});

function loadTabFromHash() {
    const hash = window.location.hash.replace("#", "").trim();
    if (!hash) return;
    const exists = Array.from(tabButtons).some((button) => button.dataset.tab === hash);
    if (exists) {
        switchTab(hash, { updateHash: false });
    }
}

window.addEventListener("hashchange", loadTabFromHash);

dropZone.addEventListener("click", () => fileInput.click());

dropZone.addEventListener("dragover", (event) => {
    event.preventDefault();
    dropZone.classList.add("dragover");
});

dropZone.addEventListener("dragleave", () => {
    dropZone.classList.remove("dragover");
});

dropZone.addEventListener("drop", (event) => {
    event.preventDefault();
    dropZone.classList.remove("dragover");
    handleFiles(event.dataTransfer.files);
});

fileInput.addEventListener("change", (event) => handleFiles(event.target.files));

async function loadModels(kind, select, preferredValue, statusElement, onComplete) {
    if (!select) return;
    select.innerHTML = `<option value="">Loading ${kind} models...</option>`;
    select.disabled = true;
    try {
        const response = await fetch(`/api/models/${kind}`);
        const data = await response.json();
        if (data.ok && Array.isArray(data.models) && data.models.length) {
            select.innerHTML = "";
            data.models.forEach((id) => {
                const option = document.createElement("option");
                option.value = id;
                option.textContent = id;
                select.appendChild(option);
            });
            if (preferredValue && data.models.includes(preferredValue)) {
                select.value = preferredValue;
            }
        } else {
            select.innerHTML = `<option value="">No ${kind} models</option>`;
            if (statusElement) {
                setStatus(statusElement, data.message || `No ${kind} models available.`);
            }
        }
    } catch (error) {
        select.innerHTML = `<option value="">Unavailable</option>`;
        if (statusElement) {
            setStatus(statusElement, `❌ Failed to load ${kind} models: ${error}`);
        }
    } finally {
        select.disabled = false;
        if (typeof onComplete === "function") onComplete();
    }
}

function populateDeckSelect(select, decks) {
    if (!select) return;
    select.innerHTML = "";
    decks.forEach((deck) => {
        const option = document.createElement("option");
        option.value = deck;
        option.textContent = deck;
        select.appendChild(option);
    });
    if (!select.value && decks.length) {
        select.value = decks[0];
    }
}

async function loadDecks() {
    const selects = Array.from(new Set([audioDeckSelect, imageDeckSelect, chatDeckSelect]));
    selects.forEach((select) => {
        if (select) {
            select.innerHTML = '<option value="">Loading decks...</option>';
            select.disabled = true;
        }
    });
    setStatus(statusLogAudio, "Fetching decks...");
    setStatus(statusLogImages, "Fetching decks...");
    setStatus(statusLogGallery, "Fetching decks...");
    if (chatThread) {
        chatHistory = [];
        renderChatThread();
    }
    try {
        const response = await fetch("/api/decks");
        const data = await response.json();
        if (data.ok && Array.isArray(data.decks) && data.decks.length) {
            selects.forEach((select) => populateDeckSelect(select, data.decks));
            setStatus(statusLogAudio, "Select a deck and model to begin.");
            setStatus(statusLogImages, "Select a deck and model to begin.");
            setStatus(statusLogGallery, "Select a deck to browse cards.");
            if (chatThread) {
                chatHistory = [];
                renderChatThread();
            }
            updateAudioCoverage();
            updateImageCoverage();
            if (imageDeckSelect?.value) {
                galleryPage = 1;
                loadGallery();
            }
        } else {
            const message = data.message || "No decks found.";
            selects.forEach((select) => {
                if (select) {
                    select.innerHTML = `<option value="">${message}</option>`;
                }
            });
            setStatus(statusLogAudio, message);
            setStatus(statusLogImages, message);
            setStatus(statusLogGallery, message);
            if (chatThread) {
                chatHistory = [{ role: "assistant", content: message }];
                renderChatThread();
            }
        }
    } catch (error) {
        selects.forEach((select) => {
            if (select) {
                select.innerHTML = '<option value="">Error loading decks</option>';
            }
        });
        setStatus(statusLogAudio, `❌ Failed to fetch decks: ${error}`);
        setStatus(statusLogImages, `❌ Failed to fetch decks: ${error}`);
        setStatus(statusLogGallery, `❌ Failed to fetch decks: ${error}`);
        if (chatThread) {
            chatHistory = [{ role: "assistant", content: `❌ Failed to fetch decks: ${error}` }];
            renderChatThread();
        }
    } finally {
        selects.forEach((select) => {
            if (select) select.disabled = false;
        });
        updateAudioControls();
        updateImageControls();
        updateGalleryControls();
        updateChatControls();
    }
}

function updateAudioControls() {
    const hasDeck = Boolean(audioDeckSelect?.value);
    const hasModel = audioModelSelect ? Boolean(audioModelSelect?.value) : true;
    generateAudioButton.disabled = audioJobRunning || !(hasDeck && hasModel);
}

function updateImageControls() {
    const hasDeck = Boolean(imageDeckSelect?.value);
    const hasImageModel = Boolean(imageModelSelect?.value);
    generateImagesButton.disabled = imageJobRunning || !(hasDeck && hasImageModel);
}

function updateGalleryControls() {
    if (!galleryPrevButton || !galleryNextButton || !galleryPageInfo) return;
    const totalPages = Math.max(1, Math.ceil(galleryTotal / getGalleryPageSize()));
    galleryPrevButton.disabled = galleryPage <= 1;
    galleryNextButton.disabled = galleryPage >= totalPages;
    galleryPageInfo.textContent = `Page ${galleryPage} of ${totalPages}`;
}

function updateChatControls() {
    if (!chatAskButton) return;
    const hasDeck = Boolean(chatDeckSelect?.value);
    const hasModel = Boolean(chatModelSelect?.value);
    const hasQuestion = Boolean(chatQuestion?.value?.trim());
    chatAskButton.disabled = !(hasDeck && hasModel && hasQuestion);
}

syncButton.addEventListener("click", async () => {
    if (!selectedFile || !textModelSelect.value) {
        return;
    }

    setStatus(statusLogSync, "Uploading and syncing deck...");
    const progressNode = showProgress(statusLogSync, "Processing PDF...");
    syncButton.disabled = true;

    const formData = new FormData();
    formData.append("file", selectedFile);
    formData.append("deck", deckNameInput.value);
    formData.append("model", textModelSelect.value);
    formData.append("romanized", romanizedToggle.checked ? "true" : "false");

    try {
        const response = await fetch("/sync", {
            method: "POST",
            body: formData,
        });
        const data = await response.json();
        if (data.ok) {
            updateProgress(progressNode, "Sync complete!", data.eta_text);
            const processed = data.items_processed !== undefined ? ` (cards processed: ${data.items_processed})` : "";
            setStatus(
                statusLogSync,
                `✅ ${data.message}${processed}\n\n${data.stdout}`
            );
            if (data.stderr) {
                setStatus(statusLogSync, data.stderr, true);
            }
        } else {
            setStatus(
                statusLogSync,
                `⚠️ ${data.message}\n\n${data.stdout || ""}\n${data.stderr || ""}`
            );
        }
    } catch (error) {
        setStatus(statusLogSync, `❌ Request failed: ${error}`);
    } finally {
        syncButton.disabled = false;
        removeProgress(progressNode);
    }
});

async function generateAudio() {
    const deck = audioDeckSelect.value;
    const model = (audioModelSelect && audioModelSelect.value) ? audioModelSelect.value : "gpt-4o-mini-tts";
    const workers = 10;
    if (!deck || !model) {
        setStatus(statusLogAudio, "Please select a deck.");
        return;
    }
    if (mediaLogContainer) {
        mediaLogContainer.classList.remove("is-hidden");
    }

    setStatus(statusLogAudio, `Starting audio job for "${deck}"...`);
    const progressNode = showProgress(statusLogAudio, "Generating audio...");
    audioJobRunning = true;
    updateAudioControls();

    try {
        const response = await fetch("/api/jobs/audio", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ deck, model, workers }),
        });
        const data = await response.json();
        if (!data.ok) {
            setStatus(statusLogAudio, `⚠️ ${data.message}`);
            audioJobRunning = false;
            updateAudioControls();
            return;
        }

        setStatus(statusLogAudio, "");
        streamJob(data.stream_url, {
            onLog: (message) => appendStatus(statusLogAudio, message),
            onProgress: (current, total) => {
                updateProgress(progressNode, `Progress: ${current}/${total}`);
            },
            onDone: (payload) => {
                const ok = payload?.ok;
                const summary = payload?.summary;
                if (summary?.candidates !== undefined) {
                    appendStatus(
                        statusLogAudio,
                        `Summary: ${summary.added || 0} added, ${summary.skipped || 0} skipped, ${summary.failed || 0} failed.`
                    );
                }
                updateProgress(progressNode, ok ? "Audio complete!" : "Audio failed.");
                if (!ok) {
                    appendStatus(statusLogAudio, "⚠️ Audio generation failed.");
                }
                removeProgress(progressNode);
                audioJobRunning = false;
                updateAudioControls();
                updateAudioCoverage();
            },
            onError: (error) => {
                appendStatus(statusLogAudio, `⚠️ Stream error: ${error}`);
                updateProgress(progressNode, "Audio stream closed.");
                removeProgress(progressNode);
                audioJobRunning = false;
                updateAudioControls();
            },
        });
    } catch (error) {
        setStatus(statusLogAudio, `❌ Request failed: ${error}`);
        audioJobRunning = false;
        updateAudioControls();
    }
}

generateAudioButton.addEventListener("click", generateAudio);

async function generateImages() {
    const deck = imageDeckSelect.value;
    const imageModel = imageModelSelect.value;
    const workers = 3;

    if (!deck || !imageModel) {
        setStatus(statusLogImages, "Please select a deck and image model.");
        return;
    }
    if (mediaLogContainer) {
        mediaLogContainer.classList.remove("is-hidden");
    }

    setStatus(statusLogImages, `Starting image job for "${deck}"...`);
    const progressNode = showProgress(statusLogImages, "Generating images...");
    imageJobRunning = true;
    updateImageControls();

    const payload = {
        deck,
        image_model: imageModel,
        skip_gating: false,
        workers: Number(workers),
    };

    try {
        const response = await fetch("/api/jobs/images", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload),
        });
        const data = await response.json();
        if (!data.ok) {
            setStatus(statusLogImages, `⚠️ ${data.message}`);
            imageJobRunning = false;
            updateImageControls();
            return;
        }

        setStatus(statusLogImages, "");
        streamJob(data.stream_url, {
            onLog: (message) => appendStatus(statusLogImages, message),
            onProgress: (current, total) => {
                updateProgress(progressNode, `Progress: ${current}/${total}`);
            },
            onDone: (payload) => {
                const ok = payload?.ok;
                const summary = payload?.summary;
                if (summary?.candidates !== undefined) {
                    appendStatus(
                        statusLogImages,
                        `Summary: ${summary.added || 0} added, ${summary.skipped || 0} skipped, ${summary.failed || 0} failed.`
                    );
                }
                updateProgress(progressNode, ok ? "Images complete!" : "Image job failed.");
                if (!ok) {
                    appendStatus(statusLogImages, "⚠️ Image generation failed.");
                }
                removeProgress(progressNode);
                imageJobRunning = false;
                updateImageControls();
                updateImageCoverage();
            },
            onError: (error) => {
                appendStatus(statusLogImages, `⚠️ Stream error: ${error}`);
                updateProgress(progressNode, "Image stream closed.");
                removeProgress(progressNode);
                imageJobRunning = false;
                updateImageControls();
            },
        });
    } catch (error) {
        setStatus(statusLogImages, `❌ Request failed: ${error}`);
        imageJobRunning = false;
        updateImageControls();
    }
}

generateImagesButton.addEventListener("click", generateImages);

if (refreshDecksAudio === refreshDecksImages) {
    refreshDecksImages.addEventListener("click", loadDecks);
} else {
    refreshDecksAudio.addEventListener("click", loadDecks);
    refreshDecksImages.addEventListener("click", loadDecks);
}
if (refreshDecksChat) {
    refreshDecksChat.addEventListener("click", loadDecks);
}

audioDeckSelect.addEventListener("change", updateAudioControls);
if (audioModelSelect) {
    audioModelSelect.addEventListener("change", updateAudioControls);
}
audioDeckSelect.addEventListener("change", updateAudioCoverage);

imageDeckSelect.addEventListener("change", updateImageControls);
imageModelSelect.addEventListener("change", updateImageControls);
imageDeckSelect.addEventListener("change", updateImageCoverage);
imageDeckSelect.addEventListener("change", () => {
    updateGalleryControls();
    if (imageDeckSelect.value) {
        galleryPage = 1;
        loadGallery();
    } else if (galleryGrid) {
        galleryGrid.classList.add("empty");
        galleryGrid.innerHTML = "<p>No deck selected.</p>";
        setStatus(statusLogSearch, "");
    }
});

if (galleryPrevButton) {
    galleryPrevButton.addEventListener("click", () => {
        if (galleryPage > 1) {
            galleryPage -= 1;
            loadGallery();
        }
    });
}

if (galleryNextButton) {
    galleryNextButton.addEventListener("click", () => {
        const totalPages = Math.max(1, Math.ceil(galleryTotal / getGalleryPageSize()));
        if (galleryPage < totalPages) {
            galleryPage += 1;
            loadGallery();
        }
    });
}

if (browserSearchTerm) {
    browserSearchTerm.addEventListener("input", queueSearch);
    browserSearchTerm.addEventListener("keydown", (event) => {
        if (event.key === "Enter") {
            event.preventDefault();
            if (searchDebounceTimer) {
                clearTimeout(searchDebounceTimer);
                searchDebounceTimer = null;
            }
            applyBrowserFilter();
        }
    });
}
if (chatDeckSelect) {
    chatDeckSelect.addEventListener("change", () => {
        updateChatControls();
        chatHistory = [];
        renderChatThread();
    });
}
if (chatModelSelect) {
    chatModelSelect.addEventListener("change", updateChatControls);
}
if (chatQuestion) {
    chatQuestion.addEventListener("input", updateChatControls);
    chatQuestion.addEventListener("keydown", (event) => {
        if (event.key === "Enter" && !event.shiftKey) {
            event.preventDefault();
            if (!chatAskButton?.disabled) {
                askDeck();
            }
        }
    });
}
if (chatAskButton) {
    chatAskButton.addEventListener("click", askDeck);
}

function renderGallery(items) {
    if (!galleryGrid) return;
    if (!items.length) {
        galleryGrid.classList.add("empty");
        galleryGrid.innerHTML = "<p>No cards found for this deck/filter.</p>";
        return;
    }
    galleryGrid.classList.remove("empty");
    galleryGrid.innerHTML = items
        .map((item) => {
            const frontText = item.front_text || "(No Front text)";
            const backText = item.back_text || "(No Back text)";
            const visual = item.has_image && item.image_url
                ? `<img src="${item.image_url}" alt="${frontText}" loading="lazy" />`
                : `<div class="image-placeholder">No image</div>`;
            return `
            <div class="image-card" data-sound="${item.sound_filename || ""}">
                ${visual}
                <div class="caption">Front: ${frontText}<br />Back: ${backText}</div>
            </div>`;
        })
        .join("");
}

async function playAudioFromGallery(cardEl) {
    const filename = cardEl?.dataset?.sound;
    if (!filename) return;
    try {
        const response = await fetch(`/api/media?filename=${encodeURIComponent(filename)}`);
        if (!response.ok) {
            throw new Error("Audio not found.");
        }
        const buffer = await response.arrayBuffer();
        const blob = new Blob([buffer], { type: response.headers.get("Content-Type") || "audio/mpeg" });
        const url = URL.createObjectURL(blob);
        const audio = new Audio(url);
        audio.addEventListener("ended", () => URL.revokeObjectURL(url));
        audio.play();
    } catch (error) {
        console.error("Failed to play audio", error);
    }
}

if (galleryGrid) {
    galleryGrid.addEventListener("click", (event) => {
        const card = event.target.closest(".image-card");
        if (!card) return;
        playAudioFromGallery(card);
    });
}

async function loadGallery() {
    const deck = imageDeckSelect?.value;
    if (!deck) {
        setStatus(statusLogGallery, "Select a deck to browse cards.");
        return;
    }
    updateGalleryControls();
    const term = (browserSearchTerm?.value || "").trim();
    setStatus(statusLogGallery, `Loading cards for "${deck}"${term ? ` (filter: "${term}")` : ""}...`);
    try {
        const response = await fetch(
            `/api/deck-gallery?deck=${encodeURIComponent(deck)}&term=${encodeURIComponent(term)}&page=${galleryPage}&page_size=${getGalleryPageSize()}`
        );
        const data = await response.json();
        if (!response.ok || !data.ok) {
            throw new Error(data.message || "Failed to fetch deck gallery.");
        }
        galleryTotal = data.total ?? 0;
        renderGallery(data.items || []);
        updateGalleryControls();
        if ((data.items || []).length) {
            setStatus(statusLogGallery, "");
            setStatus(statusLogSearch, "");
        } else {
            setStatus(statusLogGallery, `No cards found for "${deck}"${term ? ` with "${term}"` : ""}.`);
        }
    } catch (error) {
        setStatus(statusLogGallery, `❌ Failed to load gallery: ${error}`);
        if (galleryGrid) {
            galleryGrid.classList.add("empty");
            galleryGrid.innerHTML = "<p>Unable to load gallery.</p>";
        }
    }
}

function getGalleryPageSize() {
    return 12;
}

function applyBrowserFilter() {
    galleryPage = 1;
    loadGallery();
}

function queueSearch() {
    const term = browserSearchTerm?.value?.trim();
    if (searchDebounceTimer) {
        clearTimeout(searchDebounceTimer);
    }
    searchDebounceTimer = setTimeout(() => {
        searchDebounceTimer = null;
        if (!term) {
            applyBrowserFilter();
            return;
        }
        applyBrowserFilter();
    }, SEARCH_DEBOUNCE_MS);
}

loadDecks();
loadModels("text", textModelSelect, "gpt-4.1-mini", statusLogSync, updateSyncButton);
if (audioModelSelect) {
    loadModels("audio", audioModelSelect, "gpt-4o-mini-tts", statusLogAudio, updateAudioControls);
}
loadModels("image", imageModelSelect, "gpt-image-1", statusLogImages, updateImageControls);
loadModels("text", chatModelSelect, "gpt-5.2", null, updateChatControls);

textModelSelect.addEventListener("change", updateSyncButton);
updateSyncButton();
updateAudioControls();
updateImageControls();
updateGalleryControls();
updateAudioCoverage();
updateImageCoverage();
loadTabFromHash();
updateChatControls();

async function updateAudioCoverage() {
    if (!audioCoverageBadge || !audioDeckSelect?.value) {
        if (audioCoverageBadge) audioCoverageBadge.textContent = "";
        return;
    }
    const deck = audioDeckSelect.value;
    try {
        const response = await fetch(`/api/deck-audio-stats?deck=${encodeURIComponent(deck)}`);
        const data = await response.json();
        if (!response.ok || !data.ok) {
            throw new Error(data.message || "Failed to fetch audio stats.");
        }
        const total = data.total ?? 0;
        const withAudio = data.with_audio ?? 0;
        const coverage = data.coverage ?? 0;
        audioCoverageBadge.textContent = `Audio: ${withAudio}/${total} (${coverage}%)`;
    } catch (error) {
        audioCoverageBadge.textContent = "Audio: unavailable";
    }
}

async function askDeck() {
    const deck = chatDeckSelect?.value;
    const question = chatQuestion?.value?.trim();
    if (!deck || !question) {
        if (chatThread) {
            chatHistory = [{ role: "assistant", content: "Select a deck and ask a question." }];
            renderChatThread();
        }
        if (chatModel) chatModel.textContent = "";
        return;
    }
    if (chatQuestion) {
        chatQuestion.value = "";
    }
    chatHistory.push({ role: "user", content: question });
    const assistantIndex = chatHistory.push({ role: "assistant", content: "" }) - 1;
    renderChatThread();
    if (chatAskButton) chatAskButton.disabled = true;
    try {
        const response = await fetch("/api/deck-chat", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                deck,
                question,
                model: chatModelSelect?.value || null,
                history: chatHistory.slice(0, -1),
                stream: true,
            }),
        });
        if (!response.ok) {
            const data = await response.json();
            throw new Error(data.message || "Failed to get answer.");
        }
        const reader = response.body?.getReader();
        if (!reader) {
            const data = await response.json();
            chatHistory[assistantIndex].content = data.answer || "No answer.";
            renderChatThread();
            if (chatModel && data.model) {
                chatModel.textContent = `Model: ${data.model}`;
            }
            return;
        }
        if (chatModel) chatModel.textContent = "Model: streaming";
        const decoder = new TextDecoder();
        let buffer = "";
        while (true) {
            const { value, done } = await reader.read();
            if (done) break;
            const chunk = decoder.decode(value, { stream: true });
            if (chunk) {
                if (chunk.startsWith(buffer)) {
                    buffer = chunk;
                } else {
                    buffer += chunk;
                }
            }
            chatHistory[assistantIndex].content = buffer;
            renderChatThread();
        }
        if (chatModel) {
            const modelFromResponse = response.headers.get("X-Chat-Model");
            chatModel.textContent = modelFromResponse ? `Model: ${modelFromResponse}` : "Model: unknown";
        }
    } catch (error) {
        chatHistory[assistantIndex].content = `❌ ${error}`;
        renderChatThread();
        if (chatModel) chatModel.textContent = "";
    } finally {
        updateChatControls();
    }
}

async function updateImageCoverage() {
    if (!imageCoverageBadge || !imageDeckSelect?.value) {
        if (imageCoverageBadge) imageCoverageBadge.textContent = "";
        return;
    }
    const deck = imageDeckSelect.value;
    try {
        const response = await fetch(`/api/deck-image-stats?deck=${encodeURIComponent(deck)}`);
        const data = await response.json();
        if (!response.ok || !data.ok) {
            throw new Error(data.message || "Failed to fetch image stats.");
        }
        const total = data.total ?? 0;
        const withImages = data.with_images ?? 0;
        const coverage = data.coverage ?? 0;
        imageCoverageBadge.textContent = `Images: ${withImages}/${total} (${coverage}%)`;
    } catch (error) {
        imageCoverageBadge.textContent = "Images: unavailable";
    }
}
