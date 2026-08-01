// ─── Deepfake Detector — Client-side interactions ───

document.addEventListener("DOMContentLoaded", () => {
    const dropZone = document.getElementById("drop-zone");
    const fileInput = document.getElementById("file-input");
    const submitBtn = document.getElementById("submit-btn");
    const form = document.getElementById("upload-form");
    const dropContent = document.getElementById("drop-content");
    const filePreview = document.getElementById("file-preview");
    const fileName = document.getElementById("file-name");
    const fileSize = document.getElementById("file-size");

    if (!dropZone || !fileInput) return;

    // ── File Selection ──
    fileInput.addEventListener("change", () => {
        if (fileInput.files.length > 0) {
            showFilePreview(fileInput.files[0]);
        }
    });

    // ── Drag & Drop ──
    ["dragenter", "dragover"].forEach(event => {
        dropZone.addEventListener(event, (e) => {
            e.preventDefault();
            dropZone.classList.add("drag-over");
        });
    });

    ["dragleave", "drop"].forEach(event => {
        dropZone.addEventListener(event, (e) => {
            e.preventDefault();
            dropZone.classList.remove("drag-over");
        });
    });

    dropZone.addEventListener("drop", (e) => {
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            fileInput.files = files;
            showFilePreview(files[0]);
        }
    });

    // ── Form Submission ──
    if (form) {
        form.addEventListener("submit", () => {
            const btnText = submitBtn.querySelector(".btn-text");
            const btnLoading = submitBtn.querySelector(".btn-loading");
            if (btnText) btnText.style.display = "none";
            if (btnLoading) btnLoading.style.display = "inline-flex";
            submitBtn.disabled = true;
        });
    }

    // ── Helpers ──
    function showFilePreview(file) {
        dropContent.style.display = "none";
        filePreview.style.display = "flex";
        fileName.textContent = file.name;
        fileSize.textContent = formatSize(file.size);
        submitBtn.disabled = false;
    }

    function formatSize(bytes) {
        if (bytes < 1024) return bytes + " B";
        if (bytes < 1048576) return (bytes / 1024).toFixed(1) + " KB";
        return (bytes / 1048576).toFixed(1) + " MB";
    }
});
