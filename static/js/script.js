
const resumeFiles = document.getElementById("resumeFiles");
const fileList = document.getElementById("fileList");
const dropZone = document.getElementById("dropZone");
const screeningForm = document.getElementById("screeningForm");
const screenButton = document.getElementById("screenButton");
const buttonText = document.getElementById("buttonText");
const jobDescription = document.getElementById("jobDescription");
const jobCount = document.getElementById("jobCount");

// Keep selected files between file-picker operations.
const selectedFiles = new Map();

function fileKey(file) {
    return `${file.name}::${file.size}::${file.lastModified}`;
}

function formatFileSize(bytes) {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) {
        return `${(bytes / 1024).toFixed(1)} KB`;
    }
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function isSupportedFile(file) {
    const name = file.name.toLowerCase();
    return name.endsWith(".pdf") || name.endsWith(".txt");
}

function syncInputFiles() {
    const transfer = new DataTransfer();

    selectedFiles.forEach((file) => {
        transfer.items.add(file);
    });

    // This is the complete list submitted to Flask.
    resumeFiles.files = transfer.files;
}

function renderFiles() {
    fileList.replaceChildren();

    const count = selectedFiles.size;
    const countLabel = document.createElement("div");
    countLabel.className = "file-count";
    countLabel.textContent =
        `${count} resume${count === 1 ? "" : "s"} selected`;

    fileList.appendChild(countLabel);

    selectedFiles.forEach((file, key) => {
        const item = document.createElement("div");
        item.className = "file-item";

        const name = document.createElement("span");
        name.className = "file-item-name";
        name.textContent = file.name;

        const meta = document.createElement("span");
        meta.className = "file-item-type";
        meta.textContent = formatFileSize(file.size);

        const remove = document.createElement("button");
        remove.type = "button";
        remove.className = "remove-file";
        remove.textContent = "×";
        remove.setAttribute("aria-label", `Remove ${file.name}`);

        remove.addEventListener("click", () => {
            selectedFiles.delete(key);
            syncInputFiles();
            renderFiles();
        });

        item.append(name, meta, remove);
        fileList.appendChild(item);
    });
}

function addFiles(fileCollection) {
    const files = Array.from(fileCollection || []);
    const rejected = [];

    files.forEach((file) => {
        if (!isSupportedFile(file)) {
            rejected.push(file.name);
            return;
        }

        // Avoid adding the same file more than once.
        selectedFiles.set(fileKey(file), file);
    });

    syncInputFiles();
    renderFiles();

    if (rejected.length) {
        alert(
            "Only PDF and TXT resumes are supported.\n\n" +
            rejected.join("\n")
        );
    }
}

// Selecting files in the file picker adds them to the existing list.
if (resumeFiles) {
    resumeFiles.addEventListener("change", () => {
        addFiles(resumeFiles.files);
    });
}

// Drag-and-drop support.
if (dropZone) {
    ["dragenter", "dragover"].forEach((eventName) => {
        dropZone.addEventListener(eventName, (event) => {
            event.preventDefault();
            dropZone.classList.add("dragging");
        });
    });

    ["dragleave", "drop"].forEach((eventName) => {
        dropZone.addEventListener(eventName, (event) => {
            event.preventDefault();
            dropZone.classList.remove("dragging");
        });
    });

    dropZone.addEventListener("drop", (event) => {
        addFiles(event.dataTransfer.files);
    });
}

// Job-description character count.
if (jobDescription && jobCount) {
    const updateJobCount = () => {
        jobCount.textContent =
            `${jobDescription.value.length} characters`;
    };

    jobDescription.addEventListener("input", updateJobCount);
    updateJobCount();
}

// Prevent submission if no valid resume is selected.
if (screeningForm) {
    screeningForm.addEventListener("submit", (event) => {
        syncInputFiles();

        if (selectedFiles.size === 0) {
            event.preventDefault();
            alert("Please select at least one PDF or TXT resume.");
            return;
        }

        screenButton.classList.add("loading");
        buttonText.textContent =
            `Screening ${selectedFiles.size} resume(s)...`;
    });
}
