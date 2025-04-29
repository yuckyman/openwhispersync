document.addEventListener('DOMContentLoaded', () => {
    console.log("Read-along UI script loaded.");

    const chapterSelect = document.getElementById('chapter-select');
    const audioPlayer = document.getElementById('audio-player');
    const textDisplay = document.getElementById('text-display');
    
    // Get names from body data attributes set by Flask
    const filesName = document.body.dataset.filesName || 'default_files';
    const alignmentName = document.body.dataset.alignmentName || 'default_alignment';
    // Display name is mainly for the H1, already set in template
    // const bookDisplayName = document.querySelector('h1').textContent.split(':')[1].trim(); 

    let currentChapterSentences = [];
    let currentAlignmentData = [];
    let sentenceElements = [];

    function loadChapter(fName, aName, chapter) {
        console.log(`Loading chapter ${chapter} for files: ${fName}, alignment: ${aName}`);
        textDisplay.innerHTML = '<p>Loading...</p>'; 
        audioPlayer.pause(); 
        audioPlayer.currentTime = 0;
        sentenceElements = [];
        
        // Set audio source using filesName
        audioPlayer.src = `/audio/${fName}/${chapter}`;
        
        // Fetch alignment data and actual sentences using both names
        fetch(`/data/${fName}/${aName}/${chapter}`)
            .then(response => {
                if (!response.ok) {
                     return response.json().then(err => { throw new Error(err.error || `HTTP error! status: ${response.status}`)});
                }
                return response.json();
            })
            .then(data => {
                if (data.error) {
                    throw new Error(data.error);
                }
                currentAlignmentData = data.alignment;
                currentChapterSentences = data.sentences; 
                displaySentences(currentChapterSentences);
            })
            .catch(error => {
                console.error('Error loading chapter data:', error);
                textDisplay.innerHTML = `<p>Error loading chapter: ${error.message}</p>`;
                audioPlayer.src = '';
            });
    }

    function displaySentences(sentences) {
        textDisplay.innerHTML = '';
        sentenceElements = []; 

        if (!sentences || sentences.length === 0) {
            textDisplay.innerHTML = '<p>No sentences found for this chapter.</p>';
            return;
        }
        
        sentences.forEach((sentenceText, index) => {
             const p = document.createElement('p');
             p.textContent = sentenceText;
             p.dataset.sentenceIdx = index; 
             textDisplay.appendChild(p);
             sentenceElements.push(p);
        });
        
         console.log(`Displayed ${sentenceElements.length} sentences.`);
         mapAlignmentToSentences();
    }
    
    function mapAlignmentToSentences() {
        const sentenceIdxToElement = {};
        sentenceElements.forEach(p => {
            sentenceIdxToElement[p.dataset.sentenceIdx] = p;
        });

        currentAlignmentData.forEach(item => {
            const relativeIdx = item.sentence_idx;
            const pElement = sentenceIdxToElement[relativeIdx];
            
            if (pElement) {
                const currentStart = parseFloat(pElement.dataset.startTime || 'Infinity');
                const currentEnd = parseFloat(pElement.dataset.endTime || '-Infinity');
                
                pElement.dataset.startTime = Math.min(currentStart, item.start_time);
                pElement.dataset.endTime = Math.max(currentEnd, item.end_time);
            } else {
                console.warn(`Alignment item refers to sentence index ${relativeIdx}, but no corresponding element found.`);
            }
        });
        console.log("Mapped alignment times to sentence elements.");
    }

    function highlightSentence() {
        const currentTime = audioPlayer.currentTime;
        // console.log(`Highlight check at time: ${currentTime}`); // Optional: log time on every check
        let highlighted = false;

        sentenceElements.forEach(p => {
            if (p.dataset.startTime && p.dataset.endTime) {
                const start = parseFloat(p.dataset.startTime);
                const end = parseFloat(p.dataset.endTime);
    
                // Log the comparison for the first few sentences to avoid spamming console
                const sentenceIdx = parseInt(p.dataset.sentenceIdx, 10);
                if (sentenceIdx < 5) { // Log only for first 5 sentences
                    // console.log(`Sentence ${sentenceIdx}: Comparing time ${currentTime} with range [${start}, ${end})`);
                }

                if (currentTime >= start && currentTime < end) {
                    if (!p.classList.contains('highlight')) {
                        console.log(`HIGHLIGHTING Sentence ${sentenceIdx} at time ${currentTime} (Range: ${start}-${end})`);
                        p.classList.add('highlight');
                        p.scrollIntoView({ behavior: 'smooth', block: 'center' });
                    }
                    highlighted = true;
                } else {
                    if (p.classList.contains('highlight')) { // Log only when removing highlight
                        // console.log(`DE-HIGHLIGHTING Sentence ${sentenceIdx} at time ${currentTime} (Range: ${start}-${end})`);
                        p.classList.remove('highlight');
                    }
                }
            } else {
                 if (p.classList.contains('highlight')) { // Log only when removing highlight
                     // console.log(`DE-HIGHLIGHTING Sentence ${parseInt(p.dataset.sentenceIdx, 10)} (no time data)`);
                     p.classList.remove('highlight');
                 }
            }
        });
        // if (!highlighted) { // Optional: log if nothing ended up highlighted
        //     console.log(`No sentence highlighted at time: ${currentTime}`);
        // }
    }

    // --- Initialization --- //

    // TODO: Dynamically populate chapters based on available alignment files or EPUB structure
    // For now, still hardcoded in HTML

    // Event Listeners
    chapterSelect.addEventListener('change', (e) => {
        // Pass both names when loading chapter
        loadChapter(filesName, alignmentName, e.target.value);
    });

    audioPlayer.addEventListener('timeupdate', highlightSentence);
    audioPlayer.addEventListener('error', (e) => {
        console.error("Audio player error:", e);
        const errorMsg = document.createElement('p');
        errorMsg.textContent = "Error loading audio file. Check console and server logs for details.";
        errorMsg.style.color = 'red';
        // Avoid adding multiple error messages
        if (!textDisplay.querySelector('.audio-error')) {
             errorMsg.classList.add('audio-error');
             textDisplay.prepend(errorMsg);
        }
    });

    // Initial load for the default chapter, passing both names
    loadChapter(filesName, alignmentName, chapterSelect.value); 

});
