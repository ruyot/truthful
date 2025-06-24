document.addEventListener('DOMContentLoaded', function() {
  // Tab switching
  const tabButtons = document.querySelectorAll('.tab-button');
  const tabContents = document.querySelectorAll('.tab-content');

  tabButtons.forEach(button => {
    button.addEventListener('click', () => {
      tabButtons.forEach(b => b.classList.remove('active'));
      tabContents.forEach(c => c.classList.remove('active'));
      
      button.classList.add('active');
      document.getElementById(`${button.dataset.tab}-tab`).classList.add('active');
    });
  });

  // File upload handling
  const uploadArea = document.getElementById('upload-area');
  const fileInput = document.getElementById('file-input');
  const urlInput = document.getElementById('url-input');
  const analyzeUrlBtn = document.getElementById('analyze-url-btn');
  const status = document.getElementById('status');
  const statusText = document.getElementById('status-text');

  uploadArea.addEventListener('click', () => fileInput.click());
  uploadArea.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadArea.classList.add('dragover');
  });
  uploadArea.addEventListener('dragleave', () => {
    uploadArea.classList.remove('dragover');
  });
  uploadArea.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadArea.classList.remove('dragover');
    const files = e.dataTransfer.files;
    handleFileUpload(files[0]);
  });

  fileInput.addEventListener('change', (e) => {
    handleFileUpload(e.target.files[0]);
  });

  analyzeUrlBtn.addEventListener('click', () => {
    const url = urlInput.value.trim();
    if (url) {
      analyzeVideo(null, url);
    }
  });

  // YouTube integration
  const getCurrentVideoBtn = document.getElementById('get-current-video');
  const currentVideoDiv = document.getElementById('current-video');
  const videoTitle = document.getElementById('video-title');
  const videoUrl = document.getElementById('video-url');
  const analyzeCurrentBtn = document.getElementById('analyze-current-btn');

  getCurrentVideoBtn.addEventListener('click', async () => {
    try {
      const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
      
      if (!tab.url.includes('youtube.com/watch')) {
        showStatus('Please navigate to a YouTube video first', 'error');
        return;
      }

      // Get video info from content script
      const response = await chrome.tabs.sendMessage(tab.id, { action: 'getVideoInfo' });
      
      if (response && response.success) {
        videoTitle.textContent = response.title || 'YouTube Video';
        videoUrl.textContent = response.url;
        currentVideoDiv.classList.remove('hidden');
        
        analyzeCurrentBtn.onclick = () => analyzeVideo(null, response.url);
      } else {
        showStatus('Could not get video information', 'error');
      }
    } catch (error) {
      showStatus('Please navigate to YouTube and refresh the page', 'error');
    }
  });

  function handleFileUpload(file) {
    if (!file) return;
    
    if (file.size > 50 * 1024 * 1024) { // 50MB limit
      showStatus('File too large. Maximum size is 50MB', 'error');
      return;
    }

    analyzeVideo(file, null);
  }

  async function analyzeVideo(file, url) {
    showStatus('Analyzing video...', 'loading');

    try {
      // In a real implementation, this would call your FastAPI backend
      // For now, we'll simulate the analysis
      await new Promise(resolve => setTimeout(resolve, 3000));

      const mockResult = {
        overall_likelihood: Math.floor(Math.random() * 100),
        analysis_results: {
          total_frames: Math.floor(Math.random() * 1000) + 100,
          processing_time: Math.floor(Math.random() * 60) + 30
        }
      };

      const likelihood = mockResult.overall_likelihood;
      let resultText = '';
      let resultClass = '';

      if (likelihood < 30) {
        resultText = `✅ Likely Authentic (${likelihood}% AI)`;
        resultClass = 'success';
      } else if (likelihood < 70) {
        resultText = `⚠️ Uncertain (${likelihood}% AI)`;
        resultClass = 'loading';
      } else {
        resultText = `❌ Likely AI Generated (${likelihood}% AI)`;
        resultClass = 'error';
      }

      showStatus(resultText, resultClass);

      // Store result in extension storage
      chrome.storage.local.set({
        lastAnalysis: {
          timestamp: Date.now(),
          result: mockResult,
          source: file ? file.name : url
        }
      });

    } catch (error) {
      showStatus('Analysis failed. Please try again.', 'error');
    }
  }

  function showStatus(message, type) {
    statusText.textContent = message;
    status.className = `status ${type}`;
    status.classList.remove('hidden');

    if (type !== 'loading') {
      setTimeout(() => {
        status.classList.add('hidden');
      }, 5000);
    }
  }

  // Load last analysis result if available
  chrome.storage.local.get(['lastAnalysis'], (result) => {
    if (result.lastAnalysis && Date.now() - result.lastAnalysis.timestamp < 300000) { // 5 minutes
      const analysis = result.lastAnalysis.result;
      const likelihood = analysis.overall_likelihood;
      
      let resultText = '';
      let resultClass = '';

      if (likelihood < 30) {
        resultText = `✅ Last Result: Likely Authentic (${likelihood}% AI)`;
        resultClass = 'success';
      } else if (likelihood < 70) {
        resultText = `⚠️ Last Result: Uncertain (${likelihood}% AI)`;
        resultClass = 'loading';
      } else {
        resultText = `❌ Last Result: Likely AI Generated (${likelihood}% AI)`;
        resultClass = 'error';
      }

      showStatus(resultText, resultClass);
    }
  });
});