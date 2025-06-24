// Content script for YouTube integration
(function() {
  'use strict';

  let analyzeButton = null;

  function createAnalyzeButton() {
    if (analyzeButton) return;

    const button = document.createElement('button');
    button.innerHTML = `
      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
        <path d="M12 2L2 7v10c0 5.55 3.84 10 9 11 5.16-1 9-5.45 9-11V7l-10-5z"/>
        <path d="M9 12l2 2 4-4"/>
      </svg>
      <span style="margin-left: 4px;">Truthful</span>
    `;
    button.className = 'truthful-analyze-btn';
    button.title = 'Analyze video with Truthful AI Detector';
    
    button.addEventListener('click', handleAnalyzeClick);
    
    return button;
  }

  function insertAnalyzeButton() {
    // Try to find YouTube's action buttons container
    const containers = [
      '#top-level-buttons-computed', // YouTube's like/dislike button container
      '#menu-container #top-level-buttons',
      '.ytd-menu-renderer #top-level-buttons'
    ];

    for (const selector of containers) {
      const container = document.querySelector(selector);
      if (container && !container.querySelector('.truthful-analyze-btn')) {
        analyzeButton = createAnalyzeButton();
        container.appendChild(analyzeButton);
        break;
      }
    }
  }

  async function handleAnalyzeClick() {
    const videoUrl = window.location.href;
    const videoTitle = document.querySelector('h1.ytd-video-primary-info-renderer')?.textContent?.trim() || 'YouTube Video';

    analyzeButton.innerHTML = `
      <div class="spinner"></div>
      <span style="margin-left: 4px;">Analyzing...</span>
    `;
    analyzeButton.disabled = true;

    try {
      // Simulate analysis (in production, this would call your API)
      await new Promise(resolve => setTimeout(resolve, 3000));

      const likelihood = Math.floor(Math.random() * 100);
      let resultText = '';
      let resultColor = '';

      if (likelihood < 30) {
        resultText = `✅ ${likelihood}% AI`;
        resultColor = '#10b981';
      } else if (likelihood < 70) {
        resultText = `⚠️ ${likelihood}% AI`;
        resultColor = '#f59e0b';
      } else {
        resultText = `❌ ${likelihood}% AI`;
        resultColor = '#ef4444';
      }

      analyzeButton.innerHTML = `
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <path d="M12 2L2 7v10c0 5.55 3.84 10 9 11 5.16-1 9-5.45 9-11V7l-10-5z"/>
          <path d="M9 12l2 2 4-4"/>
        </svg>
        <span style="margin-left: 4px; color: ${resultColor};">${resultText}</span>
      `;

      // Reset button after 10 seconds
      setTimeout(() => {
        analyzeButton.innerHTML = `
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <path d="M12 2L2 7v10c0 5.55 3.84 10 9 11 5.16-1 9-5.45 9-11V7l-10-5z"/>
            <path d="M9 12l2 2 4-4"/>
          </svg>
          <span style="margin-left: 4px;">Truthful</span>
        `;
        analyzeButton.disabled = false;
      }, 10000);

    } catch (error) {
      analyzeButton.innerHTML = `
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <circle cx="12" cy="12" r="10"/>
          <line x1="15" y1="9" x2="9" y2="15"/>
          <line x1="9" y1="9" x2="15" y2="15"/>
        </svg>
        <span style="margin-left: 4px; color: #ef4444;">Error</span>
      `;
      
      setTimeout(() => {
        analyzeButton.innerHTML = `
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <path d="M12 2L2 7v10c0 5.55 3.84 10 9 11 5.16-1 9-5.45 9-11V7l-10-5z"/>
            <path d="M9 12l2 2 4-4"/>
          </svg>
          <span style="margin-left: 4px;">Truthful</span>
        `;
        analyzeButton.disabled = false;
      }, 5000);
    }
  }

  // Listen for messages from popup
  chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
    if (request.action === 'getVideoInfo') {
      const videoUrl = window.location.href;
      const videoTitle = document.querySelector('h1.ytd-video-primary-info-renderer')?.textContent?.trim();
      
      if (videoUrl.includes('youtube.com/watch') && videoTitle) {
        sendResponse({
          success: true,
          url: videoUrl,
          title: videoTitle
        });
      } else {
        sendResponse({ success: false });
      }
    }
  });

  // Initialize
  function init() {
    // Wait for YouTube to load
    const checkInterval = setInterval(() => {
      if (document.querySelector('#top-level-buttons-computed, #menu-container #top-level-buttons')) {
        insertAnalyzeButton();
        clearInterval(checkInterval);
      }
    }, 1000);

    // Also listen for navigation changes (YouTube is a SPA)
    let currentUrl = window.location.href;
    new MutationObserver(() => {
      if (window.location.href !== currentUrl) {
        currentUrl = window.location.href;
        analyzeButton = null;
        setTimeout(insertAnalyzeButton, 2000); // Wait for YouTube to render
      }
    }).observe(document.body, { childList: true, subtree: true });
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();