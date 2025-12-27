// Get HTML elements
const timerDisplay = document.getElementById('timerDisplay');
const minutesInput = document.getElementById('minutesInput');
const startBtn = document.getElementById('startBtn');
const pauseBtn = document.getElementById('pauseBtn');
const resetBtn = document.getElementById('resetBtn');
const status = document.getElementById('status');

// Timer variables
let timeRemaining = 0; // in seconds
let isRunning = false;
let timerInterval = null;

// Start button
startBtn.addEventListener('click', () => {
  if (!isRunning) {
    // Get minutes from input
    const minutes = parseInt(minutesInput.value) || 0;
    
    // Only start if time is > 0
    if (minutes > 0 || timeRemaining > 0) {
      // If starting fresh, convert minutes to seconds
      if (timeRemaining === 0) {
        timeRemaining = minutes * 60;
      }
      
      isRunning = true;
      status.textContent = 'Timer running...';
      minutesInput.disabled = true;
      
      // Run the countdown every 1000ms (1 second)
      timerInterval = setInterval(() => {
        timeRemaining--;
        updateDisplay();
        
        // When time reaches 0
        if (timeRemaining <= 0) {
          clearInterval(timerInterval);
          isRunning = false;
          status.textContent = '⏰ Time\'s up!';
          playNotification();
          minutesInput.disabled = false;
        }
      }, 1000);
    } else {
      status.textContent = 'Please enter minutes';
    }
  }
});

// Pause button
pauseBtn.addEventListener('click', () => {
  if (isRunning) {
    clearInterval(timerInterval);
    isRunning = false;
    status.textContent = 'Timer paused';
  }
});

// Reset button
resetBtn.addEventListener('click', () => {
  clearInterval(timerInterval);
  timeRemaining = 0;
  isRunning = false;
  minutesInput.value = '';
  minutesInput.disabled = false;
  updateDisplay();
  status.textContent = 'Ready to start';
});

// Update the timer display
function updateDisplay() {
  const minutes = Math.floor(timeRemaining / 60);
  const seconds = timeRemaining % 60;
  
  // Format: MM:SS (add leading zeros)
  const formattedTime = 
    String(minutes).padStart(2, '0') + ':' + 
    String(seconds).padStart(2, '0');
  
  timerDisplay.textContent = formattedTime;
}

// Play sound when timer ends
function playNotification() {
  // Create a beep sound using Web Audio API
  const audioContext = new (window.AudioContext || window.webkitAudioContext)();
  const oscillator = audioContext.createOscillator();
  const gain = audioContext.createGain();
  
  oscillator.connect(gain);
  gain.connect(audioContext.destination);
  
  oscillator.frequency.value = 800; // Frequency in Hz
  oscillator.type = 'sine';
  
  gain.gain.setValueAtTime(0.3, audioContext.currentTime);
  gain.gain.exponentialRampToValueAtTime(0.01, audioContext.currentTime + 0.5);
  
  oscillator.start(audioContext.currentTime);
  oscillator.stop(audioContext.currentTime + 0.5);
}

// Initialize display
updateDisplay();