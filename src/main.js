const { app, BrowserWindow } = require('electron');
const path = require('path');

let mainWindow;

// Create the browser window
function createWindow() {
  mainWindow = new BrowserWindow({
    width: 400,
    height: 600,
    webPreferences: {
      nodeIntegration: false,
      contextIsolation: true
    }
  });

  // Load the index.html
  mainWindow.loadFile('src/index.html');

  // Open DevTools (helpful for debugging)
  // mainWindow.webContents.openDevTools();
}

// App event listeners
app.on('ready', createWindow);

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') {
    app.quit();
  }
});

app.on('activate', () => {
  if (mainWindow === null) {
    createWindow();
  }
});