// User preference handling
let userTheme = localStorage.getItem('theme');
let defaultTheme = 'light';
let currentTheme;

currentTheme = userTheme ?? defaultTheme

// Form validation
function validateForm(email) {
    let errorMessage = '';  
    errorMessage = (!email) ? 'Email is required' : null
    return errorMessage;
}


// API response handling
function processUserData(userData) {
    if (!userData) {
        return
    }
   
    const userName = userData.userName?.trim() ?? 'Friend'
    const notificationText = `${userData.notifications} ${(userData.notifications > 1 || userData.notifications == 0) ? 'notifications' : 'notification'}`
    const greeting = `Hello, ${userName}. You have ${notificationText}`  
}


// Dynamic class assignment
function getButtonClass(isActive) {
    let buttonClass;  
    buttonClass = (isActive) ? 'btn-active' : 'btn-inactive'
    return buttonClass;
}





