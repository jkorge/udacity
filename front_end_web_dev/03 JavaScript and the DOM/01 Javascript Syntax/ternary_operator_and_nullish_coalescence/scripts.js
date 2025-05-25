// User preference handling
let userTheme = localStorage.getItem('theme');
let defaultTheme = 'light';
let currentTheme;

if (userTheme) {
    currentTheme = userTheme;
} else {
    currentTheme = defaultTheme;
}

// Form validation
function validateForm(email) {

    let errorMessage = (!email) ? 'Email is required' : null;

    return errorMessage;
}


// API response handling
function processUserData(userData) {
    if (!userData) {
        return
    }

    const userName = (userData.userName?.trim()) ? userData.userName?.trim() : 'Friend';
    const notificationText = (userData.notifications == 1) ? `${userData.notifications} notification` : `${userData.notifications} notifications`;
    const greeting = `Hello, ${userName}. You have ${notificationText}`
   
}


// Dynamic class assignment
function getButtonClass(isActive) {

    let buttonClass = (isActive) ? 'btn-active' : 'btn-inactive';

    return buttonClass;
}