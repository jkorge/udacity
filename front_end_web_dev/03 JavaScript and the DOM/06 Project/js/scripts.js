// Globals for form validation
const badChars = /[^a-zA-Z0-9@._-]/;
const validEmail = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
const maxMsgLen = 300;
const errors = {
    'email_empty': 'Email cannot be empty',
    'email_chars': 'Invalid characters in email',
    'email_formt': 'Invalid email format',

    'msg_empty': 'Message cannot be empty',
    'msg_nchar': 'Message cannot be longer than 300 characters',
    'msg_chars': 'Invalid characters in message'
}

const loadJSONFile = async (filePath) => {
    try {
        const response = await fetch(filePath);
        const jsonData = await response.json();
        return jsonData;
    } catch (error) {
        console.log(`Failed to load file ${filePath}: ${error}`);
        return null;
    }
}

const populateAboutMe = async () => {

    // Load about me data
    const data = await loadJSONFile('./data/aboutMeData.json');

    // Paragraph for about me text
    const p = document.createElement('p');
    p.textContent = data.aboutMe;

    // Headshot and it's container
    const div = document.createElement('div');
    div.classList.add('headshotContainer');

    const img = document.createElement('img');
    img.src = data.headshot;
    img.alt = 'Headshot of the developer';
    div.append(img);

    // Create a document fragment with the new elements and update the DOM
    const aboutMe = document.querySelector('#aboutMe');
    const fragment = document.createDocumentFragment();
    fragment.append(p, div);
    aboutMe.append(fragment);

}

const createProjectCard = (project) => {

    // Outer div element with background image
    const div = document.createElement('div');
    div.classList.add('projectCard');
    div.id = project.project_id;
    div.style.backgroundImage = `url(${project.card_image ?? './images/card_placeholder_bg.webp'})`


    // Inner text: title and short description
    const h4 = document.createElement('h4')
    h4.textContent = project.project_name;
    const p = document.createElement('p');
    p.textContent = project.short_description ?? '';

    // Add data attributes for quick access during spotlight load
    for (attr in project) {
        div.setAttribute(`data-${attr}`, project[attr]);
    }

    // Put it all together
    div.append(h4);
    div.append(p);
    return div;

}

const populateProjectList = async () => {

    // Load project data
    const projects = await loadJSONFile('./data/projectsData.json');

    // Create project cards and accumulate in document fragment
    const fragment = document.createDocumentFragment();
    for (project of projects) {
        fragment.append(createProjectCard(project));
    }

    // Add fragment to DOM
    const projectList = document.querySelector('#projectList');
    projectList.append(fragment);

    // Set default spotlight
    populateProjectSpotlight(projects[0].project_id);

}

const populateProjectSpotlight = (project_id) => {
    const projectDiv = document.querySelector(`#${project_id}`);
    const spotlightDiv = document.querySelector('#projectSpotlight');
    const spotlightTitleDiv = document.querySelector('#spotlightTitles');

    // Set the background image and remove the inner div
    spotlightDiv.style.backgroundImage = projectDiv.style.backgroundImage;
    spotlightDiv.style.backgroundSize = 'cover';

    // Inner content: name, description, and link
    const h3 = document.createElement('h3');
    h3.textContent = projectDiv.getAttribute('data-project_name');

    const p = document.createElement('p');
    p.textContent = (projectDiv.getAttribute('data-long_description') ?? projectDiv.getAttribute('data-short_description')) ?? '';

    const a = document.createElement('a');
    a.textContent = 'Click here to see more...';
    a.href = projectDiv.getAttribute('data-url') ?? '#';

    // Replace title div's child elements
    spotlightTitleDiv.replaceChildren(h3, p, a);
}

const handleProjectClick = (event) => {
    const card = event.target.closest('.projectCard');
    populateProjectSpotlight(card.id);
}

const handleNavClick = (event) => {

    const arrow = event.target;
    const projectList = document.querySelector('#projectList');

    // Determine scroll direction and extent
    const a = arrow.classList.contains('arrow-left') ? 0 : 1;
    let dir;
    let dist;

    // CSS sets 1024px as threshold for switching arrow directions
    if (matchMedia('(width >= 1024px)').matches) {
        dir = 'top';
        dist = projectList.firstElementChild.offsetHeight;
    } else {
        dir = 'left';
        dist = projectList.firstElementChild.offsetWidth;
    }

    // left arrow => negative distance
    projectList.scrollBy({[dir]: a ? dist : -dist});
}

const handleMessageAreaKeyUp = (event) => {

    const textarea = event.target;
    const charCounter = textarea.previousElementSibling;
    const errMsg = textarea.nextElementSibling;

    // Update character count
    const nchars = textarea.value.length;
    charCounter.textContent = `Characters: ${nchars}/${maxMsgLen}`;

    // Set error message as needed
    if (nchars > 300) {
        charCounter.style.color = 'var(--error)';
        errMsg.textContent = errors['msg_nchar'];
    } else if (badChars.test(textarea.value)) {
        charCounter.style.color = '';
        errMsg.textContent = errors['msg_chars'];
    }
    else {
        charCounter.style.color = '';
        errMsg.textContent = '';
    }
}

const handleEmailInputKeyUp = (event) => {
    const input = event.target;
    const errMsg = input.nextElementSibling;

    if (badChars.test(input.value)) {
        errMsg.textContent = errors['email_chars'];
    } else {
        errMsg.textContent = '';
    }
}

const handleEmailInputFocusOut = (event) => {
    const input = event.target;
    const errMsg = input.nextElementSibling;

    if (input.value.length == 0) {
        errMsg.textContent = errors['email_empty'];
    } else if (!validEmail.test(input.value)) {
        errMsg.textContent = errors['email_formt'];
    } else if (badChars.test(input.value)) {
        errMsg.textContent = errors['email_chars'];
    } else {
        errMsg.textContent = '';
    }
}

const handleFormSubmit = (event) => {
    event.preventDefault();
    const button = event.target;
    const form = button.parentElement;
    const textarea = form.querySelector('#contactMessage');
    const input = form.querySelector('#contactEmail');

    const errs = [];

    // Check email validity
    if (input.value.length == 0) {
        errs.push(errors['email_empty']);
    } else {

        if (!validEmail.test(input.value)) {
            errs.push(errors['email_formt']);
        }

        if (badChars.test(input.value)) {
            errs.push(errors['email_chars']);
        }
    }

    // Check message validity
    if (textarea.value.length == 0) {
        errs.push(errors['msg_empty']);
    } else {
        if (textarea.value.length > 300) {
            errs.push(errors['msg_nchar']);
        }

        if (badChars.test(textarea.value)) {
            errs.push(errors['msg_chars']);
        }
    }

    // Alert user
    if (errs.length > 0) {
        alert(errs.join('\n'));
    } else {
        alert('All fields are valid');
    }

}

const startup = async () => {

    // Populate empty elements
    populateAboutMe();
    populateProjectList();

    // Event handlers
    document.querySelector('#projectList').addEventListener('pointerdown', handleProjectClick);
    document.querySelector('.arrow-left').addEventListener('pointerdown', handleNavClick);
    document.querySelector('.arrow-right').addEventListener('pointerdown', handleNavClick);
    document.querySelector('#contactMessage').addEventListener('keyup', handleMessageAreaKeyUp);
    document.querySelector('#contactEmail').addEventListener('keyup', handleEmailInputKeyUp);
    document.querySelector('#contactEmail').addEventListener('focusout', handleEmailInputFocusOut);
    document.querySelector('#formsubmit').addEventListener('pointerdown', handleFormSubmit);
}

startup();
