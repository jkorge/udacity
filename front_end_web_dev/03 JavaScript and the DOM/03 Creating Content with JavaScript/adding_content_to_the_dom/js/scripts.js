const rebuildOtter = async() => {
    // Repaints
    const title = document.querySelector('body').firstElementChild;
    title.classList.add('title');

    const cardContainer = title.nextElementSibling;
    cardContainer.id = 'cardContainer';

    // New content - add all at once with a document fragment
    const newContent = document.createDocumentFragment();

    const profileCard = document.createElement('div');
    profileCard.classList.add('profileCard')

    const picFrame = document.createElement('div');
    picFrame.classList.add('picFrame');

    const profilePic = document.createElement('img');
    profilePic.setAttribute('src', './images/otter_profile.webp');
    profilePic.setAttribute('alt', 'profilePic');

    picFrame.append(profilePic);
    profileCard.append(picFrame);

    const userInfo = document.createElement('div');
    userInfo.classList.add('userInfo');

    const userInfoDetails = document.createElement('div');
    const userName = document.createElement('h2');
    const userBio = document.createElement('p');

    userName.textContent = 'Whiskers McOtter';
    userBio.textContent = `Hi! My name is Whiskers McOtter and I'm from Seattle, Washington. Some of my favorite things are Frappuccinos and fish.`
    userInfoDetails.append(userName, userBio);
    userInfo.append(userInfoDetails);

    const buttonWrapper = document.createElement('div');
    const statusButton = document.createElement('button');

    statusButton.classList.add('active');
    statusButton.textContent = 'Online Now!';
    buttonWrapper.append(statusButton);
    userInfo.append(buttonWrapper);

    profileCard.append(userInfo);
    newContent.append(profileCard);

    cardContainer.append(newContent);
}

rebuildOtter();

